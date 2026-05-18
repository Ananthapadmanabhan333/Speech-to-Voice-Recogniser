from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = structlog.get_logger(__name__)


class FusionError(Exception):
    """Raised when embedding fusion fails."""


@dataclass
class ModalityEmbeddings:
    """Container for modality-specific embeddings."""

    gesture: Optional[np.ndarray] = None  # e.g., (128,)
    speech: Optional[np.ndarray] = None  # e.g., (512,)
    facial: Optional[np.ndarray] = None  # e.g., (128,)
    context: Optional[np.ndarray] = None  # e.g., (256,)
    timestamps: Dict[str, float] = field(default_factory=dict)

    @property
    def available_modalities(self) -> List[str]:
        return [k for k in ["gesture", "speech", "facial", "context"] if getattr(self, k) is not None]


@dataclass
class FusedEmbedding:
    """Result of multimodal embedding fusion."""

    fused_vector: np.ndarray
    modality_weights: Dict[str, float]
    alignment_scores: Dict[str, float]
    confidence: float
    metadata: Dict = field(default_factory=dict)


class CrossModalProjection(nn.Module):
    """Project embeddings from one modality to another's dimension."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class CrossModalAttentionLayer(nn.Module):
    """Cross-modal attention between two modalities."""

    def __init__(self, query_dim: int, key_dim: int, value_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = value_dim // num_heads
        assert value_dim % num_heads == 0, "value_dim must be divisible by num_heads"

        self.query_proj = nn.Linear(query_dim, value_dim)
        self.key_proj = nn.Linear(key_dim, value_dim)
        self.value_proj = nn.Linear(value_dim, value_dim)
        self.output_proj = nn.Linear(value_dim, value_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        Q = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        out = self.output_proj(out)
        return out


class MultimodalEmbeddingFusion:
    """Multimodal embedding fusion with cross-modal attention and learned weights.

    Fuses gesture, speech, facial, and context embeddings into a unified
    representation. Handles missing modalities through learned masking.

    Architecture:
    - Projects all modalities to a common dimension
    - Applies cross-modal attention between all modality pairs
    - Late fusion with learned modality weights
    - Temporal alignment for time-offset modalities
    """

    def __init__(
        self,
        gesture_dim: int = 128,
        speech_dim: int = 512,
        facial_dim: int = 128,
        context_dim: int = 256,
        fused_dim: int = 512,
        num_attention_heads: int = 8,
        device: Optional[str] = None,
    ):
        """Initialize multimodal embedding fusion.

        Args:
            gesture_dim: Gesture embedding dimension.
            speech_dim: Speech embedding dimension.
            facial_dim: Facial embedding dimension.
            context_dim: Context embedding dimension.
            fused_dim: Output fused embedding dimension.
            num_attention_heads: Number of cross-modal attention heads.
            device: Device to run on.
        """
        self._fused_dim = fused_dim
        self._modality_dims = {
            "gesture": gesture_dim,
            "speech": speech_dim,
            "facial": facial_dim,
            "context": context_dim,
        }

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Projection layers for each modality to fused dimension
        self._projections = nn.ModuleDict({
            name: CrossModalProjection(dim, fused_dim)
            for name, dim in self._modality_dims.items()
        }).to(self._device)

        # Cross-modal attention between modality pairs
        self._cross_attentions = nn.ModuleDict({
            "gesture_speech": CrossModalAttentionLayer(fused_dim, fused_dim, fused_dim, num_attention_heads),
            "gesture_facial": CrossModalAttentionLayer(fused_dim, fused_dim, fused_dim, num_attention_heads),
            "speech_facial": CrossModalAttentionLayer(fused_dim, fused_dim, fused_dim, num_attention_heads),
            "context_all": CrossModalAttentionLayer(fused_dim, fused_dim, fused_dim, num_attention_heads),
        }).to(self._device)

        # Learned modality weights (temperature parameter)
        self._modality_logits = nn.Parameter(torch.zeros(4)).to(self._device)

        # Fusion layer
        self._fusion_layer = nn.Sequential(
            nn.Linear(fused_dim * 4, fused_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fused_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
        ).to(self._device)

        # Alignment module for temporal offsets
        self._temporal_align = nn.Linear(fused_dim, fused_dim).to(self._device)

        self._eval_mode = False

        logger.info(
            "multimodal_fusion_initialized",
            modality_dims=self._modality_dims,
            fused_dim=fused_dim,
            device=str(self._device),
        )

    def fuse_embeddings(
        self,
        gesture_emb: Optional[np.ndarray] = None,
        speech_emb: Optional[np.ndarray] = None,
        facial_emb: Optional[np.ndarray] = None,
        context_emb: Optional[np.ndarray] = None,
        timestamps: Optional[Dict[str, float]] = None,
    ) -> FusedEmbedding:
        """Fuse multimodal embeddings into a unified representation.

        Args:
            gesture_emb: Gesture embedding (gesture_dim,).
            speech_emb: Speech embedding (speech_dim,).
            facial_emb: Facial embedding (facial_dim,).
            context_emb: Context embedding (context_dim,).
            timestamps: Dict of modality -> timestamp for temporal alignment.

        Returns:
            FusedEmbedding with fused vector and metadata.

        Raises:
            FusionError: If fusion fails.
        """
        modalities = ModalityEmbeddings(
            gesture=gesture_emb,
            speech=speech_emb,
            facial=facial_emb,
            context=context_emb,
            timestamps=timestamps or {},
        )

        if not modalities.available_modalities:
            raise FusionError("No modalities provided for fusion")

        try:
            # Project all available modalities to common dimension
            projected = self._project_modalities(modalities)

            # Temporal alignment
            if timestamps and len(timestamps) > 1:
                projected = self._align_temporally(projected, timestamps)

            # Cross-modal attention
            attended = self._apply_cross_modal_attention(projected)

            # Learned modality weighting
            weighted, weights = self._apply_modality_weights(attended, modalities.available_modalities)

            # Concatenate and fuse
            fused = self._fuse_weighted_embeddings(weighted)

            # Compute alignment scores
            alignment_scores = self._compute_alignment_scores(projected)

            # Estimate overall confidence
            confidence = self._estimate_fusion_confidence(weights, alignment_scores)

            return FusedEmbedding(
                fused_vector=fused,
                modality_weights=weights,
                alignment_scores=alignment_scores,
                confidence=confidence,
                metadata={
                    "available_modalities": modalities.available_modalities,
                    "projected_dims": {k: v.shape[-1] for k, v in projected.items()},
                },
            )

        except Exception as e:
            logger.error("embedding_fusion_failed", error=str(e))
            raise FusionError(f"Multimodal fusion failed: {e}") from e

    def normalize_embedding(self, embedding: np.ndarray, norm_type: str = "l2") -> np.ndarray:
        """Normalize an embedding vector.

        Args:
            embedding: Input embedding.
            norm_type: Normalization type ('l2', 'unit', 'minmax').

        Returns:
            Normalized embedding.
        """
        if norm_type == "l2":
            norm = np.linalg.norm(embedding)
            return embedding / (norm + 1e-8)
        elif norm_type == "unit":
            return embedding / (np.max(np.abs(embedding)) + 1e-8)
        elif norm_type == "minmax":
            emb_min, emb_max = embedding.min(), embedding.max()
            if emb_max - emb_min > 1e-8:
                return (embedding - emb_min) / (emb_max - emb_min)
            return embedding
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

    def save(self, model_path: str) -> None:
        """Save fusion model weights.

        Args:
            model_path: Path to save checkpoint.
        """
        checkpoint = {
            "projections": self._projections.state_dict(),
            "cross_attentions": self._cross_attentions.state_dict(),
            "modality_logits": self._modality_logits,
            "fusion_layer": self._fusion_layer.state_dict(),
            "temporal_align": self._temporal_align.state_dict(),
        }
        torch.save(checkpoint, model_path)
        logger.info("fusion_model_saved", path=model_path)

    def load(self, model_path: str) -> None:
        """Load fusion model weights.

        Args:
            model_path: Path to checkpoint.
        """
        checkpoint = torch.load(model_path, map_location=self._device)
        self._projections.load_state_dict(checkpoint["projections"])
        self._cross_attentions.load_state_dict(checkpoint["cross_attentions"])
        self._modality_logits = nn.Parameter(checkpoint["modality_logits"].to(self._device))
        self._fusion_layer.load_state_dict(checkpoint["fusion_layer"])
        self._temporal_align.load_state_dict(checkpoint["temporal_align"])
        logger.info("fusion_model_loaded", path=model_path)

    def _project_modalities(
        self, modalities: ModalityEmbeddings
    ) -> Dict[str, torch.Tensor]:
        """Project available modalities to common dimension.

        Args:
            modalities: Input modality embeddings.

        Returns:
            Dict of modality -> projected tensor.
        """
        self._ensure_eval_mode()

        projected: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name in modalities.available_modalities:
                emb = getattr(modalities, name)
                if emb is not None:
                    tensor = torch.from_numpy(emb).float().to(self._device)
                    # Add batch and seq dims
                    tensor = tensor.unsqueeze(0).unsqueeze(1)  # (1, 1, dim)
                    proj = self._projections[name](tensor)  # (1, 1, fused_dim)
                    projected[name] = proj

        return projected

    def _align_temporally(
        self, projected: Dict[str, torch.Tensor], timestamps: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """Align embeddings temporally based on timestamps.

        Shifts embeddings by learned temporal offset to align them.

        Args:
            projected: Projected modality embeddings.
            timestamps: Modality -> timestamp mapping.

        Returns:
            Time-aligned embeddings.
        """
        if len(timestamps) < 2:
            return projected

        aligned = {}
        # Use earliest timestamp as reference
        ref_time = min(timestamps.values())
        ref_mod = [k for k, v in timestamps.items() if v == ref_time][0]

        for name, emb in projected.items():
            if name == ref_mod:
                aligned[name] = emb
            else:
                time_diff = timestamps.get(name, ref_time) - ref_time
                # Simple learned temporal shift
                shift = torch.tanh(self._temporal_align(emb)) * min(abs(time_diff), 1.0)
                aligned[name] = emb + shift

        return aligned

    def _apply_cross_modal_attention(
        self, projected: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply cross-modal attention between modality pairs.

        Args:
            projected: Projected modality embeddings.

        Returns:
            Attended embeddings.
        """
        attended = dict(projected)

        # Gesture -> Speech
        if "gesture" in projected and "speech" in projected:
            attn_gs = self._cross_attentions["gesture_speech"](
                projected["gesture"], projected["speech"], projected["speech"]
            )
            attended["gesture"] = attended.get("gesture", projected["gesture"]) + attn_gs

        # Gesture -> Facial
        if "gesture" in projected and "facial" in projected:
            attn_gf = self._cross_attentions["gesture_facial"](
                projected["gesture"], projected["facial"], projected["facial"]
            )
            attended["gesture"] = attended["gesture"] + attn_gf

        # Speech -> Facial
        if "speech" in projected and "facial" in projected:
            attn_sf = self._cross_attentions["speech_facial"](
                projected["speech"], projected["facial"], projected["facial"]
            )
            attended["speech"] = attended["speech"] + attn_sf

        # Context -> All
        if "context" in projected:
            all_modalities = [v for k, v in projected.items() if k != "context"]
            if all_modalities:
                combined = torch.cat(all_modalities, dim=1)
                attn_ca = self._cross_attentions["context_all"](
                    projected["context"], combined, combined
                )
                attended["context"] = attended["context"] + attn_ca

        return attended

    def _apply_modality_weights(
        self, attended: Dict[str, torch.Tensor], available: List[str]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Apply learned modality weighting.

        Args:
            attended: Attended modality embeddings.
            available: List of available modality names.

        Returns:
            (weighted_embeddings, weights_dict).
        """
        modality_order = ["gesture", "speech", "facial", "context"]
        available_logits = []

        for i, name in enumerate(modality_order):
            if name in available:
                available_logits.append(self._modality_logits[i])
            else:
                available_logits.append(torch.tensor(-float("inf"), device=self._device))

        logits_tensor = torch.stack(available_logits)
        weights = F.softmax(logits_tensor, dim=0)

        weight_dict = {}
        weighted = {}
        for i, name in enumerate(modality_order):
            weight = float(weights[i].cpu().item())
            weight_dict[name] = weight if name in available else 0.0
            if name in attended:
                weighted[name] = attended[name] * weights[i]

        return weighted, weight_dict

    def _fuse_weighted_embeddings(self, weighted: Dict[str, torch.Tensor]) -> np.ndarray:
        """Concatenate and fuse weighted embeddings.

        Args:
            weighted: Weighted modality embeddings.

        Returns:
            Fused embedding vector.
        """
        modality_order = ["gesture", "speech", "facial", "context"]
        emb_list = []

        for name in modality_order:
            if name in weighted:
                emb_list.append(weighted[name])
            else:
                emb_list.append(torch.zeros(1, 1, self._fused_dim, device=self._device))

        combined = torch.cat(emb_list, dim=-1)  # (1, 1, fused_dim * 4)
        fused = self._fusion_layer(combined)  # (1, 1, fused_dim)
        return fused.squeeze().cpu().numpy()

    def _compute_alignment_scores(
        self, projected: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute pairwise alignment scores between modalities.

        Uses cosine similarity between projected embeddings.

        Args:
            projected: Projected modality embeddings.

        Returns:
            Dict of pair_name -> alignment score.
        """
        scores: Dict[str, float] = {}
        names = list(projected.keys())

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                emb_i = projected[names[i]].squeeze()
                emb_j = projected[names[j]].squeeze()
                cos_sim = F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0))
                scores[f"{names[i]}_{names[j]}"] = float(cos_sim.cpu().item())

        return scores

    def _estimate_fusion_confidence(
        self, weights: Dict[str, float], alignment_scores: Dict[str, float]
    ) -> float:
        """Estimate overall confidence of the fused embedding.

        Based on number of modalities available, their weights, and alignment.

        Args:
            weights: Modality weights.
            alignment_scores: Pairwise alignment scores.

        Returns:
            Confidence score in [0, 1].
        """
        # More modalities = higher confidence
        available_count = sum(1 for w in weights.values() if w > 0.01)
        modality_factor = available_count / 4.0

        # Weight distribution: less entropy = higher confidence
        weight_vals = np.array([w for w in weights.values() if w > 0])
        if len(weight_vals) > 0:
            weight_entropy = -np.sum(weight_vals * np.log(weight_vals + 1e-8)) / np.log(len(weight_vals))
            weight_factor = 1.0 - weight_entropy
        else:
            weight_factor = 0.0

        # Alignment scores
        if alignment_scores:
            alignment_factor = float(np.mean(list(alignment_scores.values())))
            alignment_factor = max(0.0, min(1.0, (alignment_factor + 1) / 2))
        else:
            alignment_factor = 0.5

        confidence = 0.4 * modality_factor + 0.3 * weight_factor + 0.3 * alignment_factor
        return float(np.clip(confidence, 0.0, 1.0))

    def _ensure_eval_mode(self) -> None:
        """Ensure model is in evaluation mode."""
        if not self._eval_mode:
            self._projections.eval()
            self._cross_attentions.eval()
            self._fusion_layer.eval()
            self._temporal_align.eval()
            self._eval_mode = True
