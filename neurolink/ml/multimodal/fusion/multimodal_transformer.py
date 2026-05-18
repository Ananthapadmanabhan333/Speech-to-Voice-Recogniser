"""
Multimodal fusion transformer for Neurolink.

Provides the MultimodalTransformer that fuses multiple modalities (gesture,
speech, facial, text) using separate encoders, cross-modal attention layers,
and a late fusion gating mechanism. Supports multiple output heads for
intent, emotion, urgency, and next-action prediction.
"""

import json
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MultimodalConfig:
    """Configuration for the MultimodalTransformer."""

    # Modality dimensions
    gesture_dim: int = 128
    speech_dim: int = 768  # wav2vec2 features
    facial_dim: int = 512  # ResNet features
    text_dim: int = 512  # BERT features

    # Model dimensions
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 4
    num_cross_attn_layers: int = 3
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = "relu"

    # Output heads
    num_intents: int = 20
    num_emotions: int = 8
    num_urgency_levels: int = 5
    num_actions: int = 15

    # Fusion
    fusion_hidden_dim: int = 256
    gating_hidden_dim: int = 128
    modality_dropout: float = 0.2  # probability to drop a modality during training

    # Modality names
    modality_names: List[str] = field(
        default_factory=lambda: ["gesture", "speech", "facial", "text"]
    )

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ModalityProjection(nn.Module):
    """Project a modality input to the common d_model dimension."""

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CrossModalAttention(nn.Module):
    """Cross-modal multi-head attention layer.

    Allows query modality q to attend to key/value modality kv.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)

        attn_out, _ = self.cross_attn(
            q, kv, kv, key_padding_mask=key_padding_mask
        )
        query = query + self.dropout(attn_out)
        query = query + self.dropout(self.ffn(self.norm_ffn(query)))
        return query


class GatingMechanism(nn.Module):
    """Learnable gating mechanism for late fusion of modality representations."""

    def __init__(
        self,
        num_modalities: int,
        d_model: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * num_modalities, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1),
        )

    def forward(
        self, modality_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute weighted fusion of modality features.

        Args:
            modality_features: List of (batch, d_model) tensors per modality.

        Returns:
            Fused representation: (batch, d_model).
        """
        # Global average pooling across sequence dim if needed
        pooled = []
        for feat in modality_features:
            if feat.dim() == 3:
                feat = feat.mean(dim=1)  # (batch, d_model)
            pooled.append(feat)

        concat = torch.cat(pooled, dim=-1)  # (batch, num_mod * d_model)
        weights = self.gate_network(concat)  # (batch, num_mod)

        fused = torch.zeros_like(pooled[0])
        for i, feat in enumerate(pooled):
            fused = fused + weights[:, i : i + 1] * feat

        return fused, weights


class ModalityEncoder(nn.Module):
    """Encoder for a single modality with self-attention."""

    def __init__(self, config: MultimodalConfig, input_dim: int):
        super().__init__()
        self.projection = ModalityProjection(input_dim, config.d_model, config.dropout)
        self.pos_encoding = PositionalEncoding(config.d_model, dropout=config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.projection(x)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return x


class OutputHead(nn.Module):
    """Prediction head for a specific output task."""

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling if sequential
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.head(x)


class MultimodalTransformer(nn.Module):
    """Multimodal fusion transformer with separate encoders and cross-modal attention.

    Supports configurable modality sets, cross-modal fusion via attention,
    gated late fusion, and multiple output heads.

    Architecture:
        Modality-specific encoders -> Cross-modal attention stack ->
        Gated late fusion -> Output heads (intent, emotion, urgency, next_action)
    """

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.modality_names = config.modality_names

        # Modality dimensions mapping
        self._modality_dims = {
            "gesture": config.gesture_dim,
            "speech": config.speech_dim,
            "facial": config.facial_dim,
            "text": config.text_dim,
        }

        # Separate encoders for each modality
        self.encoders = nn.ModuleDict()
        for name in self.modality_names:
            input_dim = self._modality_dims.get(name, config.d_model)
            self.encoders[name] = ModalityEncoder(config, input_dim)

        # Cross-modal attention layers
        self.cross_attn_layers = nn.ModuleList()
        for _ in range(config.num_cross_attn_layers):
            self.cross_attn_layers.append(
                CrossModalAttention(config.d_model, config.nhead, config.dropout)
            )

        # Gating mechanism for late fusion
        self.gating = GatingMechanism(
            len(self.modality_names), config.d_model, config.gating_hidden_dim
        )

        # Prediction heads
        self.intent_head = OutputHead(config.d_model, config.num_intents)
        self.emotion_head = OutputHead(config.d_model, config.num_emotions)
        self.urgency_head = OutputHead(config.d_model, config.num_urgency_levels)
        self.action_head = OutputHead(config.d_model, config.num_actions)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _apply_modality_dropout(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        training: bool,
    ) -> Dict[str, torch.Tensor]:
        """Randomly drop modalities during training for robustness."""
        if not training:
            return modality_inputs

        result = dict(modality_inputs)
        drop_prob = self.config.modality_dropout

        for name in list(result.keys()):
            if torch.rand(1).item() < drop_prob:
                # Replace with zeros of same shape
                result[name] = torch.zeros_like(result[name])

        # Ensure at least one modality remains
        if all((v == 0).all() for v in result.values()):
            # Restore the first modality
            first_name = list(modality_inputs.keys())[0]
            result[first_name] = modality_inputs[first_name]

        return result

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        padding_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the multimodal transformer.

        Args:
            modality_inputs: Dict mapping modality name to (batch, seq_len, feat_dim).
            padding_masks: Dict mapping modality name to (batch, seq_len) bool mask
                          (True = masked/padded position).

        Returns:
            Dict with keys: 'intent_logits', 'emotion_logits', 'urgency_logits',
                           'action_logits', 'fused_embedding', 'gate_weights'
        """
        # Modality dropout for robustness
        inputs = self._apply_modality_dropout(modality_inputs, self.training)

        # Encode each modality
        encoded: Dict[str, torch.Tensor] = {}
        mask: Optional[torch.Tensor] = None
        for name in self.modality_names:
            if name in inputs:
                x = inputs[name]
                m = padding_masks.get(name) if padding_masks else None
                encoded[name] = self.encoders[name](x, key_padding_mask=m)
            else:
                # Modality not provided: use zeros
                device = next(self.parameters()).device
                encoded[name] = torch.zeros(
                    1, 1, self.config.d_model, device=device
                )

        # Cross-modal attention: each modality attends to all others
        modality_list = list(encoded.values())
        for cross_attn in self.cross_attn_layers:
            new_modality_list = []
            for i, feat in enumerate(modality_list):
                # Attend to all other modalities sequentially
                for j, other in enumerate(modality_list):
                    if i != j:
                        feat = cross_attn(feat, other)
                new_modality_list.append(feat)
            modality_list = new_modality_list

        # Gated late fusion
        fused_embedding, gate_weights = self.gating(modality_list)

        # Output heads
        outputs = {
            "intent_logits": self.intent_head(fused_embedding),
            "emotion_logits": self.emotion_head(fused_embedding),
            "urgency_logits": self.urgency_head(fused_embedding),
            "action_logits": self.action_head(fused_embedding),
            "fused_embedding": fused_embedding,
            "gate_weights": gate_weights,
        }

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss.

        Args:
            outputs: Dict from forward().
            targets: Dict with keys 'intent', 'emotion', 'urgency', 'action'.
            weights: Optional per-task loss weights.

        Returns:
            Dict with individual losses and 'total_loss'.
        """
        weight_map = weights or {
            "intent": 1.0,
            "emotion": 1.0,
            "urgency": 0.5,
            "action": 1.0,
        }

        losses = {}
        total = 0.0

        tasks = [
            ("intent", "intent_logits"),
            ("emotion", "emotion_logits"),
            ("urgency", "urgency_logits"),
            ("action", "action_logits"),
        ]

        for task_name, logits_key in tasks:
            if logits_key in outputs and task_name in targets:
                loss = F.cross_entropy(
                    outputs[logits_key], targets[task_name]
                )
                w = weight_map.get(task_name, 1.0)
                losses[f"{task_name}_loss"] = loss
                total += loss * w

        losses["total_loss"] = total
        return losses

    def save_pretrained(self, path: Union[str, Path]):
        """Save model weights and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        config_dict = {
            k: v for k, v in self.config.__dict__.items() if not k.startswith("_")
        }
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        logger.info(f"MultimodalTransformer saved to {path}")

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> "MultimodalTransformer":
        """Load model from saved weights."""
        path = Path(path)
        with open(path / "config.json", "r") as f:
            config_dict = json.load(f)
        config = MultimodalConfig(**config_dict)
        model = cls(config)
        model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
        logger.info(f"MultimodalTransformer loaded from {path}")
        return model
