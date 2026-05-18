"""
Multimodal attention network for Neurolink.

Provides the MultimodalAttentionNet with modality-specific feature extractors,
cross-modal multi-head attention, learnable temporal alignment, context-aware
attention masking, and hierarchical attention (frame -> segment -> session).
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AttentionNetConfig:
    """Configuration for MultimodalAttentionNet."""

    modality_dims: Dict[str, int] = field(
        default_factory=lambda: {
            "gesture": 128,
            "speech": 768,
            "facial": 512,
            "text": 512,
        }
    )
    d_model: int = 512
    nhead: int = 8
    num_cross_attn_layers: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # Temporal alignment
    max_temporal_offset: int = 10
    num_temporal_offsets: int = 21  # -10 to +10

    # Hierarchical attention
    frames_per_segment: int = 16
    segments_per_session: int = 8

    # Output dimensions
    num_intents: int = 20
    num_emotions: int = 8

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ModalityFeatureExtractor(nn.Module):
    """Extract features from a single modality."""

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.input_proj(x)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm(x + attn_out)
        return x


class CrossModalMultiHeadAttention(nn.Module):
    """Cross-modal attention where query attends to key/value from another modality."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(key)
        attn_out, attn_weights = self.attention(
            q, kv, value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        query = query + attn_out
        query = query + self.ffn(self.norm_ffn(query))
        return query, attn_weights


class TemporalAlignmentModule(nn.Module):
    """Learnable temporal offsets for aligning cross-modal sequences.

    Uses a small MLP to predict frame-wise offsets that align one modality
    to another in time.
    """

    def __init__(
        self,
        d_model: int,
        max_offset: int = 10,
        num_offsets: int = 21,
    ):
        super().__init__()
        self.max_offset = max_offset
        self.num_offsets = num_offsets
        self.offset_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align source to target in temporal dimension.

        For each frame in source, predict an offset and sample from target.

        Args:
            source: (batch, src_len, d_model)
            target: (batch, tgt_len, d_model)

        Returns:
            aligned_source: (batch, src_len, d_model)
            offsets: (batch, src_len) predicted offsets
        """
        batch, src_len, _ = source.shape
        tgt_len = target.size(1)

        # Predict offset for each source frame
        offset_logits = self.offset_predictor(source).squeeze(-1)  # (batch, src_len)
        offsets = torch.tanh(offset_logits) * self.max_offset  # (batch, src_len)

        # Construct interpolation grid
        grid = torch.arange(src_len, device=source.device).float()
        grid = grid.unsqueeze(0).expand(batch, -1)  # (batch, src_len)
        aligned_indices = grid + offsets  # (batch, src_len)

        # Clamp to valid range
        aligned_indices = aligned_indices.clamp(0, tgt_len - 1)

        # Gather from target
        aligned = torch.zeros_like(source)
        for b in range(batch):
            for s in range(src_len):
                idx = int(aligned_indices[b, s].round().item())
                aligned[b, s] = target[b, idx]

        return aligned, offsets


class ContextAwareAttentionMask(nn.Module):
    """Generate context-aware attention masks for cross-modal attention.

    Masks out positions that are semantically irrelevant based on learned
    context scores.
    """

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.context_scorer = nn.Linear(d_model, nhead)
        self.nhead = nhead

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        base_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate attention mask.

        Args:
            query: (batch, q_len, d_model)
            key: (batch, kv_len, d_model)
            base_mask: Optional (batch, kv_len) padding mask.

        Returns:
            attn_mask: (batch * nhead, q_len, kv_len)
        """
        batch, q_len, _ = query.shape
        kv_len = key.size(1)

        # Context scores for each head
        context_scores = self.context_scorer(query)  # (batch, q_len, nhead)
        context_scores = context_scores.permute(0, 2, 1)  # (batch, nhead, q_len)

        # Expand to full mask
        attn_mask = context_scores.unsqueeze(-1).expand(
            batch, self.nhead, q_len, kv_len
        )  # (batch, nhead, q_len, kv_len)

        # Apply base padding mask
        if base_mask is not None:
            padding_mask = base_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, kv_len)
            attn_mask = attn_mask.masked_fill(padding_mask, float("-inf"))

        # Reshape for MultiheadAttention
        attn_mask = attn_mask.reshape(
            batch * self.nhead, q_len, kv_len
        )

        return attn_mask


class HierarchicalAttention(nn.Module):
    """Hierarchical attention across frame, segment, and session levels.

    Computes attention at three levels:
        1. Frame-level: within each segment
        2. Segment-level: between segments in a session
        3. Session-level: across the full session
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        frames_per_segment: int = 16,
        segments_per_session: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.frames_per_segment = frames_per_segment
        self.segments_per_session = segments_per_session

        self.frame_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.segment_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.session_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.segment_pool = nn.AdaptiveAvgPool1d(1)
        self.session_pool = nn.AdaptiveAvgPool1d(1)

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Apply hierarchical attention.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            attended: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        # 1. Frame-level attention within each segment
        num_segments = seq_len // self.frames_per_segment
        if num_segments == 0:
            num_segments = 1
        seg_len = min(self.frames_per_segment, seq_len)

        frame_attended = torch.zeros_like(x)
        for s in range(num_segments):
            start = s * seg_len
            end = min(start + seg_len, seq_len)
            if start >= seq_len:
                break
            seg = x[:, start:end, :]  # (batch, seg_len, d_model)
            attn_out, _ = self.frame_attn(seg, seg, seg)
            frame_attended[:, start:end, :] = seg + attn_out

        # 2. Segment-level attention
        # Pool each segment to a single vector
        segment_reprs = []
        for s in range(num_segments):
            start = s * seg_len
            end = min(start + seg_len, seq_len)
            if start >= seq_len:
                break
            seg = frame_attended[:, start:end, :]
            pooled = seg.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
            segment_reprs.append(pooled)

        if segment_reprs:
            segments = torch.cat(segment_reprs, dim=1)  # (batch, num_seg, d_model)
            seg_attended, _ = self.segment_attn(segments, segments, segments)

            # Expand segment attention back to frames
            for s in range(num_segments):
                if s >= seg_attended.size(1):
                    break
                start = s * seg_len
                end = min(start + seg_len, seq_len)
                frame_attended[:, start:end, :] = (
                    frame_attended[:, start:end, :] + seg_attended[:, s:s+1, :]
                )

        # 3. Session-level attention
        session_pooled = frame_attended.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        session_attended, _ = self.session_attn(
            frame_attended, session_pooled.expand(-1, seq_len, -1),
            session_pooled.expand(-1, seq_len, -1),
        )
        frame_attended = frame_attended + session_attended

        return self.norm(frame_attended)


class MultimodalAttentionNet(nn.Module):
    """Multimodal attention network with full cross-modal and hierarchical attention.

    Architecture:
        1. Modality-specific feature extractors
        2. Temporal alignment (learnable offsets between modalities)
        3. Cross-modal multi-head attention with context-aware masking
        4. Hierarchical attention (frame -> segment -> session)
        5. Prediction heads
    """

    def __init__(self, config: AttentionNetConfig):
        super().__init__()
        self.config = config
        self.modality_names = list(config.modality_dims.keys())

        # 1. Modality-specific feature extractors
        self.feature_extractors = nn.ModuleDict()
        for name, dim in config.modality_dims.items():
            self.feature_extractors[name] = ModalityFeatureExtractor(
                dim, config.d_model, config.dropout
            )

        # 2. Temporal alignment modules (for each pair of modalities)
        self.temporal_alignments = nn.ModuleDict()
        for i, name in enumerate(self.modality_names):
            for j, other in enumerate(self.modality_names):
                if i != j:
                    self.temporal_alignments[f"{name}_to_{other}"] = (
                        TemporalAlignmentModule(
                            config.d_model,
                            config.max_temporal_offset,
                            config.num_temporal_offsets,
                        )
                    )

        # 3. Cross-modal attention layers
        self.cross_modal_layers = nn.ModuleList()
        self.context_masks = nn.ModuleList()
        for _ in range(config.num_cross_attn_layers):
            self.cross_modal_layers.append(
                CrossModalMultiHeadAttention(
                    config.d_model, config.nhead, config.dropout
                )
            )
            self.context_masks.append(
                ContextAwareAttentionMask(config.d_model, config.nhead)
            )

        # 4. Hierarchical attention
        self.hierarchical_attention = HierarchicalAttention(
            config.d_model,
            config.nhead,
            config.frames_per_segment,
            config.segments_per_session,
            config.dropout,
        )

        # 5. Fusion layer
        self.fusion_proj = nn.Sequential(
            nn.Linear(config.d_model * len(self.modality_names), config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # 6. Prediction heads
        self.intent_head = nn.Linear(config.d_model, config.num_intents)
        self.emotion_head = nn.Linear(config.d_model, config.num_emotions)

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        padding_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            modality_inputs: {name: (batch, seq_len, feat_dim)} dict.
            padding_masks: Optional {name: (batch, seq_len)} bool mask.

        Returns:
            Dict with 'intent_logits', 'emotion_logits', 'fused_embedding'.
        """
        # 1. Extract features for each modality
        features: Dict[str, torch.Tensor] = {}
        for name in self.modality_names:
            if name in modality_inputs:
                mask = padding_masks.get(name) if padding_masks else None
                features[name] = self.feature_extractors[name](
                    modality_inputs[name], mask
                )
            else:
                device = next(self.parameters()).device
                features[name] = torch.zeros(
                    1, 1, self.config.d_model, device=device
                )

        # 2. Temporal alignment (align all modalities to the first)
        aligned: Dict[str, torch.Tensor] = {}
        first_mod = self.modality_names[0]
        aligned[first_mod] = features[first_mod]

        for name in self.modality_names[1:]:
            if name in features:
                key = f"{name}_to_{first_mod}"
                if key in self.temporal_alignments:
                    aligned[name], offsets = self.temporal_alignments[key](
                        features[name], features[first_mod]
                    )
                else:
                    aligned[name] = features[name]
            else:
                aligned[name] = features[name]

        # 3. Cross-modal attention with context-aware masking
        modality_list = [aligned[name] for name in self.modality_names]
        for cross_attn, ctx_mask in zip(
            self.cross_modal_layers, self.context_masks
        ):
            new_list = []
            for i, feat in enumerate(modality_list):
                for j, other in enumerate(modality_list):
                    if i != j:
                        attn_mask = ctx_mask(feat, other)
                        feat, attn_weights = cross_attn(
                            feat, other, other,
                            attn_mask=attn_mask,
                        )
                new_list.append(feat)
            modality_list = new_list

        # 4. Hierarchical attention on each modality
        for i in range(len(modality_list)):
            modality_list[i] = self.hierarchical_attention(modality_list[i])

        # 5. Fuse all modalities
        # Global average pooling
        pooled = [m.mean(dim=1) for m in modality_list]  # (batch, d_model)
        concat = torch.cat(pooled, dim=-1)
        fused = self.fusion_proj(concat)

        # 6. Predictions
        return {
            "intent_logits": self.intent_head(fused),
            "emotion_logits": self.emotion_head(fused),
            "fused_embedding": fused,
        }
