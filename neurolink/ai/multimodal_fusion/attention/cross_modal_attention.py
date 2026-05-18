from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog

logger = structlog.get_logger(__name__)


class CrossModalAttention(nn.Module):
    """Multi-head cross-modal attention with learnable modality weighting.

    Supports cross-attention between gesture-speech, gesture-emotion,
    speech-emotion, and all combinations. Uses efficient scaled dot-product
    attention with optional FlashAttention compatibility.

    Architecture:
    - Separate Q, K, V projections per modality pair
    - Learnable modality weighting via gating mechanism
    - Context-aware attention masking
    - Residual connections with layer normalization
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_flash_attention: bool = False,
    ):
        """Initialize cross-modal attention module.

        Args:
            modality_dims: Dict mapping modality name to embedding dimension.
                Example: {"gesture": 128, "speech": 512, "emotion": 128}.
            d_model: Common attention dimension.
            num_heads: Number of attention heads (must divide d_model).
            dropout: Dropout rate.
            use_flash_attention: Enable FlashAttention if available.
        """
        super().__init__()
        self._modality_names = list(modality_dims.keys())
        self._modality_dims = modality_dims
        self._d_model = d_model
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads
        self._use_flash_attention = use_flash_attention

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        # Projections for each modality to Q, K, V
        self._q_projections = nn.ModuleDict({
            name: nn.Linear(dim, d_model)
            for name, dim in modality_dims.items()
        })
        self._k_projections = nn.ModuleDict({
            name: nn.Linear(dim, d_model)
            for name, dim in modality_dims.items()
        })
        self._v_projections = nn.ModuleDict({
            name: nn.Linear(dim, d_model)
            for name, dim in modality_dims.items()
        })

        # Output projections per modality
        self._output_projections = nn.ModuleDict({
            name: nn.Linear(d_model, dim)
            for name, dim in modality_dims.items()
        })

        # Modality gating network
        self._modality_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Layer norms
        self._layer_norms = nn.ModuleDict({
            name: nn.LayerNorm(dim)
            for name, dim in modality_dims.items()
        })

        self._dropout = nn.Dropout(dropout)
        self._scale = math.sqrt(self._head_dim)

        logger.info(
            "cross_modal_attention_initialized",
            modalities=self._modality_names,
            d_model=d_model,
            num_heads=num_heads,
            use_flash=use_flash_attention,
        )

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[Dict[str, torch.Tensor]] = None,
        return_attention_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Apply cross-modal attention between all modality pairs.

        For each pair (A, B):
        1. A attends to B using multi-head cross-attention
        2. Result is gated and added to A via residual
        3. Layer normalization is applied

        Args:
            modality_inputs: Dict mapping modality name -> tensor (batch, seq_len, dim).
            attention_mask: Optional dict of modality -> mask (batch, seq_len).
            return_attention_weights: If True, returns attention weights dict.

        Returns:
            Dict of updated modality embeddings.
            If return_attention_weights, returns (outputs, attention_weights).
        """
        device = next(self.parameters()).device
        batch_size = next(iter(modality_inputs.values())).size(0)

        # Validate and pad/truncate sequences
        inputs = self._prepare_inputs(modality_inputs, device)

        # Project all modalities to Q, K, V
        projections = {}
        for name in self._modality_names:
            if name in inputs:
                projections[name] = {
                    "q": self._q_projections[name](inputs[name]),
                    "k": self._k_projections[name](inputs[name]),
                    "v": self._v_projections[name](inputs[name]),
                }

        # Apply cross-attention for each pair
        outputs = {name: inputs[name].clone() for name in inputs}
        attention_weights: Dict[str, torch.Tensor] = {}

        for query_name in self._modality_names:
            if query_name not in projections:
                continue

            for key_name in self._modality_names:
                if key_name not in projections or key_name == query_name:
                    continue

                q = projections[query_name]["q"]
                k = projections[key_name]["k"]
                v = projections[key_name]["v"]

                # Reshape for multi-head attention
                q = self._reshape_for_attention(q, batch_size)  # (batch, heads, seq_q, head_dim)
                k = self._reshape_for_attention(k, batch_size)
                v = self._reshape_for_attention(v, batch_size)

                # Compute attention scores
                if self._use_flash_attention and hasattr(F, "scaled_dot_product_attention"):
                    # FlashAttention path
                    attn_mask = self._get_pair_mask(attention_mask, query_name, key_name, device)
                    attn_output = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attn_mask,
                        dropout_p=self._dropout.p if self.training else 0.0,
                        is_causal=False,
                    )
                    attn_weights = None
                else:
                    # Standard attention
                    scores = torch.matmul(q, k.transpose(-2, -1)) / self._scale
                    scores = self._apply_attention_mask(scores, attention_mask, query_name, key_name)
                    attn_weights = F.softmax(scores, dim=-1)
                    attn_weights = self._dropout(attn_weights)
                    attn_output = torch.matmul(attn_weights, v)

                # Reshape back
                attn_output = attn_output.transpose(1, 2).contiguous().view(
                    batch_size, -1, self._d_model
                )

                # Project back to original dimension
                attn_output = self._output_projections[key_name](attn_output)

                # Compute gate for residual connection
                gate_input = torch.cat([outputs[query_name], attn_output], dim=-1)
                gate = self._modality_gate(gate_input)

                # Residual with gating
                outputs[query_name] = outputs[query_name] + gate * attn_output

                if return_attention_weights and attn_weights is not None:
                    pair_key = f"{query_name}_to_{key_name}"
                    attention_weights[pair_key] = attn_weights.detach()

        # Apply layer normalization
        for name in outputs:
            outputs[name] = self._layer_norms[name](outputs[name])

        if return_attention_weights:
            return outputs, attention_weights
        return outputs

    def get_modality_weights(
        self, modality_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Get learned modality importance weights.

        Computes importance scores based on the gating network responses.

        Args:
            modality_inputs: Input modality embeddings.

        Returns:
            Dict of modality name -> importance weight.
        """
        device = next(self.parameters()).device
        inputs = self._prepare_inputs(modality_inputs, device)

        weights: Dict[str, float] = {}
        for name in inputs:
            # Average gate values as importance proxy
            gate_input = torch.cat([inputs[name], inputs[name]], dim=-1)
            gate = self._modality_gate(gate_input)
            weights[name] = float(gate.mean().cpu().item())

        # Normalize
        total = sum(weights.values()) + 1e-8
        weights = {k: v / total for k, v in weights.items()}
        return weights

    def _reshape_for_attention(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Reshape tensor for multi-head attention.

        Args:
            x: (batch, seq_len, d_model)
            batch_size: Batch size.

        Returns:
            (batch, num_heads, seq_len, head_dim)
        """
        seq_len = x.size(1)
        return x.view(batch_size, seq_len, self._num_heads, self._head_dim).transpose(1, 2)

    def _prepare_inputs(
        self, modality_inputs: Dict[str, torch.Tensor], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs: move to device and validate.

        Args:
            modality_inputs: Raw input dict.
            device: Target device.

        Returns:
            Prepared inputs on correct device.
        """
        inputs = {}
        for name in self._modality_names:
            if name in modality_inputs:
                t = modality_inputs[name].to(device)
                if t.dim() == 2:
                    t = t.unsqueeze(1)  # Add sequence dimension
                inputs[name] = t
        return inputs

    def _get_pair_mask(
        self,
        attention_mask: Optional[Dict[str, torch.Tensor]],
        query_name: str,
        key_name: str,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Get attention mask for a query-key pair.

        Args:
            attention_mask: Per-modality masks.
            query_name: Query modality name.
            key_name: Key modality name.
            device: Target device.

        Returns:
            Combined attention mask or None.
        """
        if attention_mask is None:
            return None

        q_mask = attention_mask.get(query_name)
        k_mask = attention_mask.get(key_name)

        if q_mask is None or k_mask is None:
            return None

        # Create pairwise mask
        q_mask = q_mask.to(device).float()  # (batch, seq_q)
        k_mask = k_mask.to(device).float()  # (batch, seq_k)

        # (batch, 1, seq_q, seq_k)
        pair_mask = q_mask.unsqueeze(2) * k_mask.unsqueeze(1)
        pair_mask = pair_mask.unsqueeze(1)  # (batch, 1, seq_q, seq_k)

        # Convert to additive mask (0 = keep, -inf = mask)
        pair_mask = (1.0 - pair_mask) * float("-inf")
        return pair_mask

    def _apply_attention_mask(
        self,
        scores: torch.Tensor,
        attention_mask: Optional[Dict[str, torch.Tensor]],
        query_name: str,
        key_name: str,
    ) -> torch.Tensor:
        """Apply attention mask to raw attention scores.

        Args:
            scores: Raw attention scores.
            attention_mask: Dict of masks per modality.
            query_name: Query name.
            key_name: Key name.

        Returns:
            Masked scores.
        """
        if attention_mask is None:
            return scores

        q_mask = attention_mask.get(query_name)
        k_mask = attention_mask.get(key_name)

        if q_mask is not None:
            q_mask = q_mask.to(scores.device).bool()
            scores = scores.masked_fill(~q_mask.unsqueeze(1).unsqueeze(-1), float("-inf"))
        if k_mask is not None:
            k_mask = k_mask.to(scores.device).bool()
            scores = scores.masked_fill(~k_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        return scores


class ModalityWeightedFusion(nn.Module):
    """Learnable weighted fusion of multiple modalities.

    Computes per-modality weights using a simple gating mechanism
    and fuses all modalities into a single output.
    """

    def __init__(self, modality_dims: Dict[str, int], output_dim: int):
        super().__init__()
        self._modality_names = list(modality_dims.keys())

        # Projection to common dimension
        self._projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })

        # Modality weight network
        self._weight_net = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4),
            nn.ReLU(),
            nn.Linear(output_dim // 4, 1),
            nn.Sigmoid(),
        )

        self._output_norm = nn.LayerNorm(output_dim)

    def forward(
        self, modality_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Fuse modalities with learned weights.

        Args:
            modality_inputs: Dict of modality name -> tensor (batch, dim).

        Returns:
            Fused output (batch, output_dim).
        """
        projected = {}
        weights = {}

        for name in self._modality_names:
            if name in modality_inputs:
                proj = self._projections[name](modality_inputs[name])
                projected[name] = proj
                weights[name] = self._weight_net(proj)

        # Normalize weights
        weight_sum = sum(weights.values()) + 1e-8
        fused = sum(
            projected[name] * (weights[name] / weight_sum)
            for name in projected
        )

        return self._output_norm(fused)
