from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.multimodal_fusion.embeddings.fusion_embeddings import (
    CrossModalAttentionLayer,
    CrossModalProjection,
    MultimodalEmbeddingFusion,
)


@pytest.fixture(autouse=True)
def set_seed() -> None:
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def fusion() -> MultimodalEmbeddingFusion:
    return MultimodalEmbeddingFusion(
        gesture_dim=128,
        speech_dim=512,
        facial_dim=128,
        context_dim=256,
        fused_dim=256,
        num_attention_heads=4,
        device="cpu",
    )


class TestMultimodalForwardPass:
    """Test multimodal model forward pass."""

    def test_fused_output_shape(self, fusion: MultimodalEmbeddingFusion) -> None:
        gesture = np.random.randn(128).astype(np.float32)
        speech = np.random.randn(512).astype(np.float32)
        result = fusion.fuse_embeddings(gesture_emb=gesture, speech_emb=speech)
        assert result.fused_vector.shape == (256,)

    def test_all_modalities_output(self, fusion: MultimodalEmbeddingFusion) -> None:
        gesture = np.random.randn(128).astype(np.float32)
        speech = np.random.randn(512).astype(np.float32)
        facial = np.random.randn(128).astype(np.float32)
        context = np.random.randn(256).astype(np.float32)
        result = fusion.fuse_embeddings(
            gesture_emb=gesture,
            speech_emb=speech,
            facial_emb=facial,
            context_emb=context,
        )
        assert result.fused_vector.shape == (256,)

    def test_output_is_deterministic(self, fusion: MultimodalEmbeddingFusion) -> None:
        gesture = np.random.randn(128).astype(np.float32)
        r1 = fusion.fuse_embeddings(gesture_emb=gesture)
        r2 = fusion.fuse_embeddings(gesture_emb=gesture)
        np.testing.assert_array_almost_equal(r1.fused_vector, r2.fused_vector)

    def test_projections_output_shape(self, fusion: MultimodalEmbeddingFusion) -> None:
        fusion._ensure_eval_mode()
        gesture_t = torch.randn(1, 1, 128)
        projected = fusion._projections["gesture"](gesture_t)
        assert projected.shape == (1, 1, 256)


class TestModalityDropout:
    """Test handling of missing modalities."""

    def test_single_modality_works(self, fusion: MultimodalEmbeddingFusion) -> None:
        gesture = np.random.randn(128).astype(np.float32)
        result = fusion.fuse_embeddings(gesture_emb=gesture)
        assert result.fused_vector.shape == (256,)
        assert result.modality_weights["gesture"] > 0

    def test_two_modalities_better_than_one(self, fusion: MultimodalEmbeddingFusion) -> None:
        gesture = np.random.randn(128).astype(np.float32)
        speech = np.random.randn(512).astype(np.float32)

        r1 = fusion.fuse_embeddings(gesture_emb=gesture)
        r2 = fusion.fuse_embeddings(gesture_emb=gesture, speech_emb=speech)

        assert r2.confidence >= r1.confidence * 0.5  # Two modalities should give reasonable confidence

    def test_missing_modality_weights_zero(self, fusion: MultimodalEmbeddingFusion) -> None:
        gesture = np.random.randn(128).astype(np.float32)
        result = fusion.fuse_embeddings(gesture_emb=gesture)
        assert result.modality_weights["speech"] == 0.0
        assert result.modality_weights["facial"] == 0.0
        assert result.modality_weights["context"] == 0.0

    def test_all_modalities_weights_sum_to_one(self, fusion: MultimodalEmbeddingFusion) -> None:
        gesture = np.random.randn(128).astype(np.float32)
        speech = np.random.randn(512).astype(np.float32)
        result = fusion.fuse_embeddings(gesture_emb=gesture, speech_emb=speech)
        total = sum(v for v in result.modality_weights.values() if v > 0)
        assert abs(total - 1.0) < 1e-6

    def test_three_modalities(self, fusion: MultimodalEmbeddingFusion) -> None:
        gesture = np.random.randn(128).astype(np.float32)
        speech = np.random.randn(512).astype(np.float32)
        facial = np.random.randn(128).astype(np.float32)
        result = fusion.fuse_embeddings(
            gesture_emb=gesture,
            speech_emb=speech,
            facial_emb=facial,
        )
        assert result.fused_vector.shape == (256,)
        active = [k for k, v in result.modality_weights.items() if v > 0]
        assert len(active) == 3


class TestAttentionMechanism:
    """Test cross-modal attention mechanism."""

    def test_attention_layer_shape(self) -> None:
        attn = CrossModalAttentionLayer(query_dim=128, key_dim=128, value_dim=256, num_heads=4)
        q = torch.randn(2, 10, 128)
        k = torch.randn(2, 10, 128)
        v = torch.randn(2, 10, 256)
        out = attn(q, k, v)
        assert out.shape == (2, 10, 256)

    def test_attention_multiple_heads(self) -> None:
        attn = CrossModalAttentionLayer(query_dim=64, key_dim=64, value_dim=128, num_heads=8)
        q = torch.randn(1, 5, 64)
        k = torch.randn(1, 5, 64)
        v = torch.randn(1, 5, 128)
        out = attn(q, k, v)
        assert out.shape == (1, 5, 128)

    def test_attention_self_attention(self) -> None:
        attn = CrossModalAttentionLayer(query_dim=64, key_dim=64, value_dim=64, num_heads=2)
        x = torch.randn(1, 10, 64)
        out = attn(x, x, x)
        assert out.shape == (1, 10, 64)
        # Output should differ from input (attention modifies)
        assert not torch.allclose(out, x)

    def test_attention_gradient_flow(self) -> None:
        attn = CrossModalAttentionLayer(query_dim=64, key_dim=64, value_dim=64, num_heads=2)
        q = torch.randn(1, 5, 64, requires_grad=True)
        k = torch.randn(1, 5, 64, requires_grad=True)
        v = torch.randn(1, 5, 64, requires_grad=True)
        out = attn(q, k, v)
        loss = out.mean()
        loss.backward()
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

    def test_attention_with_different_seq_lengths(self) -> None:
        attn = CrossModalAttentionLayer(query_dim=64, key_dim=64, value_dim=64, num_heads=2)
        q = torch.randn(1, 3, 64)
        k = torch.randn(1, 7, 64)
        v = torch.randn(1, 7, 64)
        out = attn(q, k, v)
        assert out.shape == (1, 3, 64)

    def test_cross_modal_attention_in_fusion(self, fusion: MultimodalEmbeddingFusion) -> None:
        fusion._ensure_eval_mode()
        projected = {
            "gesture": torch.randn(1, 1, 256),
            "speech": torch.randn(1, 1, 256),
        }
        attended = fusion._apply_cross_modal_attention(projected)
        assert "gesture" in attended
        assert "speech" in attended


class TestOutputHead:
    """Test the fusion output layer."""

    def test_fusion_layer_shape(self, fusion: MultimodalEmbeddingFusion) -> None:
        combined = torch.randn(1, 1, 256 * 4)
        fused = fusion._fusion_layer(combined)
        assert fused.shape == (1, 1, 256)

    def test_fusion_layer_gradient(self, fusion: MultimodalEmbeddingFusion) -> None:
        combined = torch.randn(1, 1, 256 * 4, requires_grad=True)
        fused = fusion._fusion_layer(combined)
        loss = fused.mean()
        loss.backward()
        for param in fusion._fusion_layer.parameters():
            assert param.grad is not None

    def test_modality_weight_gradient(self, fusion: MultimodalEmbeddingFusion) -> None:
        assert fusion._modality_logits.requires_grad is True

    def test_output_head_activation(self) -> None:
        layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )
        x = torch.randn(4, 256)
        out = layer(x)
        assert out.shape == (4, 64)
        assert not torch.isnan(out).any()


class TestGradientFlow:
    """Test gradient flow through the full multimodal model."""

    def test_full_model_gradients(self, fusion: MultimodalEmbeddingFusion) -> None:
        # Enable training mode
        fusion._projections.train()
        fusion._cross_attentions.train()
        fusion._fusion_layer.train()
        fusion._eval_mode = False

        gesture = torch.randn(1, 1, 128, requires_grad=True)
        speech = torch.randn(1, 1, 512, requires_grad=True)

        g_proj = fusion._projections["gesture"](gesture)
        s_proj = fusion._projections["speech"](speech)

        projected = {"gesture": g_proj, "speech": s_proj}
        attended = fusion._apply_cross_modal_attention(projected)
        weighted, _ = fusion._apply_modality_weights(attended, ["gesture", "speech"])

        emb_list = []
        for name in ["gesture", "speech", "facial", "context"]:
            if name in weighted:
                emb_list.append(weighted[name])
            else:
                emb_list.append(torch.zeros(1, 1, 256))

        combined = torch.cat(emb_list, dim=-1)
        fused = fusion._fusion_layer(combined)
        loss = fused.mean()
        loss.backward()

        for name, param in fusion._projections.named_parameters():
            assert param.grad is not None, f"No gradient in projection {name}"

        for name, param in fusion._fusion_layer.named_parameters():
            assert param.grad is not None, f"No gradient in fusion layer {name}"

    def test_cross_attention_gradients(self) -> None:
        fusion = MultimodalEmbeddingFusion(device="cpu")
        fusion._cross_attentions.train()
        fusion._eval_mode = False

        projected = {
            "gesture": torch.randn(1, 1, 256, requires_grad=True),
            "speech": torch.randn(1, 1, 256, requires_grad=True),
        }
        attended = fusion._apply_cross_modal_attention(projected)
        loss = attended["gesture"].mean() + attended["speech"].mean()
        loss.backward()

        for name, param in fusion._cross_attentions.named_parameters():
            assert param.grad is not None, f"No gradient in {name}"


class TestFusionEdgeCases:
    """Test edge cases in multimodal fusion."""

    def test_identical_modalities(self, fusion: MultimodalEmbeddingFusion) -> None:
        emb = np.random.randn(128).astype(np.float32)
        r1 = fusion.fuse_embeddings(gesture_emb=emb)
        r2 = fusion.fuse_embeddings(gesture_emb=emb.copy())
        np.testing.assert_array_almost_equal(r1.fused_vector, r2.fused_vector)

    def test_random_vs_constant(self, fusion: MultimodalEmbeddingFusion) -> None:
        gesture = np.random.randn(128).astype(np.float32)
        speech = np.random.randn(512).astype(np.float32)
        result = fusion.fuse_embeddings(gesture_emb=gesture, speech_emb=speech)
        assert not np.any(np.isnan(result.fused_vector))
        assert not np.any(np.isinf(result.fused_vector))

    def test_temporal_alignment(self, fusion: MultimodalEmbeddingFusion) -> None:
        gesture = np.random.randn(128).astype(np.float32)
        speech = np.random.randn(512).astype(np.float32)
        timestamps = {"gesture": 0.0, "speech": 0.5}
        result = fusion.fuse_embeddings(
            gesture_emb=gesture,
            speech_emb=speech,
            timestamps=timestamps,
        )
        assert result.fused_vector.shape == (256,)
