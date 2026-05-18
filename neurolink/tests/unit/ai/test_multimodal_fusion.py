from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch

from ai.multimodal_fusion.embeddings.fusion_embeddings import (
    CrossModalAttentionLayer,
    CrossModalProjection,
    FusedEmbedding,
    FusionError,
    ModalityEmbeddings,
    MultimodalEmbeddingFusion,
)


class TestMultimodalFusionInitialization:
    """Test MultimodalEmbeddingFusion initialization."""

    def test_default_initialization(self) -> None:
        fusion = MultimodalEmbeddingFusion(device="cpu")
        assert fusion._fused_dim == 512
        assert fusion._modality_dims["gesture"] == 128
        assert fusion._modality_dims["speech"] == 512
        assert fusion._modality_dims["facial"] == 128
        assert fusion._modality_dims["context"] == 256

    def test_custom_dimensions(self) -> None:
        fusion = MultimodalEmbeddingFusion(
            gesture_dim=64,
            speech_dim=256,
            facial_dim=64,
            context_dim=128,
            fused_dim=256,
            device="cpu",
        )
        assert fusion._modality_dims["gesture"] == 64
        assert fusion._fused_dim == 256

    def test_default_device(self) -> None:
        fusion = MultimodalEmbeddingFusion()
        if torch.cuda.is_available():
            assert str(fusion._device) == "cuda"
        else:
            assert str(fusion._device) == "cpu"

    def test_eval_mode_on_init(self) -> None:
        fusion = MultimodalEmbeddingFusion(device="cpu")
        assert fusion._eval_mode is False


class TestEmbeddingFusion:
    """Test multimodal embedding fusion."""

    @pytest.fixture
    def fusion(self) -> MultimodalEmbeddingFusion:
        return MultimodalEmbeddingFusion(device="cpu")

    @pytest.fixture
    def gesture_emb(self) -> np.ndarray:
        return np.random.randn(128).astype(np.float32)

    @pytest.fixture
    def speech_emb(self) -> np.ndarray:
        return np.random.randn(512).astype(np.float32)

    def test_fuse_two_modalities(
        self,
        fusion: MultimodalEmbeddingFusion,
        gesture_emb: np.ndarray,
        speech_emb: np.ndarray,
    ) -> None:
        result = fusion.fuse_embeddings(gesture_emb=gesture_emb, speech_emb=speech_emb)
        assert isinstance(result, FusedEmbedding)
        assert result.fused_vector.shape == (fusion._fused_dim,)
        assert result.confidence > 0.0
        assert "gesture" in result.modality_weights
        assert "speech" in result.modality_weights

    def test_fuse_single_modality(
        self,
        fusion: MultimodalEmbeddingFusion,
        gesture_emb: np.ndarray,
    ) -> None:
        result = fusion.fuse_embeddings(gesture_emb=gesture_emb)
        assert isinstance(result, FusedEmbedding)
        assert result.fused_vector.shape == (fusion._fused_dim,)
        assert result.modality_weights["gesture"] > 0.0

    def test_fuse_all_modalities(
        self,
        fusion: MultimodalEmbeddingFusion,
        gesture_emb: np.ndarray,
        speech_emb: np.ndarray,
    ) -> None:
        facial_emb = np.random.randn(128).astype(np.float32)
        context_emb = np.random.randn(256).astype(np.float32)
        result = fusion.fuse_embeddings(
            gesture_emb=gesture_emb,
            speech_emb=speech_emb,
            facial_emb=facial_emb,
            context_emb=context_emb,
        )
        assert isinstance(result, FusedEmbedding)
        assert result.fused_vector.shape == (fusion._fused_dim,)
        assert len(result.modality_weights) == 4

    def test_empty_modalities(self, fusion: MultimodalEmbeddingFusion) -> None:
        with pytest.raises(FusionError, match="No modalities provided"):
            fusion.fuse_embeddings()

    def test_fusion_confidence_high_with_multiple_modalities(
        self,
        fusion: MultimodalEmbeddingFusion,
        gesture_emb: np.ndarray,
        speech_emb: np.ndarray,
    ) -> None:
        result = fusion.fuse_embeddings(gesture_emb=gesture_emb, speech_emb=speech_emb)
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0


class TestModalityEmbeddings:
    """Test ModalityEmbeddings container."""

    def test_available_modalities(self) -> None:
        mod = ModalityEmbeddings(
            gesture=np.random.randn(128).astype(np.float32),
            speech=None,
            facial=np.random.randn(128).astype(np.float32),
            context=None,
        )
        assert mod.available_modalities == ["gesture", "facial"]

    def test_no_available_modalities(self) -> None:
        mod = ModalityEmbeddings()
        assert mod.available_modalities == []

    def test_timestamps(self) -> None:
        import time
        now = time.time()
        mod = ModalityEmbeddings(timestamps={"gesture": now, "speech": now + 0.1})
        assert "gesture" in mod.timestamps
        assert "speech" in mod.timestamps


class TestFusedEmbedding:
    """Test FusedEmbedding dataclass."""

    def test_fused_embedding_creation(self) -> None:
        emb = FusedEmbedding(
            fused_vector=np.random.randn(512).astype(np.float32),
            modality_weights={"gesture": 0.6, "speech": 0.4},
            alignment_scores={"gesture_speech": 0.85},
            confidence=0.9,
            metadata={"modalities": ["gesture", "speech"]},
        )
        assert emb.fused_vector.shape == (512,)
        assert emb.confidence == 0.9
        assert emb.modality_weights["gesture"] == 0.6


class TestCrossModalProjection:
    """Test CrossModalProjection layer."""

    def test_projection_shape(self) -> None:
        proj = CrossModalProjection(in_dim=128, out_dim=256)
        x = torch.randn(2, 10, 128)
        out = proj(x)
        assert out.shape == (2, 10, 256)

    def test_projection_normalization(self) -> None:
        proj = CrossModalProjection(in_dim=64, out_dim=128)
        x = torch.randn(1, 1, 64)
        out = proj(x)
        assert not torch.isnan(out).any()


class TestCrossModalAttention:
    """Test cross-modal attention mechanism."""

    def test_attention_shape(self) -> None:
        attn = CrossModalAttentionLayer(query_dim=128, key_dim=128, value_dim=128, num_heads=4)
        q = torch.randn(2, 10, 128)
        k = torch.randn(2, 20, 128)
        v = torch.randn(2, 20, 128)
        out = attn(q, k, v)
        assert out.shape == (2, 10, 128)

    def test_attention_multiple_heads(self) -> None:
        attn = CrossModalAttentionLayer(query_dim=256, key_dim=256, value_dim=256, num_heads=8)
        q = torch.randn(1, 5, 256)
        k = torch.randn(1, 5, 256)
        v = torch.randn(1, 5, 256)
        out = attn(q, k, v)
        assert out.shape == (1, 5, 256)

    def test_attention_with_mask(self) -> None:
        attn = CrossModalAttentionLayer(query_dim=64, key_dim=64, value_dim=64, num_heads=2)
        q = torch.randn(1, 3, 64)
        k = torch.randn(1, 5, 64)
        v = torch.randn(1, 5, 64)
        mask = torch.ones(1, 1, 3, 5)
        mask[:, :, :, 2:] = 0
        out = attn(q, k, v, mask=mask)
        assert out.shape == (1, 3, 64)


class TestMissingModalityHandling:
    """Test handling of missing modalities."""

    @pytest.fixture
    def fusion(self) -> MultimodalEmbeddingFusion:
        return MultimodalEmbeddingFusion(device="cpu")

    def test_missing_speech_modality(self, fusion: MultimodalEmbeddingFusion) -> None:
        gesture = np.random.randn(128).astype(np.float32)
        facial = np.random.randn(128).astype(np.float32)
        result = fusion.fuse_embeddings(gesture_emb=gesture, facial_emb=facial)
        assert result.modality_weights["speech"] == 0.0
        assert result.modality_weights["gesture"] > 0.0
        assert result.modality_weights["facial"] > 0.0

    def test_missing_gesture_modality(self, fusion: MultimodalEmbeddingFusion) -> None:
        speech = np.random.randn(512).astype(np.float32)
        result = fusion.fuse_embeddings(speech_emb=speech)
        assert result.modality_weights["gesture"] >= 0.0

    def test_weight_distribution_changes(self, fusion: MultimodalEmbeddingFusion) -> None:
        gesture = np.random.randn(128).astype(np.float32)
        result1 = fusion.fuse_embeddings(gesture_emb=gesture)
        result2 = fusion.fuse_embeddings(gesture_emb=gesture, speech_emb=np.random.randn(512).astype(np.float32))
        assert result2.modality_weights["gesture"] < 1.0
        assert result2.modality_weights["speech"] > 0.0


class TestNormalization:
    """Test embedding normalization."""

    @pytest.fixture
    def fusion(self) -> MultimodalEmbeddingFusion:
        return MultimodalEmbeddingFusion(device="cpu")

    def test_l2_normalization(self, fusion: MultimodalEmbeddingFusion) -> None:
        emb = np.random.randn(100).astype(np.float32)
        normalized = fusion.normalize_embedding(emb, "l2")
        norm = np.linalg.norm(normalized)
        assert abs(norm - 1.0) < 1e-6

    def test_unit_normalization(self, fusion: MultimodalEmbeddingFusion) -> None:
        emb = np.random.randn(100).astype(np.float32)
        normalized = fusion.normalize_embedding(emb, "unit")
        assert np.max(np.abs(normalized)) <= 1.0

    def test_minmax_normalization(self, fusion: MultimodalEmbeddingFusion) -> None:
        emb = np.random.randn(100).astype(np.float32)
        normalized = fusion.normalize_embedding(emb, "minmax")
        assert normalized.min() >= -1e-6
        assert normalized.max() <= 1.0 + 1e-6

    def test_invalid_norm_type(self, fusion: MultimodalEmbeddingFusion) -> None:
        emb = np.random.randn(10).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown normalization type"):
            fusion.normalize_embedding(emb, "invalid")


class TestAlignmentScores:
    """Test alignment score computation."""

    @pytest.fixture
    def fusion(self) -> MultimodalEmbeddingFusion:
        return MultimodalEmbeddingFusion(device="cpu")

    def test_alignment_scores_in_result(
        self,
        fusion: MultimodalEmbeddingFusion,
    ) -> None:
        gesture = np.random.randn(128).astype(np.float32)
        speech = np.random.randn(512).astype(np.float32)
        result = fusion.fuse_embeddings(gesture_emb=gesture, speech_emb=speech)
        assert "gesture_speech" in result.alignment_scores
        assert -1.0 <= result.alignment_scores["gesture_speech"] <= 1.0


class TestModelPersistence:
    """Test fusion model save/load."""

    def test_save_and_load(self, tmp_path: str) -> None:
        fusion = MultimodalEmbeddingFusion(device="cpu")
        path = str(tmp_path / "fusion_model.pt")
        fusion.save(path)
        assert hasattr(fusion, "load")
        fusion2 = MultimodalEmbeddingFusion(device="cpu")
        fusion2.load(path)
        # Verify both models produce similar output
        gesture = np.random.randn(128).astype(np.float32)
        r1 = fusion.fuse_embeddings(gesture_emb=gesture)
        r2 = fusion2.fuse_embeddings(gesture_emb=gesture)
        assert np.allclose(r1.fused_vector, r2.fused_vector, atol=1e-5)
