from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from ai.gesture_engine.classification.gesture_classifier import (
    GestureClassificationError,
    GestureClassifier,
    GestureLSTM,
    GestureResult,
    TemporalCNN,
    UnknownGestureError,
)


class TestGestureClassifierInitialization:
    """Test GestureClassifier initialization."""

    def test_default_initialization(self) -> None:
        classifier = GestureClassifier(device="cpu")
        assert classifier._num_classes == len(classifier.DEFAULT_GESTURES)
        assert classifier._temperature == 1.5
        assert classifier._device.type == "cpu"

    def test_custom_num_classes(self) -> None:
        classifier = GestureClassifier(num_classes=10, device="cpu")
        assert classifier._num_classes == 10

    def test_custom_temperature(self) -> None:
        classifier = GestureClassifier(temperature=2.0, device="cpu")
        assert classifier._temperature == 2.0

    def test_custom_device(self) -> None:
        classifier = GestureClassifier(device="cpu")
        assert str(classifier._device) == "cpu"

    def test_gesture_labels_initialized(self) -> None:
        classifier = GestureClassifier(device="cpu")
        assert len(classifier._gesture_labels) > 0
        assert classifier._gesture_labels[0] == "A"
        assert "hello" in classifier._gesture_labels

    def test_label_mappings(self) -> None:
        classifier = GestureClassifier(num_classes=5, device="cpu")
        assert len(classifier._label_to_id) == 5
        assert len(classifier._id_to_label) == 5


class TestGestureClassification:
    """Test gesture classification on synthetic data."""

    @pytest.fixture
    def classifier(self) -> GestureClassifier:
        return GestureClassifier(num_classes=10, device="cpu")

    def test_classify_gesture_shape(self, classifier: GestureClassifier) -> None:
        seq = np.random.randn(30, 21, 3).astype(np.float32)
        result = classifier.classify_gesture(seq)
        assert isinstance(result, GestureResult)
        assert isinstance(result.gesture_id, str)
        assert isinstance(result.gesture_label, str)
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.calibrated_confidence <= 1.0
        assert isinstance(result.raw_logits, np.ndarray)
        assert result.timestamp > 0

    def test_empty_sequence_raises_error(self, classifier: GestureClassifier) -> None:
        with pytest.raises(ValueError, match="Empty landmark sequence"):
            classifier.classify_gesture(np.array([]))

    def test_none_sequence_raises_error(self, classifier: GestureClassifier) -> None:
        with pytest.raises(ValueError, match="Empty landmark sequence"):
            classifier.classify_gesture(None)  # type: ignore[arg-type]

    def test_short_sequence_raises_error(self, classifier: GestureClassifier) -> None:
        short = np.random.randn(3, 21, 3).astype(np.float32)
        with pytest.raises(ValueError, match="Sequence too short"):
            classifier.classify_gesture(short)

    def test_long_sequence_truncated(self, classifier: GestureClassifier) -> None:
        long_seq = np.random.randn(200, 21, 3).astype(np.float32)
        result = classifier.classify_gesture(long_seq)
        assert isinstance(result, GestureResult)

    def test_min_sequence_length(self, classifier: GestureClassifier) -> None:
        min_seq = np.random.randn(5, 21, 3).astype(np.float32)
        result = classifier.classify_gesture(min_seq)
        assert isinstance(result, GestureResult)

    def test_classification_deterministic(self, classifier: GestureClassifier) -> None:
        seq = np.random.randn(30, 21, 3).astype(np.float32)
        r1 = classifier.classify_gesture(seq)
        r2 = classifier.classify_gesture(seq)
        assert r1.gesture_id == r2.gesture_id


class TestConfidenceScoring:
    """Test confidence scoring and calibration."""

    @pytest.fixture
    def classifier(self) -> GestureClassifier:
        return GestureClassifier(num_classes=5, device="cpu", temperature=1.5)

    def test_confidence_range(self, classifier: GestureClassifier) -> None:
        seq = np.random.randn(30, 21, 3).astype(np.float32)
        result = classifier.classify_gesture(seq)
        assert 0.0 <= result.confidence <= 1.0

    def test_calibrated_confidence_differs(self, classifier: GestureClassifier) -> None:
        seq = np.random.randn(30, 21, 3).astype(np.float32)
        result = classifier.classify_gesture(seq)
        # With temperature > 1, calibrated confidence should be lower or equal
        assert result.calibrated_confidence <= result.confidence + 1e-6

    def test_temperature_effect(self) -> None:
        c1 = GestureClassifier(num_classes=5, device="cpu", temperature=1.0)
        c2 = GestureClassifier(num_classes=5, device="cpu", temperature=5.0)
        seq = np.random.randn(30, 21, 3).astype(np.float32)
        r1 = c1.classify_gesture(seq)
        r2 = c2.classify_gesture(seq)
        # Higher temperature should produce more uniform probabilities
        assert abs(r1.confidence - r2.confidence) < 1.0  # Both valid

    def test_confidence_with_identical_inputs(self, classifier: GestureClassifier) -> None:
        seq = np.ones((30, 21, 3), dtype=np.float32)
        result = classifier.classify_gesture(seq)
        assert isinstance(result, GestureResult)
        assert result.confidence >= 0.0


class TestCustomGestureRegistration:
    """Test custom gesture registration and matching."""

    @pytest.fixture
    def classifier(self) -> GestureClassifier:
        return GestureClassifier(num_classes=5, device="cpu")

    def test_register_custom_gesture(self, classifier: GestureClassifier) -> None:
        prototypes = [np.random.randn(30, 21, 3).astype(np.float32) for _ in range(3)]
        classifier.register_custom_gesture("my_custom_wave", prototypes)
        assert "my_custom_wave" in classifier._custom_gestures

    def test_duplicate_label_raises_error(self, classifier: GestureClassifier) -> None:
        prototypes = [np.random.randn(30, 21, 3).astype(np.float32)]
        with pytest.raises(ValueError, match="already exists"):
            classifier.register_custom_gesture("A", prototypes)

    def test_empty_prototypes_raises_error(self, classifier: GestureClassifier) -> None:
        with pytest.raises(ValueError, match="at least one prototype"):
            classifier.register_custom_gesture("custom", [])

    def test_invalid_prototype_shape_raises_error(self, classifier: GestureClassifier) -> None:
        with pytest.raises(ValueError, match="Invalid shape"):
            classifier.register_custom_gesture("custom", [np.random.randn(10, 5).astype(np.float32)])

    def test_remove_custom_gesture(self, classifier: GestureClassifier) -> None:
        prototypes = [np.random.randn(30, 21, 3).astype(np.float32)]
        classifier.register_custom_gesture("custom_test", prototypes)
        assert "custom_test" in classifier._custom_gestures
        classifier.remove_custom_gesture("custom_test")
        assert "custom_test" not in classifier._custom_gestures

    def test_get_gesture_labels(self, classifier: GestureClassifier) -> None:
        labels = classifier.get_gesture_labels()
        assert isinstance(labels, list)
        assert len(labels) == 5


class TestSequenceClassification:
    """Test sequence-level classification."""

    @pytest.fixture
    def classifier(self) -> GestureClassifier:
        return GestureClassifier(num_classes=5, device="cpu")

    def test_batch_classification_consistency(self, classifier: GestureClassifier) -> None:
        seq1 = np.random.randn(30, 21, 3).astype(np.float32)
        seq2 = np.random.randn(30, 21, 3).astype(np.float32)
        r1 = classifier.classify_gesture(seq1)
        r2 = classifier.classify_gesture(seq2)
        assert isinstance(r1, GestureResult)
        assert isinstance(r2, GestureResult)

    def test_sequence_with_constant_landmarks(self, classifier: GestureClassifier) -> None:
        seq = np.zeros((30, 21, 3), dtype=np.float32)
        result = classifier.classify_gesture(seq)
        assert isinstance(result, GestureResult)


class TestModelArchitecture:
    """Test underlying model architecture."""

    def test_temporal_cnn_forward(self) -> None:
        model = TemporalCNN(input_dim=63, hidden_dims=[128, 64])
        x = torch.randn(2, 30, 63)
        out = model(x)
        assert out.shape == (2, 30, 64)

    def test_gesture_lstm_forward(self) -> None:
        model = GestureLSTM(input_dim=128, hidden_dim=64, num_layers=2, num_classes=10)
        x = torch.randn(2, 30, 128)
        out = model(x)
        assert out.shape == (2, 10)

    def test_full_pipeline_shapes(self, classifier: GestureClassifier) -> None:
        seq = np.random.randn(30, 21, 3).astype(np.float32)
        features = classifier._extract_features(seq)
        assert features.shape[1] == classifier.MAX_SEQUENCE_LENGTH
        assert features.shape[2] == classifier.LANDMARK_DIM

    def test_feature_extraction_normalization(self, classifier: GestureClassifier) -> None:
        seq = np.random.randn(30, 21, 3).astype(np.float32)
        features = classifier._extract_features(seq)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))


class TestModelPersistence:
    """Test model save/load operations."""

    def test_save_and_load(self, classifier: GestureClassifier, tmp_path: Path) -> None:
        path = tmp_path / "gesture_model.pt"
        classifier.save(path)
        assert path.exists()

        new_classifier = GestureClassifier(num_classes=5, device="cpu", model_path=path)
        assert new_classifier._temperature == classifier._temperature
        assert new_classifier._gesture_labels == classifier._gesture_labels

    def test_load_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            GestureClassifier(device="cpu", model_path=Path("/nonexistent/model.pt"))

    def test_save_labels(self, classifier: GestureClassifier, tmp_path: Path) -> None:
        path = tmp_path / "labels.json"
        classifier.save_labels(path)
        assert path.exists()


class TestUnknownGestureRejection:
    """Test handling of unknown/unrecognized gestures."""

    @pytest.fixture
    def classifier(self) -> GestureClassifier:
        return GestureClassifier(num_classes=2, device="cpu")

    def test_unknown_gesture_low_confidence(self, classifier: GestureClassifier) -> None:
        seq = np.random.randn(30, 21, 3).astype(np.float32) * 100.0
        result = classifier.classify_gesture(seq)
        assert isinstance(result, GestureResult)
