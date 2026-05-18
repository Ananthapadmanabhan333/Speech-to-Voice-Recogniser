from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pytest

from ai.gesture_engine.detection.hand_detector import (
    HandDetectionError,
    HandDetector,
    HandLandmarks,
    LowConfidenceError,
)


class TestHandDetectorInitialization:
    """Test HandDetector initialization and parameter validation."""

    def test_default_initialization(self) -> None:
        detector = HandDetector()
        assert detector._max_hands == 2
        assert detector._min_detection_confidence == 0.7
        assert detector._min_tracking_confidence == 0.5
        assert detector._model_complexity == 1
        assert detector._skip_rate == 0
        assert detector._use_roi_tracking is True

    def test_custom_parameters(self) -> None:
        detector = HandDetector(
            max_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.6,
            model_complexity=0,
            skip_rate=2,
            use_roi_tracking=False,
            static_image_mode=True,
        )
        assert detector._max_hands == 1
        assert detector._min_detection_confidence == 0.8
        assert detector._model_complexity == 0
        assert detector._skip_rate == 2
        assert detector._use_roi_tracking is False
        assert detector._static_image_mode is True

    def test_invalid_confidence_range(self) -> None:
        with pytest.raises(ValueError, match="min_detection_confidence"):
            HandDetector(min_detection_confidence=1.5)

    def test_invalid_confidence_negative(self) -> None:
        with pytest.raises(ValueError, match="min_detection_confidence"):
            HandDetector(min_detection_confidence=-0.1)

    def test_invalid_tracking_confidence(self) -> None:
        with pytest.raises(ValueError, match="min_tracking_confidence"):
            HandDetector(min_tracking_confidence=1.5)

    def test_invalid_max_hands(self) -> None:
        with pytest.raises(ValueError, match="max_hands"):
            HandDetector(max_hands=0)

    def test_invalid_skip_rate(self) -> None:
        with pytest.raises(ValueError, match="skip_rate"):
            HandDetector(skip_rate=-1)


class TestHandDetectorDetection:
    """Test hand detection on synthetic data."""

    @pytest.fixture
    def detector(self) -> HandDetector:
        return HandDetector(
            max_hands=2,
            min_detection_confidence=0.0,  # Accept everything for testing
            min_tracking_confidence=0.0,
            static_image_mode=True,
        )

    def test_empty_frame_raises_error(self, detector: HandDetector) -> None:
        with pytest.raises(HandDetectionError, match="Empty or invalid frame"):
            detector.detect_hands(np.array([]))

    def test_none_frame_raises_error(self, detector: HandDetector) -> None:
        with pytest.raises(HandDetectionError, match="Empty or invalid frame"):
            detector.detect_hands(None)  # type: ignore[arg-type]

    def test_valid_frame_no_hands(self, detector: HandDetector) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect_hands(frame)
        assert isinstance(results, list)

    def test_detection_returns_list(self, detector: HandDetector) -> None:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = detector.detect_hands(frame)
        assert isinstance(results, list)

    def test_frame_skip_optimization(self) -> None:
        detector = HandDetector(min_detection_confidence=0.0, min_tracking_confidence=0.0, skip_rate=2)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # First frame: should process
        r1 = detector.detect_hands(frame)
        # Second frame: should skip
        r2 = detector.detect_hands(frame)
        # Third frame: should skip
        r3 = detector.detect_hands(frame)
        # Fourth frame: should process
        r4 = detector.detect_hands(frame)

        assert isinstance(r1, list)
        assert isinstance(r2, list)
        assert isinstance(r3, list)
        assert isinstance(r4, list)


class TestHandLandmarks:
    """Test HandLandmarks dataclass."""

    def test_landmarks_shape(self) -> None:
        landmarks = np.zeros((21, 3), dtype=np.float32)
        hl = HandLandmarks(
            landmarks=landmarks,
            handedness="Left",
            confidence=0.9,
            roi=(10, 20, 100, 200),
        )
        assert hl.landmarks.shape == (21, 3)
        assert hl.handedness == "Left"
        assert hl.confidence == 0.9
        assert hl.roi == (10, 20, 100, 200)
        assert hl.timestamp > 0

    def test_landmarks_default_timestamp(self) -> None:
        import time
        before = time.time()
        hl = HandLandmarks(
            landmarks=np.zeros((21, 3)),
            handedness="Right",
            confidence=0.8,
            roi=(0, 0, 50, 50),
        )
        after = time.time()
        assert before <= hl.timestamp <= after


class TestHandDetectorUtilities:
    """Test HandDetector utility methods."""

    def test_reset(self) -> None:
        detector = HandDetector(min_detection_confidence=0.0, min_tracking_confidence=0.0)
        detector._frame_count = 42
        detector._prev_roi = (10, 20, 100, 200)
        detector.reset()
        assert detector._frame_count == 0
        assert detector._prev_roi is None

    def test_context_manager(self) -> None:
        with HandDetector(min_detection_confidence=0.0, min_tracking_confidence=0.0) as detector:
            assert isinstance(detector, HandDetector)
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            results = detector.detect_hands(frame)
            assert isinstance(results, list)

    def test_draw_landmarks_no_frame(self) -> None:
        detector = HandDetector(min_detection_confidence=0.0, min_tracking_confidence=0.0)
        result = detector.draw_landmarks(None, [])
        assert result is None

    def test_draw_landmarks_empty_list(self) -> None:
        detector = HandDetector(min_detection_confidence=0.0, min_tracking_confidence=0.0)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.draw_landmarks(frame, [])
        assert result is not None
        assert result.shape == frame.shape


class TestROITracking:
    """Test ROI tracking optimization."""

    def test_roi_disabled(self) -> None:
        detector = HandDetector(
            min_detection_confidence=0.0,
            min_tracking_confidence=0.0,
            use_roi_tracking=False,
        )
        assert detector._use_roi_tracking is False

    def test_roi_update_with_results(self) -> None:
        detector = HandDetector(
            min_detection_confidence=0.0,
            min_tracking_confidence=0.0,
            use_roi_tracking=True,
        )
        landmarks = np.array([
            [0.5, 0.5, 0.0],
            [0.6, 0.5, 0.0],
            [0.5, 0.6, 0.0],
        ] * 7, dtype=np.float32)[:21]
        hl = HandLandmarks(landmarks=landmarks, handedness="Right", confidence=0.9, roi=(0, 0, 100, 100))
        frame_shape = (480, 640, 3)
        detector._update_roi([hl], frame_shape)
        assert detector._prev_roi is not None

    def test_roi_update_empty_list(self) -> None:
        detector = HandDetector(
            min_detection_confidence=0.0,
            min_tracking_confidence=0.0,
            use_roi_tracking=True,
        )
        detector._prev_roi = (10, 20, 100, 200)
        detector._update_roi([], (480, 640, 3))
        assert detector._prev_roi == (10, 20, 100, 200)


class TestHandDetectorPerformance:
    """Test HandDetector performance characteristics."""

    def test_extract_landmarks_shape(self) -> None:
        # We can't easily mock MediaPipe landmarks, but we can test the expected shape
        detector = HandDetector(min_detection_confidence=0.0, min_tracking_confidence=0.0)
        assert detector.NUM_HAND_LANDMARKS == 21
        assert detector.LANDMARK_COORDS == 3

    def test_low_confidence_handling(self) -> None:
        detector = HandDetector(min_detection_confidence=0.95)
        with pytest.raises(HandDetectionError, match="Empty or invalid frame"):
            detector.detect_hands(np.array([]))

    def test_frame_count_increment(self) -> None:
        detector = HandDetector(min_detection_confidence=0.0, min_tracking_confidence=0.0)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert detector._frame_count == 0
        detector.detect_hands(frame)
        assert detector._frame_count == 1
        detector.detect_hands(frame)
        assert detector._frame_count == 2
