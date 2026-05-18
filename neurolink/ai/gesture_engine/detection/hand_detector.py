import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class HandDetectionError(Exception):
    """Raised when hand detection fails."""


class LowConfidenceError(HandDetectionError):
    """Raised when detection confidence is below threshold."""


@dataclass
class HandLandmarks:
    """Represents detected hand landmarks."""

    landmarks: np.ndarray  # shape (21, 3) - x, y, z coordinates
    handedness: str  # "Left" or "Right"
    confidence: float
    roi: Tuple[int, int, int, int]  # x, y, w, h bounding box
    timestamp: float = field(default_factory=time.time)


class HandDetector:
    """MediaPipe-based hand detection with performance optimization and error handling.

    Detects up to max_hands hands per frame with configurable confidence thresholds.
    Supports ROI tracking and frame skip for real-time performance.

    Performance optimization:
    - Frame skip rate to process every N frames
    - ROI tracking to limit detection area
    - Static image mode toggle for different use cases
    """

    NUM_HAND_LANDMARKS: int = 21
    LANDMARK_COORDS: int = 3

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        skip_rate: int = 0,
        use_roi_tracking: bool = True,
        static_image_mode: bool = False,
    ) -> None:
        """Initialize the MediaPipe Hands solution.

        Args:
            max_hands: Maximum number of hands to detect (default: 2).
            min_detection_confidence: Minimum confidence for detection (default: 0.7).
            min_tracking_confidence: Minimum confidence for tracking (default: 0.5).
            model_complexity: 0=lite, 1=full, 2=heavy (default: 1).
            skip_rate: Process every (skip_rate+1)th frame (default: 0 = all frames).
            use_roi_tracking: Enable ROI-based tracking for performance (default: True).
            static_image_mode: Static image mode for non-video input (default: False).

        Raises:
            ValueError: If parameters are out of valid range.
        """
        if not 0 <= min_detection_confidence <= 1:
            raise ValueError("min_detection_confidence must be between 0 and 1")
        if not 0 <= min_tracking_confidence <= 1:
            raise ValueError("min_tracking_confidence must be between 0 and 1")
        if max_hands < 1:
            raise ValueError("max_hands must be at least 1")
        if skip_rate < 0:
            raise ValueError("skip_rate must be non-negative")

        self._max_hands = max_hands
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._model_complexity = model_complexity
        self._skip_rate = skip_rate
        self._use_roi_tracking = use_roi_tracking
        self._static_image_mode = static_image_mode

        self._frame_count: int = 0
        self._prev_roi: Optional[Tuple[int, int, int, int]] = None
        self._roi_margin: int = 50

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=self._static_image_mode,
            max_num_hands=self._max_hands,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            model_complexity=self._model_complexity,
        )

        logger.info(
            "hand_detector_initialized",
            max_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            model_complexity=model_complexity,
        )

    def detect_hands(self, frame: np.ndarray) -> List[HandLandmarks]:
        """Detect hands in the given frame.

        Applies frame skip and ROI tracking for performance optimization.

        Args:
            frame: Input image as BGR numpy array (H, W, 3).

        Returns:
            List of HandLandmarks for each detected hand.

        Raises:
            HandDetectionError: If detection fails entirely.
        """
        if frame is None or frame.size == 0:
            raise HandDetectionError("Empty or invalid frame provided")

        self._frame_count += 1

        # Frame skip optimization
        if self._skip_rate > 0 and self._frame_count % (self._skip_rate + 1) != 0:
            return []

        try:
            # ROI tracking: only process a sub-region if we have prior detection
            if self._use_roi_tracking and self._prev_roi is not None:
                x, y, w, h = self._prev_roi
                roi_frame = frame[y : y + h, x : x + w]
                if roi_frame.size == 0:
                    roi_frame = frame
                    self._prev_roi = None
            else:
                roi_frame = frame

            rgb_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb_frame)

            if not results or not results.multi_hand_landmarks:
                if self._frame_count > 10:
                    # After initial frames, no valid detection
                    pass
                return []

            hand_landmarks_list: List[HandLandmarks] = []
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks_array = self.extract_landmarks(hand_landmarks)
                handedness = self.calculate_handedness(results, idx)
                confidence = self._estimate_confidence(hand_landmarks)

                if confidence < self._min_detection_confidence:
                    logger.debug("low_confidence_hand", confidence=confidence)
                    continue

                roi = self._compute_roi(landmarks_array, frame.shape)

                hand_lm = HandLandmarks(
                    landmarks=landmarks_array,
                    handedness=handedness,
                    confidence=confidence,
                    roi=roi,
                )
                hand_landmarks_list.append(hand_lm)

            # Update ROI for next frame based on current detections
            if self._use_roi_tracking and hand_landmarks_list:
                self._update_roi(hand_landmarks_list, frame.shape)

            return hand_landmarks_list

        except Exception as e:
            logger.error("hand_detection_failed", error=str(e))
            raise HandDetectionError(f"Hand detection failed: {e}") from e

    def extract_landmarks(self, hand_landmarks) -> np.ndarray:
        """Extract 21 landmark coordinates as a numpy array.

        Args:
            hand_landmarks: MediaPipe NormalizedLandmarkList for a single hand.

        Returns:
            numpy array of shape (21, 3) with (x, y, z) normalized coordinates.
        """
        landmarks = np.zeros((self.NUM_HAND_LANDMARKS, self.LANDMARK_COORDS), dtype=np.float32)
        for i in range(self.NUM_HAND_LANDMARKS):
            lm = hand_landmarks.landmark[i]
            landmarks[i, 0] = lm.x
            landmarks[i, 1] = lm.y
            landmarks[i, 2] = lm.z
        return landmarks

    def calculate_handedness(self, results, idx: int) -> str:
        """Determine if the hand is left or right.

        Args:
            results: MediaPipe Hands detection results.
            idx: Index of the hand in multi_handedness list.

        Returns:
            "Left" or "Right".
        """
        if results.multi_handedness and idx < len(results.multi_handedness):
            return results.multi_handedness[idx].classification[0].label
        return "Unknown"

    def draw_landmarks(self, frame: np.ndarray, landmarks_list: List[HandLandmarks]) -> np.ndarray:
        """Draw hand landmarks and connections on the frame.

        Args:
            frame: Input image to draw on (modified in-place).
            landmarks_list: List of HandLandmarks to visualize.

        Returns:
            Annotated frame.
        """
        if frame is None:
            return frame

        for hand_lm in landmarks_list:
            # Draw ROI rectangle
            x, y, w, h = hand_lm.roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # Draw handedness label
            label = f"{hand_lm.handedness} ({hand_lm.confidence:.2f})"
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            # Draw landmarks as circles
            h_img, w_img, _ = frame.shape
            landmark_connections = self._mp_hands.HAND_CONNECTIONS

            for i in range(self.NUM_HAND_LANDMARKS):
                cx = int(hand_lm.landmarks[i, 0] * w_img)
                cy = int(hand_lm.landmarks[i, 1] * h_img)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

            # Draw connections
            for connection in landmark_connections:
                start_idx, end_idx = connection
                x1 = int(hand_lm.landmarks[start_idx, 0] * w_img)
                y1 = int(hand_lm.landmarks[start_idx, 1] * h_img)
                x2 = int(hand_lm.landmarks[end_idx, 0] * w_img)
                y2 = int(hand_lm.landmarks[end_idx, 1] * h_img)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame

    def reset(self) -> None:
        """Reset detector state (frame count, ROI)."""
        self._frame_count = 0
        self._prev_roi = None
        logger.info("hand_detector_reset")

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._hands:
            self._hands.close()
            logger.info("hand_detector_closed")

    def _estimate_confidence(self, hand_landmarks) -> float:
        """Estimate detection confidence from landmark visibility.

        Args:
            hand_landmarks: MediaPipe landmark list.

        Returns:
            Average visibility score as proxy for confidence.
        """
        visibilities = [lm.visibility for lm in hand_landmarks.landmark]
        return float(np.mean(visibilities))

    def _compute_roi(
        self, landmarks: np.ndarray, frame_shape: Tuple[int, int, int]
    ) -> Tuple[int, int, int, int]:
        """Compute bounding box ROI from landmarks.

        Args:
            landmarks: (21, 3) landmark array.
            frame_shape: (H, W, C) of the frame.

        Returns:
            (x, y, w, h) bounding box.
        """
        h_img, w_img, _ = frame_shape
        xs = landmarks[:, 0] * w_img
        ys = landmarks[:, 1] * h_img
        x_min, x_max = int(np.min(xs)), int(np.max(xs))
        y_min, y_max = int(np.min(ys)), int(np.max(ys))

        # Add margin
        margin = self._roi_margin
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w_img - 1, x_max + margin)
        y_max = min(h_img - 1, y_max + margin)

        w = x_max - x_min
        h = y_max - y_min
        return (x_min, y_min, w, h)

    def _update_roi(
        self, hand_landmarks_list: List[HandLandmarks], frame_shape: Tuple[int, int, int]
    ) -> None:
        """Update tracked ROI based on current detections for next frame.

        Combines all detected hands into a single expanded ROI.
        """
        if not hand_landmarks_list:
            return

        h_img, w_img, _ = frame_shape
        all_xs: List[int] = []
        all_ys: List[int] = []

        for hand_lm in hand_landmarks_list:
            xs = (hand_lm.landmarks[:, 0] * w_img).astype(int)
            ys = (hand_lm.landmarks[:, 1] * h_img).astype(int)
            all_xs.extend(xs.tolist())
            all_ys.extend(ys.tolist())

        if not all_xs:
            return

        margin = self._roi_margin
        x_min = max(0, min(all_xs) - margin)
        y_min = max(0, min(all_ys) - margin)
        x_max = min(w_img - 1, max(all_xs) + margin)
        y_max = min(h_img - 1, max(all_ys) + margin)

        self._prev_roi = (x_min, y_min, x_max - x_min, y_max - y_min)


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
