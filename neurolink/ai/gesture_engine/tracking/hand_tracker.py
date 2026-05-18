from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from ai.gesture_engine.detection.hand_detector import HandLandmarks

logger = structlog.get_logger(__name__)


class HandTrackingError(Exception):
    """Raised when hand tracking fails."""


class OcclusionError(HandTrackingError):
    """Raised when hand is occluded and cannot be tracked."""


@dataclass
class TrackedHand:
    """Represents a continuously tracked hand with temporal information."""

    track_id: int
    landmarks: np.ndarray  # (21, 3) current smoothed landmarks
    raw_landmarks: np.ndarray  # (21, 3) raw detected landmarks
    handedness: str
    confidence: float
    roi: Tuple[int, int, int, int]
    velocity: np.ndarray  # (21, 3) landmark velocity
    history: deque  # deque of np.ndarray, limited size
    last_seen: float
    age: int = 0  # frames since first detection
    occlusion_frames: int = 0  # consecutive frames without detection
    is_predicted: bool = False  # True if using Kalman prediction

    def __post_init__(self):
        self.history = deque(maxlen=30)


class KalmanFilter2D:
    """Simple Kalman filter for 2D landmark tracking with constant velocity model.

    State: [x, y, vx, vy] for each landmark point.
    """

    def __init__(self, dt: float = 1.0 / 30.0, process_noise: float = 1e-3, measurement_noise: float = 1e-1):
        self._dt = dt
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        # State transition matrix
        self._F = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)
        # Measurement matrix
        self._H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=np.float32)
        # Process noise covariance
        self._Q = np.eye(4, dtype=np.float32) * process_noise
        # Measurement noise covariance
        self._R = np.eye(2, dtype=np.float32) * measurement_noise
        # State covariance
        self._P = np.eye(4, dtype=np.float32) * 10.0
        # Initialized flag
        self._initialized = False
        # State vector per landmark: list of (x, y, vx, vy)
        self._states: Optional[np.ndarray] = None  # (21, 4)

    def initialize(self, landmarks: np.ndarray) -> None:
        """Initialize filter with first observation.

        Args:
            landmarks: (21, 3) array.
        """
        n = landmarks.shape[0]
        self._states = np.zeros((n, 4), dtype=np.float32)
        self._states[:, 0] = landmarks[:, 0]
        self._states[:, 1] = landmarks[:, 1]
        self._initialized = True

    def predict(self) -> np.ndarray:
        """Predict next state.

        Returns:
            (21, 2) predicted (x, y) positions.
        """
        if self._states is None:
            raise RuntimeError("Kalman filter not initialized")

        self._states = self._states @ self._F.T
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self._states[:, :2].copy()

    def update(self, landmarks: np.ndarray) -> np.ndarray:
        """Update filter with new measurement.

        Args:
            landmarks: (21, 3) array.

        Returns:
            (21, 2) updated (x, y) positions.
        """
        if not self._initialized:
            self.initialize(landmarks)
            return landmarks[:, :2].copy()

        # Predict
        self.predict()

        # Update each landmark independently
        n = landmarks.shape[0]
        for i in range(n):
            z = landmarks[i, :2]
            x = self._states[i]
            # Innovation
            y = z - self._H @ x
            # Innovation covariance
            S = self._H @ self._P @ self._H.T + self._R
            # Kalman gain
            K = self._P @ self._H.T @ np.linalg.inv(S)
            # State update
            self._states[i] = x + K @ y
            # Covariance update
            self._P = (np.eye(4) - K @ self._H) @ self._P

        return self._states[:, :2].copy()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def reset(self) -> None:
        self._initialized = False
        self._states = None
        self._P = np.eye(4, dtype=np.float32) * 10.0


class HandTracker:
    """Continuous hand tracking with Kalman filtering, occlusion handling, and re-identification.

    Tracks multiple hands across frames using temporal smoothing and motion prediction.
    Handles occlusion by predicting positions and re-identifies hands when they reappear.

    Performance optimization:
    - Efficient Kalman filter per hand
    - Track history buffer for re-identification
    - Configurable max occlusion before track termination
    """

    REID_IOU_THRESHOLD: float = 0.3
    MAX_OCCLUSION_FRAMES: int = 30
    MAX_TRACK_HISTORY: int = 60
    LATENCY_COMPENSATION_FRAMES: int = 2

    def __init__(
        self,
        max_tracks: int = 4,
        iou_threshold: float = 0.3,
        max_occlusion_frames: int = 30,
        smoothing_window: int = 5,
        latency_compensation: bool = True,
    ) -> None:
        """Initialize hand tracker.

        Args:
            max_tracks: Maximum number of simultaneous tracks.
            iou_threshold: IoU threshold for re-identification.
            max_occlusion_frames: Frames to keep track after occlusion.
            smoothing_window: Temporal smoothing buffer size.
            latency_compensation: Enable motion prediction for latency.
        """
        self._max_tracks = max_tracks
        self._smoothing_window = smoothing_window
        self._latency_compensation = latency_compensation
        self.REID_IOU_THRESHOLD = iou_threshold
        self.MAX_OCCLUSION_FRAMES = max_occlusion_frames

        self._next_track_id: int = 0
        self._tracks: OrderedDict[int, TrackedHand] = OrderedDict()
        self._kalman_filters: Dict[int, KalmanFilter2D] = {}
        self._track_history: Dict[int, deque] = {}  # for re-identification

        logger.info(
            "hand_tracker_initialized",
            max_tracks=max_tracks,
            max_occlusion_frames=max_occlusion_frames,
        )

    def update(self, detections: List[HandLandmarks]) -> List[TrackedHand]:
        """Update tracker with new detections.

        Matches detections to existing tracks, handles new tracks,
        predicts for occluded tracks, and removes stale tracks.

        Args:
            detections: List of HandLandmarks from the detector.

        Returns:
            List of active TrackedHand objects.
        """
        # Predict existing tracks
        self._predict_all()

        # Match detections to tracks
        matched_idxs = self._match_detections_to_tracks(detections)

        # Update matched tracks
        used_detections = set()
        for track_id, det_idx in matched_idxs.items():
            detection = detections[det_idx]
            self._update_track(track_id, detection)
            used_detections.add(det_idx)
            self._tracks[track_id].occlusion_frames = 0
            self._tracks[track_id].is_predicted = False

        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                self._create_track(detection)

        # Increment occlusion counters and predict unmatched tracks
        for track_id in list(self._tracks.keys()):
            track = self._tracks[track_id]
            if track_id not in matched_idxs:
                track.occlusion_frames += 1
                track.is_predicted = True
                # Predict landmarks using Kalman filter
                if track_id in self._kalman_filters:
                    kf = self._kalman_filters[track_id]
                    predicted_xy = kf.predict()
                    track.landmarks = np.column_stack([
                        predicted_xy,
                        track.landmarks[:, 2]
                    ])

        # Remove stale tracks
        self._remove_stale_tracks()

        # Apply latency compensation if enabled
        if self._latency_compensation:
            self._compensate_latency()

        # Update ages
        for track in self._tracks.values():
            track.age += 1

        active_tracks = list(self._tracks.values())
        logger.debug(
            "tracker_update",
            active_tracks=len(active_tracks),
            total_tracks=len(self._tracks),
        )
        return active_tracks

    def get_track_by_id(self, track_id: int) -> Optional[TrackedHand]:
        """Get track by its unique identifier.

        Args:
            track_id: Track identifier.

        Returns:
            TrackedHand if found, None otherwise.
        """
        return self._tracks.get(track_id)

    def get_active_tracks(self) -> List[TrackedHand]:
        """Get all currently active tracks.

        Returns:
            List of active TrackedHand objects.
        """
        return list(self._tracks.values())

    def reset(self) -> None:
        """Reset all tracks and filters."""
        self._next_track_id = 0
        self._tracks.clear()
        self._kalman_filters.clear()
        self._track_history.clear()
        logger.info("hand_tracker_reset")

    def _predict_all(self) -> None:
        """Run Kalman prediction for all initialized filters."""
        for track_id, kf in self._kalman_filters.items():
            if kf.is_initialized:
                kf.predict()

    def _match_detections_to_tracks(
        self, detections: List[HandLandmarks]
    ) -> Dict[int, int]:
        """Match detections to existing tracks using handedness and IoU.

        Args:
            detections: List of detected hand landmarks.

        Returns:
            Dict mapping track_id -> detection_index.
        """
        if not detections or not self._tracks:
            return {}

        matched: Dict[int, int] = {}
        available_tracks = list(self._tracks.keys())
        used_detections: set = set()

        for track_id in available_tracks:
            track = self._tracks[track_id]
            best_iou = self.REID_IOU_THRESHOLD
            best_det = -1

            for i, det in enumerate(detections):
                if i in used_detections:
                    continue

                # Prefer same handedness
                if track.handedness != det.handedness:
                    continue

                iou = self._compute_iou(track.roi, det.roi)
                if iou > best_iou:
                    best_iou = iou
                    best_det = i

            if best_det >= 0:
                matched[track_id] = best_det
                used_detections.add(best_det)

        return matched

    def _create_track(self, detection: HandLandmarks) -> None:
        """Create a new track from a detection.

        Args:
            detection: HandLandmarks to initialize the track.
        """
        if len(self._tracks) >= self._max_tracks:
            # Evict oldest or lowest-confidence track
            oldest_id = next(iter(self._tracks))
            self._remove_track(oldest_id)

        track_id = self._next_track_id
        self._next_track_id += 1

        track = TrackedHand(
            track_id=track_id,
            landmarks=detection.landmarks.copy(),
            raw_landmarks=detection.landmarks.copy(),
            handedness=detection.handedness,
            confidence=detection.confidence,
            roi=detection.roi,
            velocity=np.zeros((21, 3), dtype=np.float32),
            last_seen=time.time(),
        )
        self._tracks[track_id] = track

        # Initialize Kalman filter
        kf = KalmanFilter2D()
        kf.initialize(detection.landmarks)
        self._kalman_filters[track_id] = kf

        # Initialize history
        self._track_history[track_id] = deque(maxlen=self.MAX_TRACK_HISTORY)

        logger.debug("new_track_created", track_id=track_id, handedness=detection.handedness)

    def _update_track(self, track_id: int, detection: HandLandmarks) -> None:
        """Update an existing track with a new detection.

        Args:
            track_id: Track to update.
            detection: New detection data.
        """
        track = self._tracks[track_id]
        kf = self._kalman_filters.get(track_id)

        # Compute velocity before updating
        if track.history:
            prev_landmarks = track.history[-1] if track.history else track.landmarks
            track.velocity = detection.landmarks - prev_landmarks

        # Store raw landmarks
        track.raw_landmarks = detection.landmarks.copy()
        track.confidence = detection.confidence
        track.roi = detection.roi
        track.last_seen = time.time()

        # Kalman filter update for smoothing
        if kf is not None:
            updated_xy = kf.update(detection.landmarks)
            track.landmarks = np.column_stack([
                updated_xy,
                detection.landmarks[:, 2]
            ])
        else:
            track.landmarks = detection.landmarks.copy()

        # Temporal smoothing
        track.history.append(track.landmarks.copy())
        if len(track.history) >= self._smoothing_window:
            smoothed = np.mean(list(track.history)[-self._smoothing_window:], axis=0)
            track.landmarks = smoothed

        # Update track history for re-id
        self._track_history[track_id].append(detection.landmarks.copy())

    def _remove_track(self, track_id: int) -> None:
        """Remove a track and its associated resources.

        Args:
            track_id: Track to remove.
        """
        self._tracks.pop(track_id, None)
        self._kalman_filters.pop(track_id, None)
        self._track_history.pop(track_id, None)

    def _remove_stale_tracks(self) -> None:
        """Remove tracks that have been occluded for too long."""
        stale_ids = [
            tid
            for tid, track in self._tracks.items()
            if track.occlusion_frames > self.MAX_OCCLUSION_FRAMES
        ]
        for tid in stale_ids:
            logger.debug("removing_stale_track", track_id=tid, occlusion_frames=self._tracks[tid].occlusion_frames)
            self._remove_track(tid)

    def _compensate_latency(self) -> None:
        """Predict future positions to compensate for processing latency.

        Uses velocity estimates to extrapolate landmark positions by
        LATENCY_COMPENSATION_FRAMES into the future.
        """
        for track in self._tracks.values():
            if np.linalg.norm(track.velocity) > 1e-6:
                compensation = track.velocity * self.LATENCY_COMPENSATION_FRAMES
                track.landmarks = track.landmarks + compensation

    def _compute_iou(
        self, roi_a: Tuple[int, int, int, int], roi_b: Tuple[int, int, int, int]
    ) -> float:
        """Compute Intersection over Union between two ROIs.

        Args:
            roi_a: (x, y, w, h) first bounding box.
            roi_b: (x, y, w, h) second bounding box.

        Returns:
            IoU value between 0 and 1.
        """
        x1_a, y1_a, w_a, h_a = roi_a
        x2_a, y2_a = x1_a + w_a, y1_a + h_a
        x1_b, y1_b, w_b, h_b = roi_b
        x2_b, y2_b = x1_b + w_b, y1_b + h_b

        xi1 = max(x1_a, x1_b)
        yi1 = max(y1_a, y1_b)
        xi2 = min(x2_a, x2_b)
        yi2 = min(y2_a, y2_b)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area_a = w_a * h_a
        area_b = w_b * h_b
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area
