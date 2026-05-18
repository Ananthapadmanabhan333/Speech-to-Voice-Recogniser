from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = structlog.get_logger(__name__)


class FacialEmotionError(Exception):
    """Raised when facial emotion analysis fails."""


@dataclass
class EmotionResult:
    """Result of emotion analysis."""

    emotion: str
    confidence: float
    emotion_probs: Dict[str, float]
    action_units: Dict[str, float]
    expression_intensity: float  # 0 to 1
    head_pose: Tuple[float, float, float]  # yaw, pitch, roll
    landmarks: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)


# 7 basic emotions + compound emotions
EMOTIONS: List[str] = [
    "neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted",
    "contempt", "confused", "pain",
]

# Action Units (FACS-based)
ACTION_UNITS: List[str] = [
    "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10",
    "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25",
    "AU26", "AU27", "AU43",
]


class FacialEmotionCNN(nn.Module):
    """CNN for facial emotion classification from landmarks."""

    def __init__(
        self,
        num_landmarks: int = 478,
        landmark_dim: int = 3,
        num_emotions: int = 10,
        hidden_dim: int = 256,
    ):
        super().__init__()
        input_dim = num_landmarks * landmark_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_emotions),
        )

        # AU prediction head
        self.au_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(ACTION_UNITS)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.network[:6](x)  # Up to the first dropout in last block
        emotion_logits = self.network[6:](features)
        au_probs = self.au_head(features)
        return emotion_logits, au_probs


class FacialEmotionAnalyzer:
    """Facial emotion recognition using MediaPipe Face Mesh and emotion CNN.

    Extracts facial landmarks using MediaPipe, then classifies emotion using
    a neural network. Also detects Action Units (FACS), estimates expression
    intensity, and computes head pose.

    Features:
    - 7 basic emotions + compound emotions (confused, pain, contempt)
    - Facial Action Unit detection (19 AUs)
    - Expression intensity estimation
    - Head pose estimation (yaw, pitch, roll)
    - Real-time performance with frame skip
    """

    NUM_FACE_LANDMARKS: int = 478
    LANDMARK_DIMS: int = 3

    # MediaPipe face mesh landmark indices for head pose
    HEAD_POSE_INDICES: List[int] = [1, 33, 61, 199, 263, 291]

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        emotion_model_path: Optional[str] = None,
        device: Optional[str] = None,
        skip_rate: int = 2,
    ):
        """Initialize facial emotion analyzer.

        Args:
            min_detection_confidence: Min confidence for face detection.
            min_tracking_confidence: Min confidence for face tracking.
            emotion_model_path: Path to emotion classification model.
            device: Device to run on.
            skip_rate: Process every (skip_rate+1)th frame.
        """
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._skip_rate = skip_rate
        self._frame_count: int = 0

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # MediaPipe Face Mesh
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )

        # Emotion classifier
        self._classifier = FacialEmotionCNN(
            num_landmarks=self.NUM_FACE_LANDMARKS,
            num_emotions=len(EMOTIONS),
        ).to(self._device)

        self._eval_mode = False

        if emotion_model_path:
            self.load(emotion_model_path)

        # Head pose estimation (3D model points)
        self._head_pose_model_points = np.array([
            [0.0, 0.0, 0.0],       # Nose tip
            [0.0, -330.0, -65.0],   # Chin
            [-225.0, 170.0, -135.0], # Left eye left corner
            [225.0, 170.0, -135.0],  # Right eye right corner
            [-150.0, -150.0, -125.0], # Left mouth corner
            [150.0, -150.0, -125.0],  # Right mouth corner
        ], dtype=np.float32)

        logger.info(
            "facial_emotion_analyzer_initialized",
            device=str(self._device),
            emotions=EMOTIONS,
        )

    def analyze_emotion(self, frame: np.ndarray) -> EmotionResult:
        """Analyze facial emotion from an image frame.

        Args:
            frame: Input BGR image frame.

        Returns:
            EmotionResult with detected emotion and analysis.

        Raises:
            FacialEmotionError: If analysis fails.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame provided")

        self._frame_count += 1

        # Frame skip
        if self._skip_rate > 0 and self._frame_count % (self._skip_rate + 1) != 0:
            # Return cached result or neutral
            return self._get_cached_or_neutral()

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb_frame)

            if not results or not results.multi_face_landmarks:
                raise FacialEmotionError("No face detected in frame")

            face_landmarks = results.multi_face_landmarks[0]

            # Extract landmarks
            landmarks = self._extract_landmarks(face_landmarks)

            # Classify emotion
            emotion, emotion_probs = self._classify_emotion(landmarks)

            # Detect action units
            action_units = self._detect_action_units(face_landmarks)

            # Estimate expression intensity
            intensity = self._estimate_intensity(emotion_probs)

            # Estimate head pose
            head_pose = self._estimate_head_pose(face_landmarks, frame.shape)

            return EmotionResult(
                emotion=emotion,
                confidence=emotion_probs[emotion],
                emotion_probs=emotion_probs,
                action_units=action_units,
                expression_intensity=intensity,
                head_pose=head_pose,
                landmarks=landmarks,
            )

        except FacialEmotionError:
            raise
        except Exception as e:
            logger.error("facial_emotion_analysis_failed", error=str(e))
            raise FacialEmotionError(f"Facial emotion analysis failed: {e}") from e

    def extract_landmarks(self, face_landmarks) -> np.ndarray:
        """Extract face mesh landmarks as numpy array.

        Args:
            face_landmarks: MediaPipe face landmark list.

        Returns:
            Array of shape (NUM_FACE_LANDMARKS, LANDMARK_DIMS).
        """
        return self._extract_landmarks(face_landmarks)

    def detect_action_units(self, face_landmarks) -> Dict[str, float]:
        """Detect Facial Action Units from landmarks.

        Args:
            face_landmarks: MediaPipe face landmark list.

        Returns:
            Dict of AU -> activation probability.
        """
        return self._detect_action_units(face_landmarks)

    def estimate_head_pose(
        self, face_landmarks, frame_shape: Tuple[int, int, int]
    ) -> Tuple[float, float, float]:
        """Estimate head pose (yaw, pitch, roll).

        Args:
            face_landmarks: MediaPipe face landmark list.
            frame_shape: (H, W, C) frame dimensions.

        Returns:
            (yaw, pitch, roll) in degrees.
        """
        return self._estimate_head_pose(face_landmarks, frame_shape)

    def load(self, model_path: str) -> None:
        """Load emotion classification model.

        Args:
            model_path: Path to checkpoint.
        """
        checkpoint = torch.load(model_path, map_location=self._device)
        self._classifier.load_state_dict(checkpoint["classifier"])
        logger.info("facial_emotion_model_loaded", path=model_path)

    def save(self, model_path: str) -> None:
        """Save emotion classification model.

        Args:
            model_path: Path to save checkpoint.
        """
        torch.save({"classifier": self._classifier.state_dict()}, model_path)
        logger.info("facial_emotion_model_saved", path=model_path)

    def _extract_landmarks(self, face_landmarks) -> np.ndarray:
        """Extract face mesh landmarks.

        Args:
            face_landmarks: MediaPipe face landmark list.

        Returns:
            Array of shape (NUM_FACE_LANDMARKS, LANDMARK_DIMS).
        """
        landmarks = np.zeros((self.NUM_FACE_LANDMARKS, self.LANDMARK_DIMS), dtype=np.float32)
        for i in range(min(self.NUM_FACE_LANDMARKS, len(face_landmarks.landmark))):
            lm = face_landmarks.landmark[i]
            landmarks[i, 0] = lm.x
            landmarks[i, 1] = lm.y
            landmarks[i, 2] = lm.z
        return landmarks

    def _classify_emotion(
        self, landmarks: np.ndarray
    ) -> Tuple[str, Dict[str, float]]:
        """Classify emotion from normalized landmarks.

        Args:
            landmarks: (478, 3) landmark array.

        Returns:
            (emotion_label, emotion_probabilities).
        """
        # Normalize landmarks
        normalized = self._normalize_landmarks(landmarks)

        self._ensure_eval_mode()
        with torch.no_grad():
            input_tensor = torch.from_numpy(normalized).float().unsqueeze(0).to(self._device)
            emotion_logits, _ = self._classifier(input_tensor)
            probs = F.softmax(emotion_logits, dim=-1).cpu().numpy()[0]

        emotion_probs = {
            EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))
        }

        pred_idx = int(np.argmax(probs))
        emotion = EMOTIONS[pred_idx]

        return emotion, emotion_probs

    def _detect_action_units(self, face_landmarks) -> Dict[str, float]:
        """Detect action units using geometric rules.

        Args:
            face_landmarks: MediaPipe face landmark list.

        Returns:
            Dict of AU -> activation probability.
        """
        # Simplified geometric AU detection
        # In production, use a trained AU detection model
        au_values: Dict[str, float] = {au: 0.0 for au in ACTION_UNITS}

        if not face_landmarks or not face_landmarks.landmark:
            return au_values

        lm = face_landmarks.landmark

        # AU12 (Lip Corner Puller) - smile detection
        if len(lm) > 61 and len(lm) > 291:
            left_mouth = np.array([lm[61].x, lm[61].y])
            right_mouth = np.array([lm[291].x, lm[291].y])
            mouth_width = np.linalg.norm(right_mouth - left_mouth)
            au_values["AU12"] = min(1.0, mouth_width * 3)

        # AU4 (Brow Lowerer) - based on brow position
        if len(lm) > 105 and len(lm) > 334:
            left_brow_y = lm[105].y
            right_brow_y = lm[334].y
            brow_position = (left_brow_y + right_brow_y) / 2
            au_values["AU4"] = min(1.0, max(0.0, (brow_position - 0.4) * 5))

        # AU25 (Lips Part) - mouth opening
        if len(lm) > 13 and len(lm) > 14:
            mouth_open = abs(lm[13].y - lm[14].y)
            au_values["AU25"] = min(1.0, mouth_open * 10)

        # AU27 (Mouth Stretch) - extreme mouth opening
        if len(lm) > 17 and len(lm) > 16:
            mouth_stretch = abs(lm[17].y - lm[16].y)
            au_values["AU27"] = min(1.0, mouth_stretch * 8)

        return au_values

    def _estimate_intensity(self, emotion_probs: Dict[str, float]) -> float:
        """Estimate expression intensity from emotion probabilities.

        Higher confidence and more extreme emotions indicate higher intensity.

        Args:
            emotion_probs: Emotion probabilities.

        Returns:
            Intensity score in [0, 1].
        """
        max_conf = max(emotion_probs.values())
        # Intensity is higher when confidence is high AND emotion is not neutral
        neutral_conf = emotion_probs.get("neutral", 0.0)
        intensity = max_conf * (1.0 - neutral_conf)
        return float(np.clip(intensity, 0.0, 1.0))

    def _estimate_head_pose(
        self, face_landmarks, frame_shape: Tuple[int, int, int]
    ) -> Tuple[float, float, float]:
        """Estimate head pose using PnP.

        Args:
            face_landmarks: MediaPipe face landmark list.
            frame_shape: (H, W, C).

        Returns:
            (yaw, pitch, roll) in degrees.
        """
        h, w, _ = frame_shape

        if not face_landmarks or not face_landmarks.landmark:
            return (0.0, 0.0, 0.0)

        lm = face_landmarks.landmark

        # Get 2D image points from key landmarks
        image_points = []
        for idx in [1, 199, 33, 263, 61, 291]:  # Nose, chin, eyes, mouth corners
            if idx < len(lm):
                image_points.append([lm[idx].x * w, lm[idx].y * h])

        if len(image_points) < 4:
            return (0.0, 0.0, 0.0)

        image_points = np.array(image_points, dtype=np.float32)

        # Camera matrix
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1))

        try:
            _, rotation_vector, translation_vector = cv2.solvePnP(
                self._head_pose_model_points[:len(image_points)],
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            rmat, _ = cv2.Rodrigues(rotation_vector)
            yaw = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))
            pitch = float(np.degrees(np.arcsin(-rmat[2, 0])))
            roll = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))

            return (yaw, pitch, roll)

        except cv2.error:
            return (0.0, 0.0, 0.0)

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks for classification.

        Centers and scales the landmarks.

        Args:
            landmarks: (478, 3) array.

        Returns:
            Normalized (478 * 3,) array.
        """
        # Center
        center = np.mean(landmarks, axis=0)
        normalized = landmarks - center

        # Scale
        scale = np.max(np.linalg.norm(normalized, axis=1))
        if scale > 1e-6:
            normalized = normalized / scale

        return normalized.flatten()

    def _get_cached_or_neutral(self) -> EmotionResult:
        """Return neutral result when frame is skipped.

        Returns:
            Default neutral EmotionResult.
        """
        return EmotionResult(
            emotion="neutral",
            confidence=0.5,
            emotion_probs={e: 0.1 for e in EMOTIONS},
            action_units={au: 0.0 for au in ACTION_UNITS},
            expression_intensity=0.0,
            head_pose=(0.0, 0.0, 0.0),
        )

    def _ensure_eval_mode(self) -> None:
        if not self._eval_mode:
            self._classifier.eval()
            self._eval_mode = True
