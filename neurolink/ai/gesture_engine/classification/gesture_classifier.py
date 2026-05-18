from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = structlog.get_logger(__name__)


class GestureClassificationError(Exception):
    """Raised when gesture classification fails."""


class UnknownGestureError(GestureClassificationError):
    """Raised when the gesture is not recognized."""


@dataclass
class GestureResult:
    """Result of gesture classification."""

    gesture_id: str
    gesture_label: str
    confidence: float
    calibrated_confidence: float
    raw_logits: np.ndarray
    timestamp: float = field(default_factory=time.time)


class TemporalConvBlock(nn.Module):
    """Temporal convolutional block for feature extraction."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.bn(self.conv(x))))


class TemporalCNN(nn.Module):
    """Temporal CNN for landmark sequence feature extraction."""

    def __init__(self, input_dim: int = 63, hidden_dims: List[int] = None, kernel_size: int = 3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 256, 128]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(TemporalConvBlock(prev_dim, h_dim, kernel_size))
            prev_dim = h_dim
        self.cnn = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        # (batch, output_dim, seq_len) -> (batch, seq_len, output_dim)
        x = x.transpose(1, 2)
        return x


class GestureLSTM(nn.Module):
    """LSTM for temporal gesture classification."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last_out = lstm_out[:, -1, :]
        logits = self.fc(last_out)
        return logits


class GestureClassifier:
    """Temporal CNN + LSTM gesture classifier with known ASL gestures and custom registration.

    Classifies sequences of hand landmarks into gesture categories.
    Supports 26 ASL letters (A-Z) and common words.
    Uses temperature scaling for confidence calibration.

    Known gestures (built-in):
    - ASL letters: A through Z
    - Common words: hello, thanks, yes, no, help, please, sorry, good, bad,
                    stop, go, wait, come, eat, drink, bathroom, pain, medicine,
                    where, what, who, how, why, when, home, love, friend, family
    """

    ASL_LETTERS: List[str] = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    COMMON_WORDS: List[str] = [
        "hello", "thanks", "thank_you", "yes", "no", "help", "please", "sorry",
        "good", "bad", "stop", "go", "wait", "come", "eat", "drink", "bathroom",
        "pain", "medicine", "where", "what", "who", "how", "why", "when", "home",
        "love", "friend", "family", "emergency", "hurt", "hospital", "doctor",
        "water", "food", "tired", "happy", "sad", "angry", "scared",
    ]
    DEFAULT_GESTURES: List[str] = ASL_LETTERS + COMMON_WORDS

    MIN_SEQUENCE_LENGTH: int = 5
    MAX_SEQUENCE_LENGTH: int = 150
    LANDMARK_DIM: int = 63  # 21 landmarks * 3 coords
    DEFAULT_TEMPERATURE: float = 1.5

    def __init__(
        self,
        num_classes: Optional[int] = None,
        hidden_dim: int = 256,
        num_layers: int = 2,
        temperature: float = 1.5,
        device: Optional[str] = None,
        model_path: Optional[Path] = None,
    ):
        """Initialize gesture classifier.

        Args:
            num_classes: Number of gesture classes. Defaults to len(DEFAULT_GESTURES).
            hidden_dim: LSTM hidden dimension.
            num_layers: Number of LSTM layers.
            temperature: Temperature scaling parameter for calibration.
            device: Device to run on ('cpu', 'cuda', 'mps').
            model_path: Path to load pretrained model from.

        Raises:
            ValueError: If parameters are invalid.
        """
        if num_classes is None:
            num_classes = len(self.DEFAULT_GESTURES)

        self._num_classes = num_classes
        self._hidden_dim = hidden_dim
        self._temperature = temperature

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Gesture label mapping
        self._gesture_labels: List[str] = list(self.DEFAULT_GESTURES[:num_classes])
        self._label_to_id: Dict[str, int] = {
            label: i for i, label in enumerate(self._gesture_labels)
        }
        self._id_to_label: Dict[int, str] = {
            i: label for i, label in enumerate(self._gesture_labels)
        }

        # Custom gesture registry
        self._custom_gestures: Dict[str, np.ndarray] = {}  # label -> prototype embedding

        # Build model
        self._temporal_cnn = TemporalCNN(
            input_dim=self.LANDMARK_DIM,
            hidden_dims=[128, 256, 128],
        ).to(self._device)

        self._lstm = GestureLSTM(
            input_dim=128,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
        ).to(self._device)

        self._eval_mode = False

        if model_path is not None:
            self.load(model_path)

        logger.info(
            "gesture_classifier_initialized",
            num_classes=num_classes,
            device=str(self._device),
            temperature=temperature,
        )

    def classify_gesture(self, landmark_sequence: np.ndarray) -> GestureResult:
        """Classify a sequence of hand landmarks into a gesture.

        Args:
            landmark_sequence: np.ndarray of shape (seq_len, 21, 3).

        Returns:
            GestureResult with gesture_id, label, confidence, and calibrated confidence.

        Raises:
            GestureClassificationError: If classification fails.
            ValueError: If landmark_sequence is invalid.
        """
        if landmark_sequence is None or landmark_sequence.size == 0:
            raise ValueError("Empty landmark sequence")

        seq_len = landmark_sequence.shape[0]
        if seq_len < self.MIN_SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequence too short: {seq_len} < {self.MIN_SEQUENCE_LENGTH}"
            )
        if seq_len > self.MAX_SEQUENCE_LENGTH:
            landmark_sequence = landmark_sequence[-self.MAX_SEQUENCE_LENGTH:]

        try:
            # Extract features
            features = self._extract_features(landmark_sequence)
            # features: (1, seq_len, 63)

            self._ensure_eval_mode()

            with torch.no_grad():
                features_tensor = torch.from_numpy(features).float().to(self._device)
                cnn_out = self._temporal_cnn(features_tensor)
                logits = self._lstm(cnn_out)
                logits_np = logits.cpu().numpy()[0]

            # Temperature scaling
            calibrated_logits = logits_np / self._temperature
            probabilities = F.softmax(torch.from_numpy(calibrated_logits), dim=0).numpy()

            # Get top prediction
            pred_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[pred_idx])

            # Check custom gestures
            custom_label, custom_conf = self._check_custom_gestures(features[0])
            if custom_conf > confidence:
                return GestureResult(
                    gesture_id=custom_label,
                    gesture_label=custom_label,
                    confidence=custom_conf,
                    calibrated_confidence=custom_conf,
                    raw_logits=logits_np,
                )

            gesture_label = self._id_to_label.get(pred_idx, "unknown")
            calibrated_conf = self._calibrate_confidence(confidence, logits_np, pred_idx)

            return GestureResult(
                gesture_id=gesture_label,
                gesture_label=gesture_label,
                confidence=confidence,
                calibrated_confidence=calibrated_conf,
                raw_logits=logits_np,
            )

        except Exception as e:
            logger.error("gesture_classification_failed", error=str(e))
            raise GestureClassificationError(f"Classification failed: {e}") from e

    def register_custom_gesture(
        self, label: str, prototype_sequences: List[np.ndarray]
    ) -> None:
        """Register a custom gesture with prototype landmark sequences.

        Args:
            label: Unique gesture label.
            prototype_sequences: List of landmark sequences for the gesture.

        Raises:
            ValueError: If label already exists or sequences are invalid.
        """
        if label in self._label_to_id:
            raise ValueError(f"Gesture label '{label}' already exists in built-in set")
        if not prototype_sequences:
            raise ValueError("Must provide at least one prototype sequence")

        # Compute average prototype embedding
        embeddings = []
        for seq in prototype_sequences:
            if seq.shape[-2:] != (21, 3):
                raise ValueError(f"Invalid shape: {seq.shape}, expected (seq_len, 21, 3)")
            features = self._extract_features(seq)
            self._ensure_eval_mode()
            with torch.no_grad():
                feat_tensor = torch.from_numpy(features).float().to(self._device)
                emb = self._temporal_cnn(feat_tensor).mean(dim=1).cpu().numpy()[0]
                embeddings.append(emb)

        prototype_emb = np.mean(embeddings, axis=0)
        prototype_emb = prototype_emb / (np.linalg.norm(prototype_emb) + 1e-8)

        self._custom_gestures[label] = prototype_emb
        logger.info("custom_gesture_registered", label=label, num_prototypes=len(prototype_sequences))

    def remove_custom_gesture(self, label: str) -> None:
        """Remove a registered custom gesture.

        Args:
            label: Gesture label to remove.
        """
        self._custom_gestures.pop(label, None)
        logger.info("custom_gesture_removed", label=label)

    def get_gesture_labels(self) -> List[str]:
        """Get list of all known gesture labels.

        Returns:
            List of gesture label strings.
        """
        return list(self._gesture_labels)

    def load(self, model_path: Path) -> None:
        """Load model weights from a checkpoint.

        Args:
            model_path: Path to the checkpoint file.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self._device)
        self._temporal_cnn.load_state_dict(checkpoint["temporal_cnn"])
        self._lstm.load_state_dict(checkpoint["lstm"])
        if "temperature" in checkpoint:
            self._temperature = checkpoint["temperature"]
        if "gesture_labels" in checkpoint:
            self._gesture_labels = checkpoint["gesture_labels"]
        logger.info("model_loaded", path=str(model_path))

    def save(self, model_path: Path) -> None:
        """Save model weights to a checkpoint.

        Args:
            model_path: Path to save the checkpoint.
        """
        model_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "temporal_cnn": self._temporal_cnn.state_dict(),
            "lstm": self._lstm.state_dict(),
            "temperature": self._temperature,
            "gesture_labels": self._gesture_labels,
        }
        torch.save(checkpoint, model_path)
        logger.info("model_saved", path=str(model_path))

    def save_labels(self, labels_path: Path) -> None:
        """Save gesture labels mapping to JSON.

        Args:
            labels_path: Path to save labels.
        """
        data = {
            "gesture_labels": self._gesture_labels,
            "label_to_id": self._label_to_id,
            "custom_gestures": list(self._custom_gestures.keys()),
        }
        with open(labels_path, "w") as f:
            json.dump(data, f, indent=2)

    def _extract_features(self, landmark_sequence: np.ndarray) -> np.ndarray:
        """Extract features from landmark sequence.

        Computes normalized landmark positions, velocities, and distances.

        Args:
            landmark_sequence: (seq_len, 21, 3).

        Returns:
            (1, seq_len, 63) feature array.
        """
        seq_len = landmark_sequence.shape[0]
        features = np.zeros((seq_len, self.LANDMARK_DIM), dtype=np.float32)

        for t in range(seq_len):
            frame = landmark_sequence[t]
            # Flatten 21x3 -> 63
            features[t] = frame.flatten()

        # Normalize per frame
        for t in range(seq_len):
            frame = features[t].reshape(21, 3)
            # Center at wrist (landmark 0)
            frame[:, :2] -= frame[0:1, :2]
            # Scale by wrist-to-middle-mcp distance (landmarks 0 and 9)
            scale = np.linalg.norm(frame[9, :2] - frame[0, :2])
            if scale > 1e-6:
                frame[:, :2] /= scale
            features[t] = frame.flatten()

        # Add velocity features as delta
        if seq_len > 1:
            velocity = np.diff(features, axis=0)
            velocity = np.pad(velocity, ((1, 0), (0, 0)), mode="edge")
            features = np.concatenate([features, velocity], axis=-1)
        else:
            features = np.concatenate([features, np.zeros_like(features)], axis=-1)

        # Pad or truncate to MAX_SEQUENCE_LENGTH
        if seq_len < self.MAX_SEQUENCE_LENGTH:
            pad = self.MAX_SEQUENCE_LENGTH - seq_len
            features = np.pad(features, ((0, pad), (0, 0)), mode="edge")
        else:
            features = features[: self.MAX_SEQUENCE_LENGTH]

        return features[np.newaxis, :, :63]  # Return base features only (no velocity)

    def _check_custom_gestures(self, features: np.ndarray) -> Tuple[str, float]:
        """Check if features match any registered custom gesture.

        Args:
            features: Feature vector at current timestep (63,).

        Returns:
            (label, confidence) tuple. Empty string and 0.0 if no match.
        """
        if not self._custom_gestures:
            return ("", 0.0)

        self._ensure_eval_mode()
        with torch.no_grad():
            feat_tensor = torch.from_numpy(features).float().to(self._device)
            feat_tensor = feat_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 63)
            emb = self._temporal_cnn(feat_tensor).mean(dim=1).cpu().numpy()[0]

        best_label = ""
        best_score = 0.0

        for label, prototype in self._custom_gestures.items():
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            proto_norm = prototype / (np.linalg.norm(prototype) + 1e-8)
            similarity = float(np.dot(emb_norm, proto_norm))
            confidence = max(0.0, similarity)  # Clamp to [0, 1]
            if confidence > best_score:
                best_score = confidence
                best_label = label

        return (best_label, best_score)

    def _calibrate_confidence(self, confidence: float, logits: np.ndarray, pred_idx: int) -> float:
        """Apply temperature scaling calibration.

        Args:
            confidence: Raw softmax confidence.
            logits: Raw logits.
            pred_idx: Predicted class index.

        Returns:
            Calibrated confidence value.
        """
        calibrated = F.softmax(
            torch.from_numpy(logits / self._temperature), dim=0
        ).numpy()
        return float(calibrated[pred_idx])

    def _ensure_eval_mode(self) -> None:
        """Ensure model is in evaluation mode."""
        if not self._eval_mode:
            self._temporal_cnn.eval()
            self._lstm.eval()
            self._eval_mode = True
