from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = structlog.get_logger(__name__)


class VocalEmotionError(Exception):
    """Raised when vocal emotion analysis fails."""


@dataclass
class EmotionResult:
    """Result of emotion analysis."""

    emotion: str
    confidence: float
    arousal: float  # -1 (calm) to 1 (excited)
    valence: float  # -1 (negative) to 1 (positive)
    dominance: float  # -1 (submissive) to 1 (dominant)
    stress_level: float  # 0 to 1
    confusion_level: float  # 0 to 1
    emotion_probs: Dict[str, float]
    features: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EmotionClassifier(nn.Module):
    """Simple MLP for emotion classification from acoustic features."""

    def __init__(
        self,
        input_dim: int,
        num_emotions: int,
        hidden_dims: List[int] = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_emotions))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class VocalEmotionAnalyzer:
    """Vocal emotion analysis using prosodic and spectral features.

    Analyzes audio to detect emotion, arousal, valence, stress, and confusion
    levels. Uses MFCCs, spectral features, and prosodic features (pitch, energy,
    speaking rate).

    Emotions detected:
    - neutral, happy, sad, angry, fearful, surprised, disgusted

    Performance:
    - Feature extraction: ~10ms for 1s audio
    - Classification: ~5ms on GPU
    """

    EMOTIONS: List[str] = [
        "neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted",
    ]

    NUM_EMOTIONS: int = len(EMOTIONS)
    NUM_MFCC: int = 40
    FEATURE_DIM: int = NUM_MFCC + 5  # MFCC + prosodic features

    def __init__(
        self,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        sample_rate: int = 16000,
        confidence_threshold: float = 0.3,
    ):
        """Initialize vocal emotion analyzer.

        Args:
            device: Device to run on ('cpu', 'cuda').
            model_path: Path to pretrained emotion classification model.
            sample_rate: Target sample rate.
            confidence_threshold: Minimum confidence for emotion prediction.
        """
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._sample_rate = sample_rate
        self._confidence_threshold = confidence_threshold

        # Emotion classifier
        self._classifier = EmotionClassifier(
            input_dim=self.FEATURE_DIM,
            num_emotions=self.NUM_EMOTIONS,
        ).to(self._device)

        self._eval_mode = False

        if model_path:
            self.load(model_path)

        logger.info(
            "vocal_emotion_analyzer_initialized",
            device=str(self._device),
            emotions=self.EMOTIONS,
        )

    def analyze_emotion(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        **kwargs: Any,
    ) -> EmotionResult:
        """Analyze emotion from audio signal.

        Args:
            audio: Audio signal as numpy array (mono).
            sample_rate: Sample rate of the audio.
            **kwargs: Additional analysis parameters.

        Returns:
            EmotionResult with detected emotion and dimensions.

        Raises:
            VocalEmotionError: If analysis fails.
        """
        if audio is None or len(audio) == 0:
            raise ValueError("Empty audio input")

        try:
            # Preprocess
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sample_rate != self._sample_rate:
                audio = self._resample(audio, sample_rate, self._sample_rate)

            # Extract features
            features = self._extract_features(audio)

            # Classify emotion
            emotion_probs = self._classify_emotion(features)

            # Compute emotion dimensions
            arousal = self._compute_arousal(emotion_probs)
            valence = self._compute_valence(emotion_probs)
            dominance = self._compute_dominance(emotion_probs)
            stress_level = self._detect_stress(features, audio)
            confusion_level = self._detect_confusion(features, emotion_probs)

            # Get top emotion
            emotion_idx = int(np.argmax(list(emotion_probs.values())))
            emotion = self.EMOTIONS[emotion_idx]
            confidence = emotion_probs[emotion]

            # Build feature dict
            feature_dict = self._build_feature_dict(features)

            return EmotionResult(
                emotion=emotion,
                confidence=confidence,
                arousal=arousal,
                valence=valence,
                dominance=dominance,
                stress_level=stress_level,
                confusion_level=confusion_level,
                emotion_probs=emotion_probs,
                features=feature_dict,
            )

        except Exception as e:
            logger.error("vocal_emotion_analysis_failed", error=str(e))
            raise VocalEmotionError(f"Vocal emotion analysis failed: {e}") from e

    def extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract prosodic features from audio.

        Args:
            audio: Audio signal (mono, 16kHz).

        Returns:
            Dict with pitch, energy, speaking rate features.
        """
        try:
            import librosa

            # Pitch (F0)
            f0, voiced_flag, _ = librosa.pyin(
                audio, fmin=80, fmax=600, sr=self._sample_rate
            )
            f0 = f0[~np.isnan(f0)]
            pitch_mean = float(np.mean(f0)) if len(f0) > 0 else 0.0
            pitch_std = float(np.std(f0)) if len(f0) > 0 else 0.0
            pitch_range = float(np.ptp(f0)) if len(f0) > 1 else 0.0

            # Energy
            rms = librosa.feature.rms(y=audio)[0]
            energy_mean = float(np.mean(rms))
            energy_std = float(np.std(rms))
            energy_range = float(np.ptp(rms))

            # Speaking rate (estimated via onset density)
            onset_env = librosa.onset.onset_strength(y=audio, sr=self._sample_rate)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self._sample_rate)
            speaking_rate = len(onsets) / (len(audio) / self._sample_rate) if len(audio) > 0 else 0.0

        except ImportError:
            # Fallback: simple time-domain estimates
            pitch_mean = 0.0
            pitch_std = 0.0
            pitch_range = 0.0
            energy_mean = float(np.mean(np.abs(audio)))
            energy_std = float(np.std(np.abs(audio)))
            energy_range = energy_mean * 2
            speaking_rate = 3.0  # default

        return {
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_range": pitch_range,
            "energy_mean": energy_mean,
            "energy_std": energy_std,
            "energy_range": energy_range,
            "speaking_rate": speaking_rate,
        }

    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral features from audio.

        Args:
            audio: Audio signal (mono, 16kHz).

        Returns:
            Dict with spectral features.
        """
        try:
            import librosa

            # MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=self._sample_rate, n_mfcc=self.NUM_MFCC)
            mfcc_mean = np.mean(mfcc, axis=1)

            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self._sample_rate)[0]
            spectral_centroid_mean = float(np.mean(spectral_centroid))

            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self._sample_rate)[0]
            spectral_bandwidth_mean = float(np.mean(spectral_bandwidth))

            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self._sample_rate)[0]
            spectral_rolloff_mean = float(np.mean(spectral_rolloff))

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_mean = float(np.mean(zcr))

            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=self._sample_rate)
            mel_mean = float(np.mean(mel_spec))
            mel_std = float(np.std(mel_spec))

        except ImportError:
            mfcc_mean = np.zeros(self.NUM_MFCC)
            spectral_centroid_mean = 0.0
            spectral_bandwidth_mean = 0.0
            spectral_rolloff_mean = 0.0
            zcr_mean = 0.0
            mel_mean = 0.0
            mel_std = 0.0

        result = {
            "spectral_centroid": spectral_centroid_mean,
            "spectral_bandwidth": spectral_bandwidth_mean,
            "spectral_rolloff": spectral_rolloff_mean,
            "zero_crossing_rate": zcr_mean,
            "mel_mean": mel_mean,
            "mel_std": mel_std,
        }

        # Add MFCC values
        for i in range(self.NUM_MFCC):
            result[f"mfcc_{i}"] = float(mfcc_mean[i]) if i < len(mfcc_mean) else 0.0

        return result

    def load(self, model_path: str) -> None:
        """Load emotion classifier model.

        Args:
            model_path: Path to model checkpoint.
        """
        checkpoint = torch.load(model_path, map_location=self._device)
        self._classifier.load_state_dict(checkpoint["classifier"])
        logger.info("vocal_emotion_model_loaded", path=model_path)

    def save(self, model_path: str) -> None:
        """Save emotion classifier model.

        Args:
            model_path: Path to save checkpoint.
        """
        torch.save({"classifier": self._classifier.state_dict()}, model_path)
        logger.info("vocal_emotion_model_saved", path=model_path)

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract combined feature vector from audio.

        Args:
            audio: Audio signal (mono, 16kHz).

        Returns:
            Feature vector of shape (FEATURE_DIM,).
        """
        spectral = self.extract_spectral_features(audio)
        prosodic = self.extract_prosodic_features(audio)

        # MFCC vector
        mfcc = np.array([spectral.get(f"mfcc_{i}", 0.0) for i in range(self.NUM_MFCC)], dtype=np.float32)

        # Prosodic features
        prosodic_vec = np.array([
            prosodic.get("pitch_mean", 0.0),
            prosodic.get("pitch_std", 0.0),
            prosodic.get("energy_mean", 0.0),
            prosodic.get("energy_std", 0.0),
            prosodic.get("speaking_rate", 0.0),
        ], dtype=np.float32)

        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        prosodic_vec = (prosodic_vec - np.mean(prosodic_vec)) / (np.std(prosodic_vec) + 1e-8)

        return np.concatenate([mfcc, prosodic_vec])

    def _classify_emotion(self, features: np.ndarray) -> Dict[str, float]:
        """Classify emotion from feature vector.

        Args:
            features: Feature vector.

        Returns:
            Dict mapping emotion -> probability.
        """
        self._ensure_eval_mode()

        with torch.no_grad():
            feat_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self._device)
            logits = self._classifier(feat_tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        emotion_probs = {
            emotion: float(probs[i])
            for i, emotion in enumerate(self.EMOTIONS)
        }
        return emotion_probs

    def _compute_arousal(self, emotion_probs: Dict[str, float]) -> float:
        """Compute arousal dimension from emotion probabilities.

        Arousal values mapped per emotion:
        - high: happy(0.8), angry(0.7), fearful(0.6), surprised(0.7)
        - low: sad(-0.6), neutral(0.0), disgusted(-0.2)

        Args:
            emotion_probs: Emotion probability dict.

        Returns:
            Arousal value in [-1, 1].
        """
        arousal_map = {
            "neutral": 0.0, "happy": 0.8, "sad": -0.6, "angry": 0.7,
            "fearful": 0.6, "surprised": 0.7, "disgusted": -0.2,
        }
        return float(np.sum([prob * arousal_map.get(emotion, 0.0) for emotion, prob in emotion_probs.items()]))

    def _compute_valence(self, emotion_probs: Dict[str, float]) -> float:
        """Compute valence dimension from emotion probabilities.

        Valence values mapped per emotion:
        - positive: happy(0.8), surprised(0.3)
        - negative: sad(-0.7), angry(-0.6), fearful(-0.5), disgusted(-0.6)
        - neutral: neutral(0.0)

        Args:
            emotion_probs: Emotion probability dict.

        Returns:
            Valence value in [-1, 1].
        """
        valence_map = {
            "neutral": 0.0, "happy": 0.8, "sad": -0.7, "angry": -0.6,
            "fearful": -0.5, "surprised": 0.3, "disgusted": -0.6,
        }
        return float(np.sum([prob * valence_map.get(emotion, 0.0) for emotion, prob in emotion_probs.items()]))

    def _compute_dominance(self, emotion_probs: Dict[str, float]) -> float:
        """Compute dominance dimension from emotion probabilities.

        Args:
            emotion_probs: Emotion probability dict.

        Returns:
            Dominance value in [-1, 1].
        """
        dominance_map = {
            "neutral": 0.0, "happy": 0.4, "sad": -0.5, "angry": 0.7,
            "fearful": -0.6, "surprised": -0.1, "disgusted": -0.2,
        }
        return float(np.sum([prob * dominance_map.get(emotion, 0.0) for emotion, prob in emotion_probs.items()]))

    def _detect_stress(self, features: np.ndarray, audio: np.ndarray) -> float:
        """Detect stress level from vocal features.

        High stress indicators: high pitch, fast speaking rate, high energy,
        high spectral centroid.

        Args:
            features: Feature vector.
            audio: Audio signal.

        Returns:
            Stress level in [0, 1].
        """
        try:
            import librosa

            f0, _, _ = librosa.pyin(audio, fmin=80, fmax=600, sr=self._sample_rate)
            f0 = f0[~np.isnan(f0)]

            if len(f0) > 0:
                mean_pitch = float(np.mean(f0))
                # Higher pitch relative to normal range (100-250Hz) indicates stress
                pitch_stress = min(1.0, max(0.0, (mean_pitch - 150) / 200))
            else:
                pitch_stress = 0.0

            rms = librosa.feature.rms(y=audio)[0]
            energy_stress = float(np.mean(rms)) * 2

            onset_env = librosa.onset.onset_strength(y=audio, sr=self._sample_rate)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self._sample_rate)
            rate = len(onsets) / (len(audio) / self._sample_rate)
            rate_stress = min(1.0, rate / 8.0)

        except ImportError:
            energy = float(np.mean(np.abs(audio)))
            pitch_stress = 0.3
            energy_stress = energy * 2
            rate_stress = 0.3

        stress = 0.4 * pitch_stress + 0.3 * energy_stress + 0.3 * rate_stress
        return float(np.clip(stress, 0.0, 1.0))

    def _detect_confusion(self, features: np.ndarray, emotion_probs: Dict[str, float]) -> float:
        """Detect confusion level.

        Confusion indicators: high uncertainty (flat emotion probs),
        unusual pitch patterns, hesitations.

        Args:
            features: Feature vector.
            emotion_probs: Emotion probabilities.

        Returns:
            Confusion level in [0, 1].
        """
        # Uncertainty based on flatness of emotion distribution
        probs_array = np.array(list(emotion_probs.values()))
        max_prob = np.max(probs_array)
        uncertainty = 1.0 - (max_prob - 1.0 / self.NUM_EMOTIONS) / (1.0 - 1.0 / self.NUM_EMOTIONS)
        uncertainty = np.clip(uncertainty, 0.0, 1.0)

        # Pitch variability
        pitch_std = features[-4] if len(features) >= 4 else 0.0
        pitch_confusion = min(1.0, pitch_std * 5)

        confusion = 0.6 * uncertainty + 0.4 * float(pitch_confusion)
        return float(np.clip(confusion, 0.0, 1.0))

    def _build_feature_dict(self, features: np.ndarray) -> Dict[str, float]:
        """Build a human-readable feature dictionary.

        Args:
            features: Full feature vector.

        Returns:
            Dict of feature names to values.
        """
        return {
            "mfcc_mean": float(np.mean(features[: self.NUM_MFCC])),
            "mfcc_std": float(np.std(features[: self.NUM_MFCC])),
            "pitch_feature": float(features[self.NUM_MFCC]),
            "pitch_variability": float(features[self.NUM_MFCC + 1]),
            "energy_feature": float(features[self.NUM_MFCC + 2]),
            "energy_variability": float(features[self.NUM_MFCC + 3]),
            "speaking_rate_feature": float(features[self.NUM_MFCC + 4]),
        }

    def _ensure_eval_mode(self) -> None:
        """Ensure model is in evaluation mode."""
        if not self._eval_mode:
            self._classifier.eval()
            self._eval_mode = True

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import librosa

            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
        except ImportError:
            ratio = target_sr / orig_sr
            new_len = int(len(audio) * ratio)
            return np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
