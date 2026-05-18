from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from ai.emotion_engine.facial.facial_emotion import EmotionResult as FacialEmotionResult
from ai.speech_engine.emotion.vocal_emotion import EmotionResult as VocalEmotionResult

logger = structlog.get_logger(__name__)


class EmotionFusionError(Exception):
    """Raised when emotion fusion fails."""


@dataclass
class FinalEmotionResult:
    """Final fused emotion result."""

    emotion: str
    confidence: float
    arousal: float
    valence: float
    dominance: float
    stress_level: float
    urgency: float
    emotion_probs: Dict[str, float]
    trend: str  # "increasing", "decreasing", "stable"
    modality_contributions: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


class EmotionFusion:
    """Multimodal emotion fusion with temporal smoothing and trend analysis.

    Fuses facial and vocal emotion results with context awareness.
    Applies weighted fusion based on modality confidence, temporal
    smoothing, and estimates stress/urgency levels.

    Features:
    - Weighted fusion based on modality confidence
    - Temporal emotion smoothing with exponential moving average
    - Emotion trend analysis (increasing/decreasing/stable)
    - Stress and urgency estimation
    - Graceful handling of missing modalities
    """

    # Weights for each emotion dimension from each modality
    MODALITY_WEIGHTS = {
        "facial": {"emotion": 0.6, "arousal": 0.4, "valence": 0.6},
        "vocal": {"emotion": 0.4, "arousal": 0.6, "valence": 0.4},
        "context": {"emotion": 0.2, "arousal": 0.1, "valence": 0.2},
    }

    # Temporal smoothing factor (EMA alpha)
    SMOOTHING_ALPHA: float = 0.3
    TREND_WINDOW_SIZE: int = 10
    STRESS_URGENCY_WEIGHT: float = 0.5

    def __init__(
        self,
        smoothing_alpha: float = 0.3,
        trend_window: int = 10,
        confidence_threshold: float = 0.3,
    ):
        """Initialize emotion fusion engine.

        Args:
            smoothing_alpha: Exponential moving average factor (0-1).
                Higher = more responsive to changes.
            trend_window: Number of past results for trend analysis.
            confidence_threshold: Minimum confidence for valid modality.
        """
        self.SMOOTHING_ALPHA = smoothing_alpha
        self.TREND_WINDOW_SIZE = trend_window
        self._confidence_threshold = confidence_threshold

        # Temporal smoothing buffer per session
        self._smoothing_buffers: Dict[str, deque] = {}  # session_id -> deque of EmotionResult

        logger.info(
            "emotion_fusion_initialized",
            smoothing_alpha=smoothing_alpha,
            trend_window=trend_window,
        )

    def fuse_emotions(
        self,
        facial_result: Optional[FacialEmotionResult] = None,
        vocal_result: Optional[VocalEmotionResult] = None,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> FinalEmotionResult:
        """Fuse facial and vocal emotion results into a unified result.

        Args:
            facial_result: Facial emotion analysis result.
            vocal_result: Vocal emotion analysis result.
            context: Contextual information (e.g., conversation history).
            session_id: Session ID for temporal smoothing continuity.

        Returns:
            FinalEmotionResult with fused emotion and analysis.

        Raises:
            EmotionFusionError: If fusion fails.
        """
        if facial_result is None and vocal_result is None:
            raise EmotionFusionError("No emotion modalities provided")

        try:
            # Modality confidence weighting
            facial_weight = self._get_modality_weight(facial_result, "facial")
            vocal_weight = self._get_modality_weight(vocal_result, "vocal")
            context_weight = self._get_context_weight(context)
            total_weight = facial_weight + vocal_weight + context_weight

            if total_weight < 1e-6:
                total_weight = 1.0

            # Fuse emotions
            fused_emotion_probs = self._fuse_emotion_probabilities(
                facial_result, vocal_result, context,
                facial_weight, vocal_weight, context_weight,
            )

            # Fuse dimensions
            arousal = self._fuse_dimension(
                "arousal",
                facial_result, vocal_result, context,
                facial_weight, vocal_weight, context_weight,
                total_weight,
            )
            valence = self._fuse_dimension(
                "valence",
                facial_result, vocal_result, context,
                facial_weight, vocal_weight, context_weight,
                total_weight,
            )
            dominance = self._fuse_dimension(
                "dominance",
                facial_result, vocal_result, context,
                facial_weight, vocal_weight, context_weight,
                total_weight,
            )

            # Stress and urgency
            stress = self._estimate_stress(facial_result, vocal_result, context)
            urgency = self._estimate_urgency(facial_result, vocal_result, context, stress)

            # Get top emotion
            emotion = max(fused_emotion_probs, key=fused_emotion_probs.get)
            confidence = fused_emotion_probs[emotion]

            # Apply temporal smoothing
            if session_id:
                smoothed = self._apply_temporal_smoothing(
                    emotion, confidence, fused_emotion_probs,
                    arousal, valence, session_id,
                )
                emotion = smoothed["emotion"]
                confidence = smoothed["confidence"]
                fused_emotion_probs = smoothed["emotion_probs"]
                arousal = smoothed["arousal"]
                valence = smoothed["valence"]

            # Analyze trend
            trend = self._analyze_trend(session_id)

            # Modality contributions
            contributions = {
                "facial": facial_weight / total_weight if total_weight > 0 else 0.0,
                "vocal": vocal_weight / total_weight if total_weight > 0 else 0.0,
                "context": context_weight / total_weight if total_weight > 0 else 0.0,
            }

            return FinalEmotionResult(
                emotion=emotion,
                confidence=confidence,
                arousal=arousal,
                valence=valence,
                dominance=dominance,
                stress_level=stress,
                urgency=urgency,
                emotion_probs=fused_emotion_probs,
                trend=trend,
                modality_contributions=contributions,
            )

        except Exception as e:
            logger.error("emotion_fusion_failed", error=str(e))
            raise EmotionFusionError(f"Emotion fusion failed: {e}") from e

    def _get_modality_weight(
        self, result: Optional[Any], modality: str
    ) -> float:
        """Get confidence-weighted modality contribution.

        Args:
            result: Modality result (None if unavailable).
            modality: Modality name.

        Returns:
            Weight based on confidence and predefined modality weight.
        """
        if result is None:
            return 0.0
        if hasattr(result, "confidence") and result.confidence >= self._confidence_threshold:
            base_weight = self.MODALITY_WEIGHTS.get(modality, {}).get("emotion", 0.5)
            return base_weight * result.confidence
        return 0.0

    def _get_context_weight(self, context: Optional[Dict[str, Any]]) -> float:
        """Get context modality weight.

        Args:
            context: Context dict.

        Returns:
            Context weight.
        """
        if context and context.get("emotion_context"):
            return self.MODALITY_WEIGHTS["context"]["emotion"]
        return 0.0

    def _fuse_emotion_probabilities(
        self,
        facial: Optional[FacialEmotionResult],
        vocal: Optional[VocalEmotionResult],
        context: Optional[Dict[str, Any]],
        facial_weight: float,
        vocal_weight: float,
        context_weight: float,
    ) -> Dict[str, float]:
        """Fuse emotion probability distributions.

        Args:
            facial: Facial emotion result.
            vocal: Vocal emotion result.
            context: Context data.
            facial_weight: Facial modality weight.
            vocal_weight: Vocal modality weight.
            context_weight: Context weight.

        Returns:
            Fused emotion probability dict.
        """
        # Collect all emotion labels
        all_emotions: set = set()
        if facial and facial.emotion_probs:
            all_emotions.update(facial.emotion_probs.keys())
        if vocal and vocal.emotion_probs:
            all_emotions.update(vocal.emotion_probs.keys())

        fused: Dict[str, float] = {}
        total_weight = facial_weight + vocal_weight + context_weight

        for emotion in all_emotions:
            prob = 0.0

            if facial and facial.emotion_probs and facial_weight > 0:
                prob += facial.emotion_probs.get(emotion, 0.0) * facial_weight

            if vocal and vocal.emotion_probs and vocal_weight > 0:
                prob += vocal.emotion_probs.get(emotion, 0.0) * vocal_weight

            if context and context_weight > 0:
                context_prob = context.get("emotion_probs", {}).get(emotion, 0.0)
                prob += context_prob * context_weight

            fused[emotion] = prob / total_weight if total_weight > 0 else 0.0

        return fused

    def _fuse_dimension(
        self,
        dimension: str,
        facial: Optional[FacialEmotionResult],
        vocal: Optional[VocalEmotionResult],
        context: Optional[Dict[str, Any]],
        facial_weight: float,
        vocal_weight: float,
        context_weight: float,
        total_weight: float,
    ) -> float:
        """Fuse a single emotion dimension (arousal/valence/dominance).

        Args:
            dimension: Dimension name.
            facial: Facial result.
            vocal: Vocal result.
            context: Context data.
            facial_weight: Facial weight.
            vocal_weight: Vocal weight.
            context_weight: Context weight.
            total_weight: Sum of all weights.

        Returns:
            Fused dimension value.
        """
        value = 0.0

        if facial and hasattr(facial, dimension):
            value += getattr(facial, dimension, 0.0) * facial_weight

        if vocal and hasattr(vocal, dimension):
            value += getattr(vocal, dimension, 0.0) * vocal_weight

        if context and dimension in context:
            value += context[dimension] * context_weight

        return value / total_weight if total_weight > 0 else 0.0

    def _estimate_stress(
        self,
        facial: Optional[FacialEmotionResult],
        vocal: Optional[VocalEmotionResult],
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Estimate overall stress level.

        Args:
            facial: Facial emotion result.
            vocal: Vocal emotion result.
            context: Context data.

        Returns:
            Stress level in [0, 1].
        """
        stress = 0.0
        count = 0

        if vocal and hasattr(vocal, "stress_level"):
            stress += vocal.stress_level
            count += 1

        if facial:
            # High arousal + negative valence = stress
            if hasattr(facial, "head_pose"):
                # Furrowed brows (AU4) as stress indicator
                stress += facial.action_units.get("AU4", 0.0) * 0.5
                count += 1

        if context and "environment_stress" in context:
            stress += context["environment_stress"]
            count += 1

        return stress / count if count > 0 else 0.0

    def _estimate_urgency(
        self,
        facial: Optional[FacialEmotionResult],
        vocal: Optional[VocalEmotionResult],
        context: Optional[Dict[str, Any]],
        stress: float,
    ) -> float:
        """Estimate overall urgency level.

        Combines stress, emotional intensity, and context.

        Args:
            facial: Facial emotion result.
            vocal: Vocal emotion result.
            context: Context data.
            stress: Computed stress level.

        Returns:
            Urgency level in [0, 1].
        """
        urgency = stress * self.STRESS_URGENCY_WEIGHT

        # Pain/emergency emotions boost urgency
        if facial and facial.emotion in ("pain", "angry", "fearful"):
            urgency = max(urgency, facial.confidence * 0.8)

        if vocal and vocal.emotion in ("angry", "fearful", "sad"):
            urgency = max(urgency, vocal.confidence * 0.7)

        # Context override
        if context and context.get("emergency", False):
            urgency = max(urgency, 0.9)

        return float(np.clip(urgency, 0.0, 1.0))

    def _apply_temporal_smoothing(
        self,
        emotion: str,
        confidence: float,
        emotion_probs: Dict[str, float],
        arousal: float,
        valence: float,
        session_id: str,
    ) -> Dict[str, Any]:
        """Apply exponential moving average smoothing over time.

        Args:
            emotion: Current emotion.
            confidence: Current confidence.
            emotion_probs: Current probability distribution.
            arousal: Current arousal.
            valence: Current valence.
            session_id: Session for buffer lookup.

        Returns:
            Smoothed values dict.
        """
        if session_id not in self._smoothing_buffers:
            self._smoothing_buffers[session_id] = deque(maxlen=self.TREND_WINDOW_SIZE)

        buffer = self._smoothing_buffers[session_id]

        current = {
            "emotion": emotion,
            "confidence": confidence,
            "emotion_probs": emotion_probs,
            "arousal": arousal,
            "valence": valence,
        }
        buffer.append(current)

        if len(buffer) < 2:
            return current

        prev = buffer[-2]
        alpha = self.SMOOTHING_ALPHA

        # Smooth emotion probabilities
        smoothed_probs = {}
        all_emotions = set(emotion_probs.keys()) | set(prev.get("emotion_probs", {}).keys())
        for e in all_emotions:
            curr_prob = emotion_probs.get(e, 0.0)
            prev_prob = prev.get("emotion_probs", {}).get(e, 0.0)
            smoothed_probs[e] = alpha * curr_prob + (1 - alpha) * prev_prob

        # Smooth arousal and valence
        smoothed_arousal = alpha * arousal + (1 - alpha) * prev.get("arousal", 0.0)
        smoothed_valence = alpha * valence + (1 - alpha) * prev.get("valence", 0.0)

        smoothed_emotion = max(smoothed_probs, key=smoothed_probs.get)
        smoothed_confidence = smoothed_probs[smoothed_emotion]

        return {
            "emotion": smoothed_emotion,
            "confidence": smoothed_confidence,
            "emotion_probs": smoothed_probs,
            "arousal": smoothed_arousal,
            "valence": smoothed_valence,
        }

    def _analyze_trend(self, session_id: Optional[str]) -> str:
        """Analyze emotion trend from smoothing buffer.

        Args:
            session_id: Session for buffer lookup.

        Returns:
            "increasing", "decreasing", or "stable".
        """
        if not session_id or session_id not in self._smoothing_buffers:
            return "stable"

        buffer = self._smoothing_buffers[session_id]
        if len(buffer) < 3:
            return "stable"

        # Analyze arousal trend
        arousal_values = [b.get("arousal", 0.0) for b in buffer]
        if len(arousal_values) >= 3:
            first_half = np.mean(arousal_values[: len(arousal_values) // 2])
            second_half = np.mean(arousal_values[len(arousal_values) // 2:])
            diff = second_half - first_half

            if diff > 0.15:
                return "increasing"
            elif diff < -0.15:
                return "decreasing"

        return "stable"

    def clear_session(self, session_id: str) -> None:
        """Clear smoothing buffer for a session.

        Args:
            session_id: Session identifier.
        """
        self._smoothing_buffers.pop(session_id, None)
        logger.debug("session_emotion_buffer_cleared", session_id=session_id)
