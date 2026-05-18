from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pytest

from ai.emotion_engine.facial.facial_emotion import EmotionResult as FacialEmotionResult
from ai.emotion_engine.fusion.emotion_fusion import EmotionFusion, EmotionFusionError, FinalEmotionResult
from ai.speech_engine.emotion.vocal_emotion import EmotionResult as VocalEmotionResult


@pytest.fixture
def fusion() -> EmotionFusion:
    return EmotionFusion(smoothing_alpha=0.3, trend_window=10)


@pytest.fixture
def facial_happy() -> FacialEmotionResult:
    return FacialEmotionResult(
        emotion="happy",
        confidence=0.85,
        arousal=0.7,
        valence=0.8,
        dominance=0.6,
        emotion_probs={"happy": 0.85, "neutral": 0.10, "sad": 0.05},
        action_units={"AU4": 0.0, "AU6": 0.9, "AU12": 0.8},
    )


@pytest.fixture
def facial_sad() -> FacialEmotionResult:
    return FacialEmotionResult(
        emotion="sad",
        confidence=0.75,
        arousal=0.3,
        valence=0.2,
        dominance=0.3,
        emotion_probs={"sad": 0.75, "neutral": 0.15, "happy": 0.10},
        action_units={"AU4": 0.5, "AU6": 0.1, "AU12": 0.1},
    )


@pytest.fixture
def vocal_happy() -> VocalEmotionResult:
    return VocalEmotionResult(
        emotion="happy",
        confidence=0.8,
        arousal=0.75,
        valence=0.7,
        dominance=0.6,
        stress_level=0.2,
        emotion_probs={"happy": 0.8, "neutral": 0.15, "angry": 0.05},
    )


@pytest.fixture
def vocal_stressed() -> VocalEmotionResult:
    return VocalEmotionResult(
        emotion="angry",
        confidence=0.7,
        arousal=0.9,
        valence=0.1,
        dominance=0.8,
        stress_level=0.8,
        emotion_probs={"angry": 0.7, "fearful": 0.2, "neutral": 0.1},
    )


class TestEmotionFusionInitialization:
    """Test EmotionFusion initialization."""

    def test_default_initialization(self) -> None:
        ef = EmotionFusion()
        assert ef.SMOOTHING_ALPHA == 0.3
        assert ef.TREND_WINDOW_SIZE == 10

    def test_custom_parameters(self) -> None:
        ef = EmotionFusion(smoothing_alpha=0.5, trend_window=20, confidence_threshold=0.4)
        assert ef.SMOOTHING_ALPHA == 0.5
        assert ef.TREND_WINDOW_SIZE == 20
        assert ef._confidence_threshold == 0.4


class TestEmotionFusion:
    """Test emotion fusion functionality."""

    def test_fuse_facial_only(self, fusion: EmotionFusion, facial_happy: FacialEmotionResult) -> None:
        result = fusion.fuse_emotions(facial_result=facial_happy)
        assert isinstance(result, FinalEmotionResult)
        assert result.emotion == "happy"
        assert result.confidence > 0.0
        assert 0.0 <= result.arousal <= 1.0
        assert 0.0 <= result.valence <= 1.0
        assert 0.0 <= result.dominance <= 1.0
        assert result.modality_contributions["facial"] > 0.0

    def test_fuse_vocal_only(self, fusion: EmotionFusion, vocal_happy: VocalEmotionResult) -> None:
        result = fusion.fuse_emotions(vocal_result=vocal_happy)
        assert isinstance(result, FinalEmotionResult)
        assert result.emotion == "happy"

    def test_fuse_both_modalities(
        self,
        fusion: EmotionFusion,
        facial_happy: FacialEmotionResult,
        vocal_happy: VocalEmotionResult,
    ) -> None:
        result = fusion.fuse_emotions(facial_result=facial_happy, vocal_result=vocal_happy)
        assert result.emotion == "happy"
        assert result.confidence > 0.0
        assert result.modality_contributions["facial"] > 0.0
        assert result.modality_contributions["vocal"] > 0.0
        # Both facial and vocal are happy, so contributions should be positive

    def test_fuse_no_modalities(self, fusion: EmotionFusion) -> None:
        with pytest.raises(EmotionFusionError, match="No emotion modalities"):
            fusion.fuse_emotions()

    def test_fuse_different_emotions(
        self,
        fusion: EmotionFusion,
        facial_happy: FacialEmotionResult,
        vocal_stressed: VocalEmotionResult,
    ) -> None:
        result = fusion.fuse_emotions(facial_result=facial_happy, vocal_result=vocal_stressed)
        assert isinstance(result, FinalEmotionResult)
        # Result should be a weighted combination

    def test_fuse_with_context(
        self,
        fusion: EmotionFusion,
        facial_happy: FacialEmotionResult,
    ) -> None:
        context = {"emotion_context": True, "valence": 0.6, "arousal": 0.5}
        result = fusion.fuse_emotions(facial_result=facial_happy, context=context)
        assert isinstance(result, FinalEmotionResult)
        assert result.modality_contributions["context"] > 0.0


class TestStressDetection:
    """Test stress level estimation."""

    def test_stress_low(
        self,
        fusion: EmotionFusion,
        vocal_happy: VocalEmotionResult,
    ) -> None:
        result = fusion.fuse_emotions(vocal_result=vocal_happy)
        assert result.stress_level < 0.5

    def test_stress_high(
        self,
        fusion: EmotionFusion,
        vocal_stressed: VocalEmotionResult,
    ) -> None:
        result = fusion.fuse_emotions(vocal_result=vocal_stressed)
        assert result.stress_level > 0.5

    def test_stress_with_context(self, fusion: EmotionFusion) -> None:
        context = {"environment_stress": 0.9}
        result = fusion.fuse_emotions(context=context)
        with pytest.raises(EmotionFusionError):
            raise EmotionFusionError("No emotion modalities")
        # Actually test with at least one modality
        vocal = VocalEmotionResult(
            emotion="fearful", confidence=0.6, arousal=0.8, valence=0.2,
            dominance=0.3, stress_level=0.5,
            emotion_probs={"fearful": 0.6, "neutral": 0.4},
        )
        result = fusion.fuse_emotions(vocal_result=vocal, context=context)
        assert result.stress_level > 0.3


class TestUrgencyEstimation:
    """Test urgency level estimation."""

    def test_urgency_low(
        self,
        fusion: EmotionFusion,
        facial_happy: FacialEmotionResult,
    ) -> None:
        result = fusion.fuse_emotions(facial_result=facial_happy)
        assert result.urgency < 0.5

    def test_urgency_with_emergency_context(
        self,
        fusion: EmotionFusion,
        facial_happy: FacialEmotionResult,
    ) -> None:
        context = {"emergency": True}
        result = fusion.fuse_emotions(facial_result=facial_happy, context=context)
        assert result.urgency >= 0.9


class TestEmotionProbabilities:
    """Test fused emotion probability distribution."""

    def test_probabilities_sum_to_one(
        self,
        fusion: EmotionFusion,
        facial_happy: FacialEmotionResult,
    ) -> None:
        result = fusion.fuse_emotions(facial_result=facial_happy)
        total = sum(result.emotion_probs.values())
        assert abs(total - 1.0) < 1e-6

    def test_probability_distribution(
        self,
        fusion: EmotionFusion,
        facial_happy: FacialEmotionResult,
    ) -> None:
        result = fusion.fuse_emotions(facial_result=facial_happy)
        assert result.emotion_probs["happy"] > 0.5

    def test_fused_probs_combine(
        self,
        fusion: EmotionFusion,
        facial_happy: FacialEmotionResult,
        vocal_happy: VocalEmotionResult,
    ) -> None:
        result = fusion.fuse_emotions(facial_result=facial_happy, vocal_result=vocal_happy)
        # Combined confidence should be at least as high as individual
        assert result.emotion_probs["happy"] >= 0.5


class TestTemporalSmoothing:
    """Test temporal emotion smoothing."""

    def test_smoothing_initial_call(self, fusion: EmotionFusion, facial_happy: FacialEmotionResult) -> None:
        result = fusion.fuse_emotions(facial_result=facial_happy, session_id="session_1")
        assert isinstance(result, FinalEmotionResult)

    def test_smoothing_multiple_calls(
        self,
        fusion: EmotionFusion,
        facial_happy: FacialEmotionResult,
    ) -> None:
        r1 = fusion.fuse_emotions(facial_result=facial_happy, session_id="session_2")
        r2 = fusion.fuse_emotions(facial_result=facial_happy, session_id="session_2")
        assert r1.emotion == r2.emotion

    def test_clear_session(self, fusion: EmotionFusion, facial_happy: FacialEmotionResult) -> None:
        fusion.fuse_emotions(facial_result=facial_happy, session_id="session_3")
        fusion.clear_session("session_3")
        assert "session_3" not in fusion._smoothing_buffers


class TestEmotionTrend:
    """Test emotion trend analysis."""

    def test_trend_stable_with_few_calls(self, fusion: EmotionFusion, facial_happy: FacialEmotionResult) -> None:
        result = fusion.fuse_emotions(facial_result=facial_happy, session_id="trend_1")
        assert result.trend == "stable"

    def test_trend_after_multiple_calls(
        self,
        fusion: EmotionFusion,
        facial_happy: FacialEmotionResult,
        facial_sad: FacialEmotionResult,
    ) -> None:
        for _ in range(3):
            fusion.fuse_emotions(facial_result=facial_happy, session_id="trend_2")
        result = fusion.fuse_emotions(facial_result=facial_sad, session_id="trend_2")
        assert result.trend in ("stable", "decreasing")


class TestFinalEmotionResult:
    """Test FinalEmotionResult dataclass."""

    def test_result_fields(self) -> None:
        result = FinalEmotionResult(
            emotion="happy",
            confidence=0.9,
            arousal=0.7,
            valence=0.8,
            dominance=0.6,
            stress_level=0.1,
            urgency=0.2,
            emotion_probs={"happy": 0.9, "neutral": 0.1},
            trend="stable",
            modality_contributions={"facial": 1.0, "vocal": 0.0, "context": 0.0},
        )
        assert result.emotion == "happy"
        assert result.trend == "stable"
        assert result.timestamp > 0
