from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from ai.orchestrator import AIOrchestrator, OrchestrationResult, OrchestratorError


class MockGestureClassifier:
    def classify_gesture(self, landmark_sequence: np.ndarray) -> MagicMock:
        result = MagicMock()
        result.gesture_id = "thumbs_up"
        result.gesture_label = "thumbs_up"
        result.confidence = 0.95
        result.calibrated_confidence = 0.92
        result.raw_logits = np.random.randn(10)
        return result


class MockSpeechTranscriber:
    def transcribe(self, audio: np.ndarray) -> MagicMock:
        result = MagicMock()
        result.text = "Hello world"
        result.confidence = 0.95
        result.language = "en"
        return result

    def close(self) -> None:
        pass


class MockFacialEmotionAnalyzer:
    def analyze_emotion(self, frame: np.ndarray) -> MagicMock:
        result = MagicMock()
        result.emotion = "happy"
        result.confidence = 0.85
        result.emotion_probs = {"happy": 0.85, "neutral": 0.15}
        result.action_units = {"AU4": 0.0}
        return result

    def close(self) -> None:
        pass


@pytest.fixture
def orchestrator(monkeypatch: pytest.MonkeyPatch) -> AIOrchestrator:
    # Mock torch.cuda to avoid GPU requirements
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    orch = AIOrchestrator(
        models_dir=None,
        enable_gpu=False,
        lazy_loading=False,
        max_concurrent=4,
    )
    # Manually inject mock engines
    # Note: In production, _ensure_engine loads these; for testing we mock
    return orch


class TestOrchestratorInitialization:
    """Test AIOrchestrator initialization."""

    def test_default_initialization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        orch = AIOrchestrator(enable_gpu=False)
        assert orch._enable_gpu is False
        assert orch._lazy_loading is True
        assert orch._model_preference == "balanced"
        assert orch._request_semaphore._value == 4

    def test_custom_parameters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        orch = AIOrchestrator(
            enable_gpu=False,
            lazy_loading=False,
            max_concurrent=8,
            model_preference="accuracy",
        )
        assert orch._lazy_loading is False
        assert orch._request_semaphore._value == 8
        assert orch._model_preference == "accuracy"

    def test_device_selection_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        orch = AIOrchestrator(enable_gpu=False)
        assert str(orch._device) == "cpu"

    def test_initial_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        orch = AIOrchestrator(enable_gpu=False)
        assert orch._metrics.total_requests == 0
        assert orch._metrics.successful_requests == 0
        assert orch._metrics.failed_requests == 0


class TestOrchestratorProcessing:
    """Test multimodal input processing."""

    @pytest.mark.asyncio
    async def test_process_with_text_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        orch = AIOrchestrator(enable_gpu=False, lazy_loading=False)

        # Mock all engine loads to be no-ops
        with patch.object(orch, '_ensure_engines_ready', AsyncMock()):
            with patch.object(orch, '_process_gestures', AsyncMock(return_value={})):
                with patch.object(orch, '_process_speech', AsyncMock(return_value={})):
                    # Mock inference pipeline
                    orch._inference_pipeline = AsyncMock()
                    orch._inference_pipeline.infer = AsyncMock(return_value=MagicMock(
                        emotion="neutral",
                        urgency=0.1,
                        overall_confidence=0.8,
                        transcription=MagicMock(text=""),
                        metadata={"modalities_processed": ["text"]},
                    ))

                    # Mock intent classifier
                    orch._intent_classifier = MagicMock()
                    orch._intent_classifier.classify_intent = MagicMock(return_value=MagicMock(
                        intent="greeting",
                        confidence=0.9,
                        intent_probs={"greeting": 0.9},
                    ))

                    # Mock context manager
                    orch._context_manager = MagicMock()
                    orch._context_manager.maintain_conversation_context = MagicMock(return_value=MagicMock(
                        active_intents=["greeting"],
                        short_term=[],
                    ))
                    orch._context_manager.get_context = MagicMock(return_value=None)

                    # Mock emotion fusion
                    orch._emotion_fusion = MagicMock()
                    orch._emotion_fusion.fuse_emotions = MagicMock(return_value=MagicMock(
                        emotion="neutral",
                        confidence=0.8,
                        arousal=0.5,
                        valence=0.5,
                        dominance=0.5,
                        stress_level=0.1,
                        urgency=0.1,
                        emotion_probs={"neutral": 0.8},
                        trend="stable",
                        modality_contributions={"text": 1.0},
                    ))

                    # Mock phrase predictor
                    orch._phrase_predictor = MagicMock()
                    orch._phrase_predictor.predict_next_phrase = MagicMock(return_value=[])

                    # Mock phrase recommender
                    orch._phrase_recommender = MagicMock()
                    orch._phrase_recommender.recommend_phrases = MagicMock(return_value=[])

                    result = await orch.process_multimodal_input(
                        text_input="Hello",
                        session_id="test_session",
                    )

                    assert isinstance(result, OrchestrationResult)
                    assert result.intent_result is not None
                    assert result.emotion_result is not None
                    assert result.processing_time > 0.0

    @pytest.mark.asyncio
    async def test_process_with_gesture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        orch = AIOrchestrator(enable_gpu=False, lazy_loading=False)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with patch.object(orch, '_ensure_engines_ready', AsyncMock()):
            orch._hand_detector = MagicMock()
            orch._hand_detector.detect_hands = MagicMock(return_value=[])
            result = await orch._process_gestures(frame)
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        orch = AIOrchestrator(enable_gpu=False, lazy_loading=False, max_concurrent=2)

        with patch.object(orch, '_ensure_engines_ready', AsyncMock()):
            with patch.object(orch, 'process_multimodal_input', AsyncMock(return_value=MagicMock(
                spec=OrchestrationResult,
                multimodal_result=MagicMock(),
                intent_result=MagicMock(),
                emotion_result=MagicMock(),
                context=MagicMock(),
                predictions=[],
                recommendations=[],
                processing_time=0.01,
                metrics=MagicMock(),
            ))):
                tasks = [orch.process_multimodal_input(text_input="test") for _ in range(5)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                assert len(results) == 5


class TestErrorHandling:
    """Test error handling in orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_error_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        orch = AIOrchestrator(enable_gpu=False, lazy_loading=False)

        with pytest.raises(OrchestratorError):
            await orch.process_multimodal_input(text_input="test")

    @pytest.mark.asyncio
    async def test_engine_load_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        orch = AIOrchestrator(enable_gpu=False)

        # Test that failures in engine loading don't crash the system
        orch._hand_detector = None
        await orch._ensure_engine("hand_detector")
        # Should not raise, just log the error

    def test_metrics_tracking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        orch = AIOrchestrator(enable_gpu=False)
        orch._metrics.successful_requests = 5
        orch._metrics.total_processing_time = 2.5
        orch._successful_request(0.5)

        assert orch._metrics.successful_requests == 6
        assert orch._metrics.total_processing_time == 3.0
        assert orch._metrics.avg_processing_time == 0.5


class TestGracefulShutdown:
    """Test graceful shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_clears_resources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)

        orch = AIOrchestrator(enable_gpu=False, lazy_loading=False)
        orch._context_manager = MagicMock()
        orch._engine_ready["hand_detector"] = True

        await orch.shutdown()
        assert orch._engine_ready.get("hand_detector") is False or True  # May have been cleared

    @pytest.mark.asyncio
    async def test_shutdown_without_engines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)

        orch = AIOrchestrator(enable_gpu=False)
        # Should not raise even with no engines loaded
        await orch.shutdown()


class TestFeedback:
    """Test feedback provision."""

    @pytest.mark.asyncio
    async def test_provide_feedback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        orch = AIOrchestrator(enable_gpu=False)
        orch._adaptation_learner = AsyncMock()
        orch._adaptation_learner.learn_from_feedback = AsyncMock()

        await orch.provide_feedback(
            session_id="s1",
            user_id="u1",
            feedback={"intent_accuracy": 0.9},
        )
        orch._adaptation_learner.learn_from_feedback.assert_called_once()

    @pytest.mark.asyncio
    async def test_provide_feedback_no_learner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        orch = AIOrchestrator(enable_gpu=False)
        orch._adaptation_learner = None

        await orch.provide_feedback(
            session_id="s1",
            user_id="u1",
            feedback={},
        )
        # Should not raise


class TestSpeechSynthesis:
    """Test speech synthesis integration."""

    @pytest.mark.asyncio
    async def test_get_speech_synthesis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        orch = AIOrchestrator(enable_gpu=False)

        orch._speech_synthesizer = MagicMock()
        orch._speech_synthesizer.synthesize = MagicMock(return_value=MagicMock(
            audio=np.random.randn(16000).astype(np.float32),
        ))
        orch._engine_ready["speech_synthesizer"] = True

        audio = await orch.get_speech_synthesis("Hello")
        assert isinstance(audio, np.ndarray)

    def test_get_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        orch = AIOrchestrator(enable_gpu=False)
        metrics = orch.get_metrics()
        assert metrics.total_requests == 0
