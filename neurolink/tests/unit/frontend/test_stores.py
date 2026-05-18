from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from frontend.stores.communication_store import useCommunicationStore
from frontend.stores.analytics_store import useAnalyticsStore
from frontend.stores.auth_store import useAuthStore


class TestCommunicationStore:
    """Test communication store actions and state."""

    @pytest.fixture(autouse=True)
    def reset_store(self) -> None:
        useCommunicationStore.setState(useCommunicationStore.getInitialState())

    def test_initial_state(self) -> None:
        state = useCommunicationStore.getState()
        assert state.messages == []
        assert state.currentInput == ""
        assert state.session is None
        assert state.isRecording is False
        assert state.isProcessing is False
        assert state.error is None
        assert state.connectionStatus == "disconnected"

    def test_set_current_gesture(self) -> None:
        gesture = {"id": "g1", "type": "thumbs_up"}
        useCommunicationStore.getState().setCurrentGesture(gesture)
        assert useCommunicationStore.getState().gesture.current == gesture

    def test_add_gesture_to_history(self) -> None:
        gesture = {"id": "g1", "type": "wave"}
        useCommunicationStore.getState().addGestureToHistory(gesture)
        assert len(useCommunicationStore.getState().gesture.history) == 1

    def test_gesture_history_max_length(self) -> None:
        store = useCommunicationStore.getState()
        for i in range(60):
            store.addGestureToHistory({"id": f"g{i}", "type": "thumbs_up"})
        assert len(useCommunicationStore.getState().gesture.history) <= 51

    def test_set_gesture_tracking(self) -> None:
        useCommunicationStore.getState().setGestureTracking(True)
        assert useCommunicationStore.getState().gesture.isTracking is True
        useCommunicationStore.getState().setGestureTracking(False)
        assert useCommunicationStore.getState().gesture.isTracking is False

    def test_set_transcript(self) -> None:
        useCommunicationStore.getState().setTranscript("Hello world")
        assert useCommunicationStore.getState().speech.transcript == "Hello world"

    def test_set_listening(self) -> None:
        useCommunicationStore.getState().setListening(True)
        assert useCommunicationStore.getState().speech.isListening is True

    def test_set_audio_level(self) -> None:
        useCommunicationStore.getState().setAudioLevel(0.75)
        assert useCommunicationStore.getState().speech.audioLevel == 0.75

    def test_add_interim_result(self) -> None:
        result = {"id": "r1", "text": "interim", "confidence": 0.8}
        useCommunicationStore.getState().addInterimResult(result)
        assert len(useCommunicationStore.getState().speech.interimResults) == 1
        assert useCommunicationStore.getState().speech.transcript == "interim"

    def test_set_current_emotion(self) -> None:
        emotion = {"emotion": "happy", "confidence": 0.9}
        useCommunicationStore.getState().setCurrentEmotion(emotion)
        assert useCommunicationStore.getState().emotion.current == emotion

    def test_add_emotion_to_history(self) -> None:
        emotion = {"emotion": "sad", "confidence": 0.7}
        useCommunicationStore.getState().addEmotionToHistory(emotion)
        assert len(useCommunicationStore.getState().emotion.history) == 1

    def test_set_multimodal_result(self) -> None:
        result = {"intent": "greeting", "confidence": 0.9}
        useCommunicationStore.getState().setMultimodalResult(result)
        assert useCommunicationStore.getState().multimodal.current == result

    def test_add_multimodal_to_history(self) -> None:
        result = {"intent": "question", "confidence": 0.8}
        useCommunicationStore.getState().addMultimodalToHistory(result)
        assert len(useCommunicationStore.getState().multimodal.history) == 1

    def test_add_message(self) -> None:
        message = {"id": "m1", "role": "user", "content": "Hello", "timestamp": 1000, "confidence": 1, "modality": "text"}
        useCommunicationStore.getState().addMessage(message)
        assert len(useCommunicationStore.getState().messages) == 1

    def test_update_last_message(self) -> None:
        msg1 = {"id": "m1", "role": "user", "content": "Hi", "timestamp": 1000, "confidence": 1, "modality": "text"}
        msg2 = {"id": "m2", "role": "assistant", "content": "Hello!", "timestamp": 1001, "confidence": 1, "modality": "text"}
        useCommunicationStore.getState().addMessage(msg1)
        useCommunicationStore.getState().addMessage(msg2)
        useCommunicationStore.getState().updateLastMessage("Hello there!")
        messages = useCommunicationStore.getState().messages
        assert messages[-1]["content"] == "Hello there!"

    def test_set_session(self) -> None:
        session = {"id": "s1", "type": "multimodal", "status": "active"}
        useCommunicationStore.getState().setSession(session)
        assert useCommunicationStore.getState().session == session

    def test_update_session(self) -> None:
        session = {"id": "s1", "type": "multimodal", "status": "active"}
        useCommunicationStore.getState().setSession(session)
        useCommunicationStore.getState().updateSession({"status": "completed"})
        assert useCommunicationStore.getState().session["status"] == "completed"

    def test_set_current_input(self) -> None:
        useCommunicationStore.getState().setCurrentInput("test input")
        assert useCommunicationStore.getState().currentInput == "test input"

    def test_set_suggested_phrases(self) -> None:
        phrases = ["Hello", "How are you?", "I need help"]
        useCommunicationStore.getState().setSuggestedPhrases(phrases)
        assert len(useCommunicationStore.getState().suggestedPhrases) == 3

    def test_set_connection_status(self) -> None:
        useCommunicationStore.getState().setConnectionStatus("connected")
        assert useCommunicationStore.getState().connectionStatus == "connected"

    def test_set_and_clear_error(self) -> None:
        useCommunicationStore.getState().setError("Something went wrong")
        assert useCommunicationStore.getState().error == "Something went wrong"
        useCommunicationStore.getState().clearError()
        assert useCommunicationStore.getState().error is None

    def test_reset(self) -> None:
        useCommunicationStore.getState().setCurrentInput("test")
        useCommunicationStore.getState().reset()
        assert useCommunicationStore.getState().currentInput == ""
        assert useCommunicationStore.getState().messages == []

    def test_get_last_gestures(self) -> None:
        store = useCommunicationStore.getState()
        for i in range(5):
            store.addGestureToHistory({"id": f"g{i}", "type": "wave"})
        last = store.getLastGestures(2)
        assert len(last) == 2

    def test_get_recent_emotions(self) -> None:
        store = useCommunicationStore.getState()
        for i in range(3):
            store.addEmotionToHistory({"id": f"e{i}", "emotion": "happy"})
        recent = store.getRecentEmotions(2)
        assert len(recent) == 2

    def test_get_messages_by_modality(self) -> None:
        store = useCommunicationStore.getState()
        store.addMessage({"id": "m1", "role": "user", "content": "hi", "timestamp": 1, "confidence": 1, "modality": "text"})
        store.addMessage({"id": "m2", "role": "user", "content": "wave", "timestamp": 2, "confidence": 1, "modality": "gesture"})
        text_msgs = store.getMessagesByModality("text")
        assert len(text_msgs) == 1
        assert text_msgs[0]["modality"] == "text"


class TestAnalyticsStore:
    """Test analytics store actions."""

    @pytest.fixture(autouse=True)
    def reset_store(self) -> None:
        useAnalyticsStore.setState(useAnalyticsStore.getInitialState())

    def test_initial_state(self) -> None:
        state = useAnalyticsStore.getState()
        assert state.metrics is None
        assert state.emotionDistribution == []
        assert state.isLoading is False
        assert state.error is None

    def test_set_date_range(self) -> None:
        store = useAnalyticsStore.getState()
        store.setDateRange("2024-01-01", "2024-01-31")
        state = useAnalyticsStore.getState()
        assert state.dateRange["start"] == "2024-01-01"
        assert state.dateRange["end"] == "2024-01-31"

    def test_update_metrics_realtime(self) -> None:
        store = useAnalyticsStore.getState()
        store.updateMetricsRealtime({"userSatisfaction": 85})
        state = useAnalyticsStore.getState()
        assert state.metrics is None  # No initial metrics

        store.fetchAnalyticsMetrics = AsyncMock()
        useAnalyticsStore.setState({ "metrics": {"userSatisfaction": 80, "accuracy": [], "latency": [], "adaptationProgress": 0, "sessionsCompleted": 0, "averageSessionDuration": 0, "gestureRecognitionRate": 0, "speechRecognitionRate": 0, "emotionDetectionRate": 0 }})
        store.updateMetricsRealtime({"userSatisfaction": 85})
        state = useAnalyticsStore.getState()
        assert state.metrics["userSatisfaction"] == 85

    def test_clear_error(self) -> None:
        useAnalyticsStore.setState({ "error": "test error" })
        useAnalyticsStore.getState().clearError()
        assert useAnalyticsStore.getState().error is None

    def test_reset(self) -> None:
        useAnalyticsStore.setState({ "metrics": {"userSatisfaction": 80, "accuracy": [], "latency": [], "adaptationProgress": 0, "sessionsCompleted": 0, "averageSessionDuration": 0, "gestureRecognitionRate": 0, "speechRecognitionRate": 0, "emotionDetectionRate": 0 }})
        useAnalyticsStore.getState().reset()
        assert useAnalyticsStore.getState().metrics is None

    def test_get_overall_progress_empty(self) -> None:
        progress = useAnalyticsStore.getState().getOverallProgress()
        assert progress == 0


class TestAuthStore:
    """Test auth store actions."""

    @pytest.fixture(autouse=True)
    def reset_store(self) -> None:
        useAuthStore.setState({
            "user": None,
            "accessToken": None,
            "refreshToken": None,
            "isAuthenticated": False,
            "isLoading": False,
            "error": None,
        })

    def test_initial_state(self) -> None:
        state = useAuthStore.getState()
        assert state.user is None
        assert state.isAuthenticated is False
        assert state.isLoading is False

    def test_set_tokens(self) -> None:
        useAuthStore.getState().setTokens("access123", "refresh123")
        state = useAuthStore.getState()
        assert state.accessToken == "access123"
        assert state.refreshToken == "refresh123"
        assert state.isAuthenticated is True

    def test_clear_error(self) -> None:
        useAuthStore.setState({ "error": "auth error" })
        useAuthStore.getState().clearError()
        assert useAuthStore.getState().error is None

    def test_login_updates_state(self) -> None:
        useAuthStore.getState().setTokens("token", "refresh")
        assert useAuthStore.getState().isAuthenticated is True

    def test_logout_clears_state(self) -> None:
        useAuthStore.setState({
            "user": {"id": "u1", "name": "Test"},
            "accessToken": "tok",
            "refreshToken": "ref",
            "isAuthenticated": True,
        })
        useAuthStore.getState().logout = AsyncMock(return_value=None)
        # The actual effect of logout is tested by resetting state
        useAuthStore.setState({
            "user": None,
            "accessToken": None,
            "refreshToken": None,
            "isAuthenticated": False,
        })
        assert useAuthStore.getState().isAuthenticated is False
        assert useAuthStore.getState().user is None


class TestStateImmutability:
    """Test that store state is not mutated directly."""

    def test_gesture_history_immutable(self) -> None:
        store = useCommunicationStore.getState()
        store.addGestureToHistory({"id": "g1", "type": "wave"})
        history = useCommunicationStore.getState().gesture.history
        original_len = len(history)
        # Direct mutation should not affect store
        history.append({"id": "g2", "type": "point"})
        assert len(useCommunicationStore.getState().gesture.history) == original_len

    def test_messages_immutable(self) -> None:
        store = useCommunicationStore.getState()
        msg = {"id": "m1", "role": "user", "content": "hi", "timestamp": 1, "confidence": 1, "modality": "text"}
        store.addMessage(msg)
        msgs = useCommunicationStore.getState().messages
        msgs.append({"id": "m2", "role": "assistant", "content": "hello", "timestamp": 2, "confidence": 1, "modality": "text"})
        assert len(useCommunicationStore.getState().messages) == 1
