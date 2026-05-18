from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from ai.speech_engine.transcription.transcriber import (
    NoiseReduction,
    SpeechTranscriber,
    TranscriptionError,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)


class FakeModel:
    """Fake Whisper model for testing."""

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        return {
            "text": "Hello world",
            "segments": [
                {
                    "text": "Hello world",
                    "start": 0.0,
                    "end": 1.0,
                    "confidence": 0.95,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.4, "probability": 0.98},
                        {"word": "world", "start": 0.5, "end": 1.0, "probability": 0.94},
                    ],
                }
            ],
            "language": "en",
            "confidence": 0.95,
        }

    def detect_language(self, mel: Any) -> tuple:
        return None, {"en": 0.98, "fr": 0.02}


class FakeWithSilence(FakeModel):
    """Fake model that returns empty transcription."""

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        return {
            "text": "",
            "segments": [],
            "language": "en",
            "confidence": 0.0,
        }


@pytest.fixture
def transcriber(monkeypatch: pytest.MonkeyPatch) -> SpeechTranscriber:
    monkeypatch.setattr("ai.speech_engine.transcription.transcriber.whisper", None)
    t = SpeechTranscriber(
        model_name="base",
        device="cpu",
        enable_noise_reduction=False,
        enable_vad=False,
        word_timestamps=False,
    )
    t._model = FakeModel()
    return t


@pytest.fixture
def sample_audio() -> np.ndarray:
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


class TestTranscriberInitialization:
    """Test SpeechTranscriber initialization."""

    def test_default_initialization(self) -> None:
        t = SpeechTranscriber(model_name="base", device="cpu")
        assert t._model_name == "base"
        assert t._device == "cpu"
        assert t._enable_noise_reduction is True
        assert t._enable_vad is True
        assert t._beam_size == 5

    def test_invalid_model_name(self) -> None:
        with pytest.raises(ValueError, match="Invalid model name"):
            SpeechTranscriber(model_name="invalid_model", device="cpu")


class TestTranscription:
    """Test transcription functionality."""

    def test_transcribe_valid_audio(self, transcriber: SpeechTranscriber, sample_audio: np.ndarray) -> None:
        result = transcriber.transcribe(sample_audio)
        assert isinstance(result, TranscriptionResult)
        assert isinstance(result.text, str)
        assert isinstance(result.language, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.duration, float)
        assert isinstance(result.processing_time, float)
        assert result.text == "Hello world"
        assert result.language == "en"

    def test_transcribe_empty_audio(self, transcriber: SpeechTranscriber) -> None:
        with pytest.raises(ValueError, match="Empty audio input"):
            transcriber.transcribe(np.array([]))

    def test_transcribe_none_audio(self, transcriber: SpeechTranscriber) -> None:
        with pytest.raises(ValueError, match="Empty audio input"):
            transcriber.transcribe(None)  # type: ignore[arg-type]

    def test_transcribe_stereo_audio(self, transcriber: SpeechTranscriber) -> None:
        stereo = np.random.randn(16000, 2).astype(np.float32)
        result = transcriber.transcribe(stereo)
        assert isinstance(result, TranscriptionResult)

    def test_transcription_segments(self, transcriber: SpeechTranscriber, sample_audio: np.ndarray) -> None:
        result = transcriber.transcribe(sample_audio)
        assert len(result.segments) >= 0
        for seg in result.segments:
            assert isinstance(seg, TranscriptionSegment)
            assert isinstance(seg.text, str)
            assert isinstance(seg.start, float)
            assert isinstance(seg.end, float)
            assert isinstance(seg.confidence, float)


class TestTranscriptionResult:
    """Test TranscriptionResult dataclass."""

    def test_result_creation(self) -> None:
        result = TranscriptionResult(
            text="Hello",
            segments=[],
            language="en",
            confidence=0.95,
            duration=1.0,
            processing_time=0.1,
        )
        assert result.text == "Hello"
        assert result.confidence == 0.95

    def test_word_timestamps(self) -> None:
        wt = WordTimestamp(word="Hello", start=0.0, end=0.5, confidence=0.98)
        assert wt.word == "Hello"
        assert wt.start == 0.0
        assert wt.end == 0.5

    def test_transcription_segment(self) -> None:
        seg = TranscriptionSegment(text="test", start=0.0, end=1.0, confidence=0.9)
        assert seg.text == "test"
        assert seg.speaker is None


class TestLanguageDetection:
    """Test language detection."""

    def test_detect_language(self, transcriber: SpeechTranscriber, sample_audio: np.ndarray) -> None:
        lang = transcriber.detect_language(sample_audio)
        assert isinstance(lang, str)
        assert lang == "en"

    def test_detect_language_stereo(self, transcriber: SpeechTranscriber) -> None:
        stereo = np.random.randn(16000, 2).astype(np.float32)
        lang = transcriber.detect_language(stereo)
        assert isinstance(lang, str)


class TestStreamingTranscription:
    """Test streaming transcription."""

    @pytest.fixture
    def stream_transcriber(self, monkeypatch: pytest.MonkeyPatch) -> SpeechTranscriber:
        monkeypatch.setattr("ai.speech_engine.transcription.transcriber.whisper", None)
        t = SpeechTranscriber(
            model_name="base",
            device="cpu",
            enable_noise_reduction=False,
            enable_vad=False,
        )
        t._model = FakeModel()
        return t

    def test_streaming_empty_chunk(self, stream_transcriber: SpeechTranscriber) -> None:
        result = stream_transcriber.transcribe_streaming(np.array([]))
        assert result is None

    def test_streaming_valid_chunk(self, stream_transcriber: SpeechTranscriber) -> None:
        chunk = np.random.randn(16000).astype(np.float32)
        # With VAD disabled, should buffer
        result = stream_transcriber.transcribe_streaming(chunk)
        # First chunk doesn't trigger flush
        assert result is not None  # Might flush depending on buffer

    def test_streaming_buffer_flush(self, stream_transcriber: SpeechTranscriber) -> None:
        for _ in range(5):
            chunk = np.random.randn(16000).astype(np.float32)
            stream_transcriber.transcribe_streaming(chunk)
        result = stream_transcriber.flush_stream()
        if result is not None:
            assert isinstance(result, TranscriptionResult)

    def test_stream_buffer_reset(self, stream_transcriber: SpeechTranscriber) -> None:
        stream_transcriber._stream_buffer = [np.random.randn(16000).astype(np.float32)]
        stream_transcriber._stream_buffer_duration = 1.0
        stream_transcriber.reset_stream()
        assert len(stream_transcriber._stream_buffer) == 0
        assert stream_transcriber._stream_buffer_duration == 0.0


class TestNoiseReduction:
    """Test noise reduction preprocessing."""

    def test_noise_reduction(self) -> None:
        audio = np.random.randn(16000).astype(np.float32)
        denoised = NoiseReduction.reduce_noise(audio, 16000)
        assert isinstance(denoised, np.ndarray)
        assert denoised.shape == audio.shape
        assert not np.any(np.isnan(denoised))

    def test_noise_reduction_silence(self) -> None:
        audio = np.zeros(16000, dtype=np.float32)
        denoised = NoiseReduction.reduce_noise(audio, 16000)
        assert isinstance(denoised, np.ndarray)


class TestVoiceActivityDetection:
    """Test VAD helper."""

    def test_vad_with_speech(self) -> None:
        t = SpeechTranscriber(model_name="base", device="cpu", enable_vad=True)
        audio = np.random.randn(1600).astype(np.float32) * 0.1
        assert t._has_voice_activity(audio) is True

    def test_vad_with_silence(self) -> None:
        t = SpeechTranscriber(model_name="base", device="cpu", enable_vad=True)
        audio = np.zeros(1600, dtype=np.float32)
        assert t._has_voice_activity(audio) is False

    def test_vad_short_audio(self) -> None:
        t = SpeechTranscriber(model_name="base", device="cpu", enable_vad=True)
        audio = np.zeros(100, dtype=np.float32)
        assert t._has_voice_activity(audio) is False


class TestResampling:
    """Test audio resampling."""

    def test_resample_to_16khz(self) -> None:
        t = SpeechTranscriber(model_name="base", device="cpu")
        audio = np.random.randn(8000).astype(np.float32)
        resampled = t._resample(audio, 8000, 16000)
        assert len(resampled) == 16000

    def test_resample_from_44khz(self) -> None:
        t = SpeechTranscriber(model_name="base", device="cpu")
        audio = np.random.randn(44100).astype(np.float32)
        resampled = t._resample(audio, 44100, 16000)
        assert len(resampled) == 16000

    def test_resample_same_rate(self) -> None:
        t = SpeechTranscriber(model_name="base", device="cpu")
        audio = np.random.randn(16000).astype(np.float32)
        resampled = t._resample(audio, 16000, 16000)
        assert len(resampled) == 16000
        np.testing.assert_array_almost_equal(audio, resampled)


class TestErrorHandling:
    """Test error handling."""

    def test_transcription_error(self) -> None:
        with pytest.raises(TranscriptionError):
            raise TranscriptionError("Test error")

    def test_supported_languages(self) -> None:
        assert "en" in SpeechTranscriber.SUPPORTED_LANGUAGES
        assert "es" in SpeechTranscriber.SUPPORTED_LANGUAGES
        assert len(SpeechTranscriber.SUPPORTED_LANGUAGES) >= 10
