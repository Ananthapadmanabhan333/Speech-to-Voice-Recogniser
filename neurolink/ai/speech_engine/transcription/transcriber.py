from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
import whisper

logger = structlog.get_logger(__name__)


class TranscriptionError(Exception):
    """Raised when transcription fails."""


@dataclass
class WordTimestamp:
    """Word-level timestamp information."""

    word: str
    start: float  # seconds
    end: float  # seconds
    confidence: float


@dataclass
class TranscriptionSegment:
    """A segment of transcribed speech."""

    text: str
    start: float
    end: float
    confidence: float
    words: List[WordTimestamp] = field(default_factory=list)
    speaker: Optional[str] = None


@dataclass
class TranscriptionResult:
    """Full transcription result."""

    text: str
    segments: List[TranscriptionSegment]
    language: str
    confidence: float
    duration: float
    processing_time: float
    word_timestamps: List[WordTimestamp] = field(default_factory=list)


class NoiseReduction:
    """Simple noise reduction preprocessing for audio."""

    @staticmethod
    def reduce_noise(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Apply basic spectral gating noise reduction.

        Args:
            audio: Audio signal as numpy array.
            sample_rate: Audio sample rate.

        Returns:
            Noise-reduced audio.
        """
        try:
            import librosa

            # Spectral gating using librosa
            D = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude, phase = np.abs(D), np.angle(D)

            # Estimate noise profile from first 500ms
            noise_frames = max(1, int(0.5 * sample_rate / 512))
            noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

            # Spectral subtraction
            magnitude_denoised = np.maximum(
                magnitude - 1.5 * noise_profile, 0.0
            )

            # Reconstruct
            D_denoised = magnitude_denoised * np.exp(1j * phase)
            denoised = librosa.istft(D_denoised, hop_length=512)

            # Normalize
            peak = np.max(np.abs(denoised))
            if peak > 0:
                denoised = denoised / peak

            return denoised.astype(np.float32)

        except ImportError:
            logger.warning("librosa not available, skipping noise reduction")
            return audio
        except Exception as e:
            logger.error("noise_reduction_failed", error=str(e))
            return audio


class SpeechTranscriber:
    """Whisper-based speech transcription with streaming support.

    Handles real-time transcription with language detection, noise reduction,
    and optional speaker diarization support.

    Features:
    - Real-time and batch transcription
    - Automatic language detection
    - Word-level timestamps
    - Noise reduction preprocessing
    - Speaker diarization interface
    """

    SUPPORTED_LANGUAGES: List[str] = [
        "en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja",
        "ko", "ar", "hi", "tr", "pl", "sv", "da", "fi", "el", "th",
    ]

    def __init__(
        self,
        model_name: str = "large-v3",
        device: Optional[str] = None,
        compute_type: str = "float16",
        language: Optional[str] = None,
        enable_noise_reduction: bool = True,
        enable_vad: bool = True,
        beam_size: int = 5,
        word_timestamps: bool = True,
        model_path: Optional[Path] = None,
    ):
        """Initialize the speech transcriber.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large, large-v3).
            device: Device to run on ('cpu', 'cuda').
            compute_type: Computation type for model.
            language: Language code to force (None for auto-detect).
            enable_noise_reduction: Apply noise reduction preprocessing.
            enable_vad: Enable voice activity detection.
            beam_size: Beam search size.
            word_timestamps: Enable word-level timestamps.
            model_path: Custom model path.

        Raises:
            ValueError: If model_name is invalid.
        """
        valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if model_name not in valid_models:
            raise ValueError(f"Invalid model name: {model_name}. Choose from {valid_models}")

        self._model_name = model_name
        self._language = language
        self._enable_noise_reduction = enable_noise_reduction
        self._enable_vad = enable_vad
        self._beam_size = beam_size
        self._word_timestamps = word_timestamps

        if device is None:
            self._device = "cuda" if whisper.torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        logger.info(
            "loading_whisper_model",
            model=model_name,
            device=self._device,
        )

        try:
            if model_path and model_path.exists():
                self._model = whisper.load_model(model_path, device=self._device)
            else:
                self._model = whisper.load_model(model_name, device=self._device)
        except Exception as e:
            logger.error("failed_to_load_whisper_model", error=str(e))
            raise TranscriptionError(f"Failed to load Whisper model: {e}") from e

        # VAD state
        self._noise_reducer = NoiseReduction()
        self._stream_buffer: List[np.ndarray] = []
        self._stream_buffer_duration: float = 0.0
        self._max_buffer_duration: float = 30.0  # max seconds before forced transcription

        logger.info(
            "transcriber_initialized",
            model=model_name,
            device=self._device,
            language=language or "auto",
        )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio signal as numpy array (mono, float32).
            sample_rate: Sample rate of the audio.
            language: Language code (overrides default).
            **kwargs: Additional arguments for whisper.transcribe.

        Returns:
            TranscriptionResult with text, segments, and timestamps.

        Raises:
            TranscriptionError: If transcription fails.
            ValueError: If audio is invalid.
        """
        if audio is None or len(audio) == 0:
            raise ValueError("Empty audio input")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono

        # Resample if needed
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)

        # Apply noise reduction
        if self._enable_noise_reduction:
            audio = self._noise_reducer.reduce_noise(audio, 16000)

        # Normalize audio
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        start_time = time.time()

        try:
            transcribe_opts = {
                "language": language or self._language,
                "beam_size": self._beam_size,
                "word_timestamps": self._word_timestamps,
                "condition_on_previous_text": True,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            }
            transcribe_opts.update(kwargs)

            result = self._model.transcribe(audio, **transcribe_opts)

        except Exception as e:
            logger.error("transcription_failed", error=str(e))
            raise TranscriptionError(f"Transcription failed: {e}") from e

        processing_time = time.time() - start_time
        duration = len(audio) / 16000

        # Parse segments
        segments = []
        all_word_timestamps: List[WordTimestamp] = []

        for seg in result.get("segments", []):
            words = []
            for word_info in seg.get("words", []):
                wt = WordTimestamp(
                    word=word_info.get("word", "").strip(),
                    start=word_info.get("start", 0.0),
                    end=word_info.get("end", 0.0),
                    confidence=word_info.get("probability", word_info.get("confidence", 1.0)),
                )
                words.append(wt)
                all_word_timestamps.append(wt)

            segment = TranscriptionSegment(
                text=seg.get("text", "").strip(),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                confidence=seg.get("confidence", seg.get("probability", 1.0)),
                words=words,
            )
            segments.append(segment)

        full_text = result.get("text", "").strip()
        detected_language = result.get("language", "en")

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=detected_language,
            confidence=result.get("confidence", 1.0) if segments else 0.0,
            duration=duration,
            processing_time=processing_time,
            word_timestamps=all_word_timestamps,
        )

    def transcribe_streaming(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        **kwargs: Any,
    ) -> Optional[TranscriptionResult]:
        """Process streaming audio chunk and return transcription when ready.

        Accumulates audio chunks until a pause is detected, then transcribes.

        Args:
            audio_chunk: Audio chunk as numpy array.
            sample_rate: Sample rate of the audio.
            **kwargs: Additional arguments.

        Returns:
            TranscriptionResult if chunk is ready, None otherwise.
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return None

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio_chunk = self._resample(audio_chunk, sample_rate, 16000)

        # Apply noise reduction
        if self._enable_noise_reduction:
            audio_chunk = self._noise_reducer.reduce_noise(audio_chunk, 16000)

        # Check VAD
        if self._enable_vad and not self._has_voice_activity(audio_chunk):
            # If silence detected and buffer exists, transcribe buffer
            if self._stream_buffer:
                return self._flush_stream_buffer(**kwargs)
            return None

        # Add to buffer
        self._stream_buffer.append(audio_chunk)
        chunk_duration = len(audio_chunk) / 16000
        self._stream_buffer_duration += chunk_duration

        # Flush if buffer exceeds max duration
        if self._stream_buffer_duration >= self._max_buffer_duration:
            return self._flush_stream_buffer(**kwargs)

        return None

    def flush_stream(self, **kwargs: Any) -> Optional[TranscriptionResult]:
        """Flush the streaming buffer and transcribe remaining audio.

        Args:
            **kwargs: Additional transcription arguments.

        Returns:
            TranscriptionResult from buffered audio.
        """
        return self._flush_stream_buffer(**kwargs)

    def detect_language(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Detect the language of the audio.

        Args:
            audio: Audio signal.
            sample_rate: Sample rate.

        Returns:
            ISO 639-1 language code.
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)

        mel = whisper.log_mel_spectrogram(audio).to(self._device)
        _, probs = self._model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)

        logger.debug("language_detected", language=detected_lang, probability=probs[detected_lang])
        return detected_lang

    def transcribe_with_diarization(
        self,
        audio: np.ndarray,
        num_speakers: int = 2,
        sample_rate: int = 16000,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Transcribe with speaker diarization support.

        Note: Full diarization requires an external speaker segmentation model.
        This provides the interface for integrating with such models.

        Args:
            audio: Audio signal.
            num_speakers: Number of speakers expected.
            sample_rate: Sample rate.
            **kwargs: Additional transcription arguments.

        Returns:
            TranscriptionResult with speaker labels on segments.
        """
        result = self.transcribe(audio, sample_rate=sample_rate, **kwargs)

        # Placeholder for diarization integration
        # In production, integrate with pyannote.audio or similar
        if result.segments and num_speakers > 1:
            # Simple heuristic: alternate speakers based on gaps
            current_speaker = 0
            for i, seg in enumerate(result.segments):
                seg.speaker = f"speaker_{current_speaker}"
                if i > 0 and (seg.start - result.segments[i - 1].end) > 0.5:
                    current_speaker = (current_speaker + 1) % num_speakers

        return result

    def reset_stream(self) -> None:
        """Reset streaming buffer."""
        self._stream_buffer.clear()
        self._stream_buffer_duration = 0.0
        logger.debug("stream_buffer_reset")

    def _flush_stream_buffer(self, **kwargs: Any) -> Optional[TranscriptionResult]:
        """Transcribe and clear the stream buffer.

        Returns:
            TranscriptionResult or None if no data.
        """
        if not self._stream_buffer:
            return None

        audio = np.concatenate(self._stream_buffer)
        self.reset_stream()

        if len(audio) < 1600:  # Less than 100ms
            return None

        return self.transcribe(audio, **kwargs)

    def _has_voice_activity(self, audio: np.ndarray) -> bool:
        """Simple energy-based VAD.

        Args:
            audio: Audio chunk.

        Returns:
            True if voice activity detected.
        """
        if len(audio) < 160:  # 10ms at 16kHz
            return False

        energy = np.sqrt(np.mean(audio**2))
        return energy > 0.01

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate.

        Args:
            audio: Input audio.
            orig_sr: Original sample rate.
            target_sr: Target sample rate.

        Returns:
            Resampled audio.
        """
        if orig_sr == target_sr:
            return audio

        try:
            import librosa

            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
        except ImportError:
            # Simple linear interpolation fallback
            ratio = target_sr / orig_sr
            new_len = int(len(audio) * ratio)
            return np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
