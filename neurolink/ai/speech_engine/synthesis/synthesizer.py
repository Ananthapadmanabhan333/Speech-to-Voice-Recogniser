from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import structlog
import torch

logger = structlog.get_logger(__name__)


class SynthesisError(Exception):
    """Raised when speech synthesis fails."""


class VoiceNotFoundError(SynthesisError):
    """Raised when requested voice is not found."""


@dataclass
class VoiceProfile:
    """Voice profile for speech synthesis."""

    voice_id: str
    name: str
    gender: Optional[str] = None
    language: str = "en"
    emotion_capabilities: List[str] = field(default_factory=lambda: ["neutral"])
    sample_rate: int = 24000
    speaker_embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisResult:
    """Result of speech synthesis."""

    audio: np.ndarray
    sample_rate: int
    duration: float
    voice_id: str
    text_hash: str
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SSMLProcessor:
    """Process SSML (Speech Synthesis Markup Language) input."""

    @staticmethod
    def process_ssml(ssml_text: str) -> str:
        """Extract plain text from basic SSML.

        Supports: <speak>, <voice>, <prosody>, <break>, <emphasis>, <say-as>

        Args:
            ssml_text: SSML markup string.

        Returns:
            Plain text with prosody annotations.
        """
        import re

        text = ssml_text

        # Remove XML declaration
        text = re.sub(r'<\?xml[^>]*\?>', '', text)

        # Extract prosody tags
        text = re.sub(r'<prosody[^>]*>(.*?)</prosody>', r'\1', text)
        text = re.sub(r'<emphasis[^>]*>(.*?)</emphasis>', r'\1', text)
        text = re.sub(r'<voice[^>]*>(.*?)</voice>', r'\1', text)

        # Replace break with pause marker
        text = re.sub(r'<break[^>]*/>', ' ... ', text)
        text = re.sub(r'<break[^>]*>', ' ... ', text)

        # Handle say-as
        text = re.sub(r'<say-as[^>]*>(.*?)</say-as>', r'\1', text)

        # Remove remaining tags
        text = re.sub(r'<[^>]+>', '', text)

        # Decode HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&quot;', '"').replace('&#39;', "'")

        return text.strip()

    @staticmethod
    def extract_prosody(ssml_text: str) -> Dict[str, Any]:
        """Extract prosody settings from SSML.

        Returns dict with rate, pitch, volume settings.
        """
        import re

        prosody: Dict[str, Any] = {}

        match = re.search(r'<prosody([^>]*)>', ssml_text)
        if match:
            attrs = match.group(1)
            rate_match = re.search(r'rate=["\']([^"\']*)["\']', attrs)
            if rate_match:
                prosody['rate'] = rate_match.group(1)
            pitch_match = re.search(r'pitch=["\']([^"\']*)["\']', attrs)
            if pitch_match:
                prosody['pitch'] = pitch_match.group(1)
            volume_match = re.search(r'volume=["\']([^"\']*)["\']', attrs)
            if volume_match:
                prosody['volume'] = volume_match.group(1)

        return prosody


class SpeechSynthesizer:
    """Text-to-speech synthesis with multi-voice support, emotion-aware prosody,
    streaming generation, and SSML support.

    Features:
    - Multi-voice support with voice profiles
    - Emotion-aware prosody control
    - Streaming audio generation
    - SSML markup processing
    - Voice cloning interface
    """

    # Default voice profiles
    DEFAULT_VOICES: Dict[str, VoiceProfile] = {
        "default_female": VoiceProfile(
            voice_id="default_female",
            name="Default Female",
            gender="female",
            language="en",
            emotion_capabilities=["neutral", "happy", "sad", "angry", "surprised"],
        ),
        "default_male": VoiceProfile(
            voice_id="default_male",
            name="Default Male",
            gender="male",
            language="en",
            emotion_capabilities=["neutral", "happy", "sad", "angry", "surprised"],
        ),
    }

    # Emotion to prosody parameter mapping
    EMOTION_PROSODY_MAP: Dict[str, Dict[str, float]] = {
        "neutral": {"rate": 1.0, "pitch": 1.0, "energy": 1.0},
        "happy": {"rate": 1.15, "pitch": 1.2, "energy": 1.3},
        "sad": {"rate": 0.85, "pitch": 0.8, "energy": 0.7},
        "angry": {"rate": 1.2, "pitch": 1.1, "energy": 1.5},
        "surprised": {"rate": 1.1, "pitch": 1.3, "energy": 1.2},
        "fearful": {"rate": 1.05, "pitch": 1.25, "energy": 0.9},
        "disgusted": {"rate": 0.9, "pitch": 0.85, "energy": 0.8},
        "calm": {"rate": 0.8, "pitch": 0.9, "energy": 0.6},
    }

    def __init__(
        self,
        model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
        device: Optional[str] = None,
        sample_rate: int = 24000,
        voice_dir: Optional[Path] = None,
        enable_gpu: bool = True,
    ):
        """Initialize the speech synthesizer.

        Args:
            model_name: Coqui TTS model name.
            device: Device to run on ('cpu', 'cuda').
            sample_rate: Output sample rate.
            voice_dir: Directory with voice profiles/speaker embeddings.
            enable_gpu: Enable GPU acceleration.
        """
        self._model_name = model_name
        self._sample_rate = sample_rate
        self._voice_dir = voice_dir

        if device is None:
            self._device = "cuda" if torch.cuda.is_available() and enable_gpu else "cpu"
        else:
            self._device = device

        # Voice registry
        self._voices: Dict[str, VoiceProfile] = dict(self.DEFAULT_VOICES)
        self._load_custom_voices()

        # TTS model (lazy loaded)
        self._tts = None
        self._model_loaded = False

        logger.info(
            "synthesizer_initialized",
            model=model_name,
            device=self._device,
            voices=list(self._voices.keys()),
        )

    def synthesize(
        self,
        text: str,
        voice_id: str = "default_female",
        emotion: Optional[str] = None,
        ssml: bool = False,
        **kwargs: Any,
    ) -> SynthesisResult:
        """Synthesize speech from text.

        Args:
            text: Input text or SSML markup.
            voice_id: Voice profile ID to use.
            emotion: Emotion for prosody modulation.
            ssml: If True, text is treated as SSML markup.
            **kwargs: Additional synthesis parameters.

        Returns:
            SynthesisResult with audio data.

        Raises:
            SynthesisError: If synthesis fails.
            VoiceNotFoundError: If voice_id not found.
        """
        if not text or not text.strip():
            raise ValueError("Empty text provided")

        if voice_id not in self._voices:
            raise VoiceNotFoundError(f"Voice '{voice_id}' not found. Available: {list(self._voices.keys())}")

        voice = self._voices[voice_id]

        # Process SSML
        if ssml:
            text = SSMLProcessor.process_ssml(text)

        # Apply emotion prosody
        prosody = self.EMOTION_PROSODY_MAP.get(emotion or "neutral", self.EMOTION_PROSODY_MAP["neutral"])

        start_time = time.time()
        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]

        try:
            if self._use_external_tts():
                audio = self._synthesize_with_tts(text, voice, prosody, **kwargs)
            else:
                audio = self._synthesize_fallback(text, voice, prosody, **kwargs)

        except Exception as e:
            logger.error("synthesis_failed", error=str(e), text_hash=text_hash)
            raise SynthesisError(f"Speech synthesis failed: {e}") from e

        processing_time = time.time() - start_time
        duration = len(audio) / self._sample_rate

        if duration < 0.01:
            logger.warning("very_short_synthesis", duration=duration, text=text)

        return SynthesisResult(
            audio=audio,
            sample_rate=self._sample_rate,
            duration=duration,
            voice_id=voice_id,
            text_hash=text_hash,
            processing_time=processing_time,
            metadata={
                "emotion": emotion,
                "prosody": prosody,
                "model": self._model_name if self._model_loaded else "fallback",
            },
        )

    def synthesize_streaming(
        self,
        text: str,
        voice_id: str = "default_female",
        chunk_size: int = 50,
        **kwargs: Any,
    ) -> List[SynthesisResult]:
        """Generate speech in streaming chunks.

        Splits text into smaller chunks and synthesizes each independently.

        Args:
            text: Input text.
            voice_id: Voice profile ID.
            chunk_size: Maximum characters per chunk.
            **kwargs: Additional parameters.

        Returns:
            List of SynthesisResult chunks.
        """
        chunks = self._split_text_into_chunks(text, chunk_size)
        results: List[SynthesisResult] = []

        for chunk in chunks:
            result = self.synthesize(chunk, voice_id, **kwargs)
            results.append(result)

        return results

    def register_voice(
        self,
        voice_id: str,
        name: str,
        gender: Optional[str] = None,
        language: str = "en",
        speaker_embedding: Optional[np.ndarray] = None,
        **metadata: Any,
    ) -> VoiceProfile:
        """Register a new voice profile.

        Args:
            voice_id: Unique identifier for the voice.
            name: Display name.
            gender: Voice gender.
            language: Language code.
            speaker_embedding: Speaker embedding vector for voice cloning.
            **metadata: Additional metadata.

        Returns:
            Created VoiceProfile.

        Raises:
            ValueError: If voice_id already exists.
        """
        if voice_id in self._voices:
            raise ValueError(f"Voice '{voice_id}' already exists")

        profile = VoiceProfile(
            voice_id=voice_id,
            name=name,
            gender=gender,
            language=language,
            speaker_embedding=speaker_embedding,
            metadata=metadata,
        )
        self._voices[voice_id] = profile
        logger.info("voice_registered", voice_id=voice_id, name=name)
        return profile

    def remove_voice(self, voice_id: str) -> None:
        """Remove a registered voice profile.

        Args:
            voice_id: Voice profile ID to remove.
        """
        if voice_id in self.DEFAULT_VOICES:
            raise ValueError(f"Cannot remove default voice '{voice_id}'")
        self._voices.pop(voice_id, None)
        logger.info("voice_removed", voice_id=voice_id)

    def clone_voice(
        self,
        audio_samples: List[np.ndarray],
        voice_name: str,
        sample_rate: int = 16000,
    ) -> VoiceProfile:
        """Clone a voice from audio samples.

        Computes a speaker embedding from the provided audio samples
        and registers a new voice profile.

        Args:
            audio_samples: List of audio arrays for cloning.
            voice_name: Name for the cloned voice.
            sample_rate: Sample rate of the audio samples.

        Returns:
            Created VoiceProfile.

        Raises:
            SynthesisError: If voice cloning fails.
        """
        if not audio_samples:
            raise ValueError("At least one audio sample required")

        try:
            import librosa
            import soundfile as sf

            # Load and preprocess samples
            embeddings: List[np.ndarray] = []
            for audio in audio_samples:
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sample_rate != 16000:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

                # Extract MFCC-based embedding (simple approach)
                mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
                embedding = np.mean(mfcc, axis=1)
                embeddings.append(embedding)

            # Average embeddings
            speaker_embedding = np.mean(embeddings, axis=0)
            speaker_embedding = speaker_embedding / (np.linalg.norm(speaker_embedding) + 1e-8)

        except ImportError:
            logger.warning("librosa not available, using placeholder embedding")
            speaker_embedding = np.random.randn(40).astype(np.float32)
            speaker_embedding = speaker_embedding / np.linalg.norm(speaker_embedding)

        voice_id = f"cloned_{voice_name.lower().replace(' ', '_')}_{int(time.time())}"
        return self.register_voice(
            voice_id=voice_id,
            name=voice_name,
            speaker_embedding=speaker_embedding,
            is_cloned=True,
        )

    def get_available_voices(self) -> List[VoiceProfile]:
        """Get list of all available voice profiles.

        Returns:
            List of VoiceProfile objects.
        """
        return list(self._voices.values())

    def _load_custom_voices(self) -> None:
        """Load custom voice profiles from voice directory."""
        if not self._voice_dir or not self._voice_dir.exists():
            return

        for voice_file in self._voice_dir.glob("*.json"):
            try:
                with open(voice_file) as f:
                    data = json.load(f)
                profile = VoiceProfile(**data)
                self._voices[profile.voice_id] = profile
                logger.debug("custom_voice_loaded", voice_id=profile.voice_id)
            except Exception as e:
                logger.warning("failed_to_load_voice", file=str(voice_file), error=str(e))

    def _synthesize_with_tts(
        self, text: str, voice: VoiceProfile, prosody: Dict[str, float], **kwargs: Any
    ) -> np.ndarray:
        """Synthesize using TTS library.

        Args:
            text: Input text.
            voice: Voice profile.
            prosody: Prosody parameters.
            **kwargs: Additional parameters.

        Returns:
            Audio array.
        """
        if not self._model_loaded:
            self._load_tts_model()

        if self._tts is None:
            return self._synthesize_fallback(text, voice, prosody, **kwargs)

        try:
            # Apply prosody modulation through text preprocessing
            processed_text = self._apply_prosody(text, prosody)

            # Synthesize
            wav = self._tts.tts(processed_text, speaker=voice.voice_id)

            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()

            # Resample if needed
            tts_sample_rate = getattr(self._tts, "sample_rate", 22050)
            if tts_sample_rate != self._sample_rate:
                wav = self._resample(wav, tts_sample_rate, self._sample_rate)

            return wav.astype(np.float32)

        except Exception as e:
            logger.error("tts_synthesis_failed", error=str(e))
            return self._synthesize_fallback(text, voice, prosody, **kwargs)

    def _synthesize_fallback(
        self, text: str, voice: VoiceProfile, prosody: Dict[str, float], **kwargs: Any
    ) -> np.ndarray:
        """Fallback synthesis method when TTS model is unavailable.

        Generates a simple sinusoidal approximation for testing.

        Args:
            text: Input text.
            voice: Voice profile.
            prosody: Prosody parameters.

        Returns:
            Audio array.
        """
        duration = max(0.5, len(text) * 0.08) * prosody.get("rate", 1.0)
        t = np.linspace(0, duration, int(self._sample_rate * duration), endpoint=False)

        # Base frequency with pitch modulation
        base_freq = 180.0 if voice.gender == "female" else 120.0
        pitch_factor = prosody.get("pitch", 1.0)
        freq = base_freq * pitch_factor

        # Simple formant synthesis
        audio = (
            0.5 * np.sin(2 * np.pi * freq * t)
            + 0.25 * np.sin(2 * np.pi * freq * 2 * t)
            + 0.125 * np.sin(2 * np.pi * freq * 3 * t)
        )

        # Apply energy envelope
        energy = prosody.get("energy", 1.0)
        envelope = np.exp(-3 * t / duration)
        audio = audio * envelope * energy * 0.3

        # Add silence padding
        silence = np.zeros(int(self._sample_rate * 0.05))
        audio = np.concatenate([silence, audio, silence])

        return audio.astype(np.float32)

    def _load_tts_model(self) -> None:
        """Lazy-load the TTS model."""
        try:
            from TTS.api import TTS

            self._tts = TTS(model_name=self._model_name, progress_bar=False)
            self._model_loaded = True

            # Register available voices from model
            if hasattr(self._tts, "speakers") and self._tts.speakers:
                for spk in self._tts.speakers:
                    if spk not in self._voices:
                        self._voices[spk] = VoiceProfile(
                            voice_id=spk,
                            name=spk,
                            language="en",
                        )

            logger.info("tts_model_loaded", model=self._model_name)

        except ImportError:
            logger.warning("TTS library not available, using fallback synthesis")
            self._tts = None
            self._model_loaded = False
        except Exception as e:
            logger.error("failed_to_load_tts_model", error=str(e))
            self._tts = None
            self._model_loaded = False

    def _use_external_tts(self) -> bool:
        """Check if external TTS library should be used."""
        if self._model_loaded:
            return True
        if not self._model_loaded and self._tts is None:
            return False
        return False

    def _apply_prosody(self, text: str, prosody: Dict[str, float]) -> str:
        """Apply prosody markers to text.

        Args:
            text: Input text.
            prosody: Prosody parameters.

        Returns:
            Text with SSML-like prosody markers.
        """
        rate = prosody.get("rate", 1.0)
        pitch = prosody.get("pitch", 1.0)

        if abs(rate - 1.0) > 0.05:
            rate_str = f"+{int((rate - 1.0) * 100)}%" if rate > 1.0 else f"{int((1.0 - rate) * 100)}%"
            text = f"<prosody rate='{rate_str}'>{text}</prosody>"

        if abs(pitch - 1.0) > 0.05:
            pitch_str = f"+{int((pitch - 1.0) * 100)}%" if pitch > 1.0 else f"{int((1.0 - pitch) * 100)}%"
            text = f"<prosody pitch='{pitch_str}'>{text}</prosody>"

        return text

    def _split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks at sentence boundaries.

        Args:
            text: Input text.
            chunk_size: Maximum characters per chunk.

        Returns:
            List of text chunks.
        """
        import re

        if len(text) <= chunk_size:
            return [text]

        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks: List[str] = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio array.

        Args:
            audio: Input audio.
            orig_sr: Original sample rate.
            target_sr: Target sample rate.

        Returns:
            Resampled audio.
        """
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
