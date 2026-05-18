from __future__ import annotations

import base64
import io
import time
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from pydantic import BaseModel, Field

from neurolink.backend.api.middleware import get_current_user_id
from neurolink.backend.core.config import settings
from neurolink.backend.core.exceptions import SpeechProcessingError
from neurolink.backend.core.logging import get_logger
from neurolink.backend.db import get_session
from neurolink.backend.db.models import TranslationHistory

logger = get_logger("api.speech")
router = APIRouter()


# ── Schemas ────────────────────────────────────────────────────────────────

class STTRequest(BaseModel):
    audio_data: str = Field(..., description="Base64-encoded audio")
    language: str = Field(default="en", max_length=16)
    sample_rate: int = Field(default=16000, ge=8000, le=48000)


class STTResponse(BaseModel):
    text: str
    confidence: float
    language: str
    duration_seconds: float
    processing_time_ms: float


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    language: str = Field(default="en", max_length=16)
    voice: str | None = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class TTSResponse(BaseModel):
    audio_data: str
    format: str = "wav"
    sample_rate: int = 24000
    duration_seconds: float


class EmotionAnalysisResponse(BaseModel):
    emotion: str
    confidence: float
    emotions: dict[str, float]
    processing_time_ms: float


class LanguageInfo(BaseModel):
    code: str
    name: str
    stt_supported: bool
    tts_supported: bool


class LanguagesResponse(BaseModel):
    languages: list[LanguageInfo]


# ── Supported languages ────────────────────────────────────────────────────

_SUPPORTED_LANGUAGES: list[dict[str, Any]] = [
    {"code": "en", "name": "English", "stt": True, "tts": True},
    {"code": "es", "name": "Spanish", "stt": True, "tts": True},
    {"code": "fr", "name": "French", "stt": True, "tts": True},
    {"code": "de", "name": "German", "stt": True, "tts": True},
    {"code": "zh", "name": "Chinese", "stt": True, "tts": True},
    {"code": "ja", "name": "Japanese", "stt": True, "tts": True},
    {"code": "ko", "name": "Korean", "stt": True, "tts": False},
    {"code": "ar", "name": "Arabic", "stt": True, "tts": False},
    {"code": "hi", "name": "Hindi", "stt": True, "tts": True},
    {"code": "pt", "name": "Portuguese", "stt": True, "tts": True},
    {"code": "ru", "name": "Russian", "stt": True, "tts": True},
    {"code": "it", "name": "Italian", "stt": True, "tts": True},
    {"code": "nl", "name": "Dutch", "stt": True, "tts": False},
    {"code": "tr", "name": "Turkish", "stt": True, "tts": False},
    {"code": "pl", "name": "Polish", "stt": True, "tts": False},
]


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/stt", response_model=STTResponse)
async def speech_to_text(
    body: STTRequest,
    user_id: str = Depends(get_current_user_id),
) -> STTResponse:
    start = time.monotonic()
    try:
        from neurolink.backend.speech.stt_engine import STTEngine
        engine = STTEngine()
        audio_bytes = base64.b64decode(body.audio_data)
        result = await engine.transcribe(
            audio_bytes=audio_bytes,
            language=body.language,
            sample_rate=body.sample_rate,
        )
        elapsed = (time.monotonic() - start) * 1000
        logger.info("stt_completed", text_len=len(result["text"]), confidence=result["confidence"], elapsed_ms=round(elapsed, 2))

        return STTResponse(
            text=result["text"],
            confidence=result["confidence"],
            language=result.get("language", body.language),
            duration_seconds=result.get("duration", 0.0),
            processing_time_ms=round(elapsed, 2),
        )
    except Exception as exc:
        logger.error("stt_failed", error=str(exc))
        raise SpeechProcessingError(f"Speech-to-text failed: {exc}") from exc


@router.post("/stt/upload", response_model=STTResponse)
async def speech_to_text_upload(
    file: UploadFile = File(...),
    language: str = Form("en"),
    user_id: str = Depends(get_current_user_id),
) -> STTResponse:
    start = time.monotonic()
    try:
        audio_bytes = await file.read()
        from neurolink.backend.speech.stt_engine import STTEngine
        engine = STTEngine()
        result = await engine.transcribe(
            audio_bytes=audio_bytes,
            language=language,
            sample_rate=16000,
        )
        elapsed = (time.monotonic() - start) * 1000
        return STTResponse(
            text=result["text"],
            confidence=result["confidence"],
            language=result.get("language", language),
            duration_seconds=result.get("duration", 0.0),
            processing_time_ms=round(elapsed, 2),
        )
    except Exception as exc:
        logger.error("stt_upload_failed", error=str(exc))
        raise SpeechProcessingError(f"Speech-to-text failed: {exc}") from exc


@router.post("/tts", response_model=TTSResponse)
async def text_to_speech(
    body: TTSRequest,
    user_id: str = Depends(get_current_user_id),
) -> TTSResponse:
    start = time.monotonic()
    try:
        from neurolink.backend.speech.tts_engine import TTSEngine
        engine = TTSEngine()
        result = await engine.synthesize(
            text=body.text,
            language=body.language,
            voice=body.voice,
            speed=body.speed,
        )
        elapsed = (time.monotonic() - start) * 1000
        logger.info("tts_completed", text_len=len(body.text), duration=result.get("duration", 0), elapsed_ms=round(elapsed, 2))

        return TTSResponse(
            audio_data=result["audio_base64"],
            format=result.get("format", "wav"),
            sample_rate=result.get("sample_rate", 24000),
            duration_seconds=result.get("duration", 0.0),
        )
    except Exception as exc:
        logger.error("tts_failed", error=str(exc))
        raise SpeechProcessingError(f"Text-to-speech failed: {exc}") from exc


@router.post("/analyze-emotion", response_model=EmotionAnalysisResponse)
async def analyze_speech_emotion(
    body: STTRequest,
    user_id: str = Depends(get_current_user_id),
) -> EmotionAnalysisResponse:
    start = time.monotonic()
    try:
        from neurolink.backend.emotions.speech_emotion import SpeechEmotionAnalyzer
        analyzer = SpeechEmotionAnalyzer()
        audio_bytes = base64.b64decode(body.audio_data)
        result = await analyzer.analyze(audio_bytes=audio_bytes, sample_rate=body.sample_rate)
        elapsed = (time.monotonic() - start) * 1000
        logger.info("speech_emotion_analyzed", emotion=result["emotion"], confidence=result["confidence"], elapsed_ms=round(elapsed, 2))

        return EmotionAnalysisResponse(
            emotion=result["emotion"],
            confidence=result["confidence"],
            emotions=result.get("emotions", {}),
            processing_time_ms=round(elapsed, 2),
        )
    except Exception as exc:
        logger.error("speech_emotion_analysis_failed", error=str(exc))
        raise SpeechProcessingError(f"Emotion analysis failed: {exc}") from exc


@router.get("/languages", response_model=LanguagesResponse)
async def get_supported_languages() -> LanguagesResponse:
    return LanguagesResponse(
        languages=[
            LanguageInfo(
                code=lang["code"],
                name=lang["name"],
                stt_supported=lang["stt"],
                tts_supported=lang["tts"],
            )
            for lang in _SUPPORTED_LANGUAGES
        ]
    )
