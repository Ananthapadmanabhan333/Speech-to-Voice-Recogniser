from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from neurolink.backend.api.middleware import get_current_user_id
from neurolink.backend.core.exceptions import MultimodalFusionError, NotFoundError
from neurolink.backend.core.logging import get_logger
from neurolink.backend.db import get_session
from neurolink.backend.db.models import CommunicationSession, TranslationHistory

logger = get_logger("api.communication")
router = APIRouter()


# ── Schemas ────────────────────────────────────────────────────────────────

class SessionCreateRequest(BaseModel):
    session_type: str = Field(..., pattern=r"^(gesture|speech|multimodal|text)$")
    metadata: dict[str, Any] | None = None


class SessionResponse(BaseModel):
    id: str
    session_type: str
    start_time: str
    is_active: bool
    metadata: dict[str, Any] | None

    class Config:
        from_attributes = True


class TranslateRequest(BaseModel):
    source_text: str = Field(..., min_length=1, max_length=5000)
    source_lang: str = Field(default="en", max_length=16)
    target_lang: str = Field(..., max_length=16)


class TranslateResponse(BaseModel):
    target_text: str
    source_lang: str
    target_lang: str
    confidence: float
    processing_time_ms: float


class SuggestionResponse(BaseModel):
    suggestions: list[str]
    context: dict[str, Any] | None = None
    processing_time_ms: float


class FeedbackRequest(BaseModel):
    session_id: str
    rating: int = Field(..., ge=1, le=5)
    feedback_type: str = Field(default="general", max_length=64)
    comment: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    message: str


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/session", response_model=SessionResponse, status_code=201)
async def create_session(
    body: SessionCreateRequest,
    user_id: str = Depends(get_current_user_id),
    session: Any = Depends(get_session),
) -> SessionResponse:
    comm_session = CommunicationSession(
        user_id=UUID(user_id),
        session_type=body.session_type,
        metadata=body.metadata or {},
    )
    session.add(comm_session)
    await session.flush()
    await session.refresh(comm_session)

    logger.info("session_created", session_id=str(comm_session.id), type=body.session_type)
    return SessionResponse(
        id=str(comm_session.id),
        session_type=comm_session.session_type,
        start_time=comm_session.start_time.isoformat() if comm_session.start_time else "",
        is_active=comm_session.is_active,
        metadata=comm_session.metadata,
    )


@router.post("/translate", response_model=TranslateResponse)
async def translate_communication(
    body: TranslateRequest,
    user_id: str = Depends(get_current_user_id),
    session: Any = Depends(get_session),
) -> TranslateResponse:
    start = time.monotonic()
    try:
        from neurolink.backend.translation.translator import TranslationEngine
        engine = TranslationEngine()
        result = await engine.translate(
            text=body.source_text,
            source_lang=body.source_lang,
            target_lang=body.target_lang,
        )
        elapsed = (time.monotonic() - start) * 1000

        # Persist translation history
        history = TranslationHistory(
            user_id=UUID(user_id),
            source_text=body.source_text,
            target_text=result["text"],
            source_lang=body.source_lang,
            target_lang=body.target_lang,
            confidence=result["confidence"],
        )
        session.add(history)

        logger.info("translation_completed", src=body.source_lang, tgt=body.target_lang, elapsed_ms=round(elapsed, 2))
        return TranslateResponse(
            target_text=result["text"],
            source_lang=body.source_lang,
            target_lang=body.target_lang,
            confidence=result["confidence"],
            processing_time_ms=round(elapsed, 2),
        )
    except Exception as exc:
        logger.error("translation_failed", error=str(exc))
        raise MultimodalFusionError(f"Translation failed: {exc}") from exc


@router.get("/suggest", response_model=SuggestionResponse)
async def get_suggestions(
    context: str = Query("", description="Current context for suggestions"),
    max_suggestions: int = Query(5, ge=1, le=20),
    user_id: str = Depends(get_current_user_id),
) -> SuggestionResponse:
    start = time.monotonic()
    try:
        from neurolink.backend.personalization.suggestion_engine import SuggestionEngine
        engine = SuggestionEngine()
        result = await engine.get_suggestions(
            user_id=UUID(user_id),
            context=context,
            max_suggestions=max_suggestions,
        )
        elapsed = (time.monotonic() - start) * 1000
        return SuggestionResponse(
            suggestions=result.get("suggestions", []),
            context=result.get("context"),
            processing_time_ms=round(elapsed, 2),
        )
    except Exception as exc:
        logger.error("suggestion_failed", error=str(exc))
        return SuggestionResponse(
            suggestions=["Hello", "How are you?", "Thank you", "Yes", "No"],
            processing_time_ms=0,
        )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    body: FeedbackRequest,
    user_id: str = Depends(get_current_user_id),
    session: Any = Depends(get_session),
) -> FeedbackResponse:
    try:
        comm_session = await session.get(CommunicationSession, UUID(body.session_id))
        if not comm_session or str(comm_session.user_id) != user_id:
            raise NotFoundError("Session not found")

        current_meta = comm_session.metadata or {}
        feedback_list = current_meta.get("feedback", [])
        feedback_list.append({
            "rating": body.rating,
            "feedback_type": body.feedback_type,
            "comment": body.comment,
            "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        })
        current_meta["feedback"] = feedback_list
        comm_session.metadata = current_meta
        session.add(comm_session)

        logger.info("feedback_submitted", session_id=body.session_id, rating=body.rating)
        return FeedbackResponse(status="success", message="Feedback recorded")
    except NotFoundError:
        raise
    except Exception as exc:
        logger.error("feedback_failed", error=str(exc))
        return FeedbackResponse(status="error", message=str(exc))
