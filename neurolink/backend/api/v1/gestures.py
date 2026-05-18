from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, Field

from neurolink.backend.api.middleware import get_current_user_id
from neurolink.backend.core.exceptions import GestureProcessingError, NotFoundError
from neurolink.backend.core.logging import get_logger
from neurolink.backend.db import get_session
from neurolink.backend.db.models import GestureHistory

logger = get_logger("api.gestures")
router = APIRouter()


# ── Schemas ────────────────────────────────────────────────────────────────

class RecognizeRequest(BaseModel):
    frame_data: str = Field(..., description="Base64-encoded video frame")
    model_version: str | None = None


class RecognizeResponse(BaseModel):
    gesture_type: str
    confidence: float
    landmarks: list[dict[str, float]] | None = None
    processing_time_ms: float


class TrainRequest(BaseModel):
    gesture_type: str = Field(..., min_length=1, max_length=128)
    landmarks: list[dict[str, float]] = Field(default_factory=list)
    label: str | None = None


class TrainResponse(BaseModel):
    gesture_type: str
    samples_collected: int
    status: str


class GestureHistoryItem(BaseModel):
    id: str
    gesture_type: str
    confidence: float
    timestamp: str

    class Config:
        from_attributes = True


class GestureHistoryResponse(BaseModel):
    items: list[GestureHistoryItem]
    total: int
    page: int
    page_size: int


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/recognize", response_model=RecognizeResponse)
async def recognize_gesture(
    body: RecognizeRequest,
    user_id: str = Depends(get_current_user_id),
    session: Any = Depends(get_session),
) -> RecognizeResponse:
    import time
    start = time.monotonic()

    try:
        from neurolink.backend.ml.gesture_recognizer import GestureRecognizer
        recognizer = GestureRecognizer()
        result = await recognizer.predict(body.frame_data, model_version=body.model_version)

        history = GestureHistory(
            user_id=UUID(user_id),
            gesture_type=result["gesture_type"],
            landmarks=result.get("landmarks"),
            confidence=result["confidence"],
            metadata={"model_version": body.model_version or "default"},
        )
        session.add(history)

        elapsed = (time.monotonic() - start) * 1000
        logger.info("gesture_recognized", gesture=result["gesture_type"], confidence=result["confidence"], elapsed_ms=round(elapsed, 2))

        return RecognizeResponse(
            gesture_type=result["gesture_type"],
            confidence=result["confidence"],
            landmarks=result.get("landmarks"),
            processing_time_ms=round(elapsed, 2),
        )

    except Exception as exc:
        logger.error("gesture_recognition_failed", error=str(exc))
        raise GestureProcessingError(f"Recognition failed: {exc}") from exc


@router.post("/train", response_model=TrainResponse)
async def train_gesture(
    body: TrainRequest,
    user_id: str = Depends(get_current_user_id),
) -> TrainResponse:
    try:
        from neurolink.backend.ml.gesture_trainer import GestureTrainer
        trainer = GestureTrainer()
        result = await trainer.add_sample(
            user_id=UUID(user_id),
            gesture_type=body.gesture_type,
            landmarks=body.landmarks,
            label=body.label,
        )
        logger.info("gesture_trained", gesture=body.gesture_type, samples=result.get("samples", 0))
        return TrainResponse(
            gesture_type=body.gesture_type,
            samples_collected=result.get("samples", 0),
            status="success",
        )
    except Exception as exc:
        logger.error("gesture_training_failed", error=str(exc))
        raise GestureProcessingError(f"Training failed: {exc}") from exc


@router.get("/history", response_model=GestureHistoryResponse)
async def get_gesture_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    gesture_type: str | None = Query(None),
    user_id: str = Depends(get_current_user_id),
    session: Any = Depends(get_session),
) -> GestureHistoryResponse:
    from sqlalchemy import func, select

    base = select(GestureHistory).where(GestureHistory.user_id == UUID(user_id))
    if gesture_type:
        base = base.where(GestureHistory.gesture_type == gesture_type)

    count_q = select(func.count()).select_from(base.subquery())
    total_result = await session.execute(count_q)
    total = total_result.scalar() or 0

    items_q = base.order_by(GestureHistory.timestamp.desc()).offset((page - 1) * page_size).limit(page_size)
    items_result = await session.execute(items_q)
    items = items_result.scalars().all()

    return GestureHistoryResponse(
        items=[
            GestureHistoryItem(
                id=str(item.id),
                gesture_type=item.gesture_type,
                confidence=item.confidence,
                timestamp=item.timestamp.isoformat() if item.timestamp else "",
            )
            for item in items
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.delete("/gesture/{gesture_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_gesture(
    gesture_id: str,
    user_id: str = Depends(get_current_user_id),
    session: Any = Depends(get_session),
) -> None:
    try:
        gid = UUID(gesture_id)
    except ValueError:
        raise GestureProcessingError("Invalid gesture ID")

    gesture = await session.get(GestureHistory, gid)
    if not gesture or str(gesture.user_id) != user_id:
        raise NotFoundError("Gesture not found")

    await session.delete(gesture)
    logger.info("gesture_deleted", gesture_id=gesture_id)
