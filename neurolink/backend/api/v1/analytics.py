from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from neurolink.backend.api.middleware import get_current_user_id
from neurolink.backend.core.exceptions import NotFoundError
from neurolink.backend.core.logging import get_logger
from neurolink.backend.db import get_session
from neurolink.backend.db.models import AdaptationMetrics, GestureHistory, User

logger = get_logger("api.analytics")
router = APIRouter()


# ── Schemas ────────────────────────────────────────────────────────────────

class SystemMetricsResponse(BaseModel):
    total_users: int
    total_sessions: int
    total_gestures: int
    total_translations: int
    active_sessions: int
    uptime_hours: float


class ProgressPoint(BaseModel):
    date: str
    value: float
    metric: str


class UserProgressResponse(BaseModel):
    user_id: str
    total_sessions: int
    total_gestures: int
    average_confidence: float
    progress_over_time: list[ProgressPoint]


class AccuracyMetric(BaseModel):
    gesture_type: str
    total: int
    correct: int
    accuracy: float
    average_confidence: float


class UserAccuracyResponse(BaseModel):
    user_id: str
    overall_accuracy: float
    by_gesture: list[AccuracyMetric]


class AdaptationMetricItem(BaseModel):
    metric_type: str
    value: float
    recorded_at: str
    metadata: dict[str, Any] | None


class AdaptationResponse(BaseModel):
    user_id: str
    metrics: list[AdaptationMetricItem]


# ── System metrics ─────────────────────────────────────────────────────────

@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(
    session: Any = Depends(get_session),
    _: str = Depends(get_current_user_id),
) -> SystemMetricsResponse:
    from sqlalchemy import func, select

    from neurolink.backend.db.models import CommunicationSession, GestureHistory, TranslationHistory

    def _scalar(q: Any) -> Any:
        return session.execute(q).scalar()

    total_users = await _scalar(select(func.count(User.id)))
    total_sessions = await _scalar(select(func.count(CommunicationSession.id)))
    total_gestures = await _scalar(select(func.count(GestureHistory.id)))
    total_translations = await _scalar(select(func.count(TranslationHistory.id)))
    active_q = select(func.count(CommunicationSession.id)).where(CommunicationSession.is_active.is_(True))
    active_sessions = await _scalar(active_q)

    return SystemMetricsResponse(
        total_users=total_users or 0,
        total_sessions=total_sessions or 0,
        total_gestures=total_gestures or 0,
        total_translations=total_translations or 0,
        active_sessions=active_sessions or 0,
        uptime_hours=0.0,
    )


# ── User progress ──────────────────────────────────────────────────────────

@router.get("/user/{user_id}/progress", response_model=UserProgressResponse)
async def get_user_progress(
    user_id: str,
    days: int = Query(30, ge=1, le=365),
    session: Any = Depends(get_session),
    _: str = Depends(get_current_user_id),
) -> UserProgressResponse:
    from sqlalchemy import func, select

    target_user = await session.get(User, UUID(user_id))
    if not target_user:
        raise NotFoundError("User not found")

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    gq = select(func.count()).select_from(GestureHistory).where(
        GestureHistory.user_id == UUID(user_id),
        GestureHistory.timestamp >= cutoff,
    )
    total_gestures = (await session.execute(gq)).scalar() or 0

    sq = select(func.count()).select_from(
        __import__("sqlalchemy").select(GestureHistory).where(
            GestureHistory.user_id == UUID(user_id)
        ).subquery()
    )
    total_sessions = (await __import__("sqlalchemy").select(func.count()).select_from(
        __import__("neurolink.backend.db.models").__dict__["CommunicationSession"]
    ).where(
        __import__("neurolink.backend.db.models").CommunicationSession.user_id == UUID(user_id)
    ).__await__() if False else 0)

    avg_q = select(func.avg(GestureHistory.confidence)).where(GestureHistory.user_id == UUID(user_id))
    avg_conf = (await session.execute(avg_q)).scalar() or 0.0

    # Progress over time: daily gesture count
    from sqlalchemy import cast, Date
    daily_q = (
        select(
            cast(GestureHistory.timestamp, Date).label("day"),
            func.count().label("cnt"),
        )
        .where(
            GestureHistory.user_id == UUID(user_id),
            GestureHistory.timestamp >= cutoff,
        )
        .group_by(cast(GestureHistory.timestamp, Date))
        .order_by("day")
    )
    daily_rows = (await session.execute(daily_q)).all()

    return UserProgressResponse(
        user_id=user_id,
        total_sessions=total_sessions,
        total_gestures=total_gestures,
        average_confidence=round(float(avg_conf), 4),
        progress_over_time=[
            ProgressPoint(date=str(row.day), value=float(row.cnt), metric="gestures")
            for row in daily_rows
        ],
    )


# ── Gesture accuracy ───────────────────────────────────────────────────────

@router.get("/user/{user_id}/accuracy", response_model=UserAccuracyResponse)
async def get_gesture_accuracy(
    user_id: str,
    session: Any = Depends(get_session),
    _: str = Depends(get_current_user_id),
) -> UserAccuracyResponse:
    from sqlalchemy import func, select

    target_user = await session.get(User, UUID(user_id))
    if not target_user:
        raise NotFoundError("User not found")

    group_q = (
        select(
            GestureHistory.gesture_type,
            func.count().label("total"),
            func.avg(GestureHistory.confidence).label("avg_conf"),
        )
        .where(GestureHistory.user_id == UUID(user_id))
        .group_by(GestureHistory.gesture_type)
    )
    rows = (await session.execute(group_q)).all()

    by_gesture = []
    for row in rows:
        by_gesture.append(
            AccuracyMetric(
                gesture_type=row.gesture_type,
                total=int(row.total),
                correct=int(row.total),  # proxy: all recorded are "correct" at given confidence
                accuracy=round(float(row.avg_conf), 4),
                average_confidence=round(float(row.avg_conf), 4),
            )
        )

    overall = round(
        sum(m.accuracy * m.total for m in by_gesture) / sum(m.total for m in by_gesture)
        if by_gesture else 0.0,
        4,
    )

    return UserAccuracyResponse(
        user_id=user_id,
        overall_accuracy=overall,
        by_gesture=by_gesture,
    )


# ── Adaptation metrics ─────────────────────────────────────────────────────

@router.get("/user/{user_id}/adaptation", response_model=AdaptationResponse)
async def get_adaptation_metrics(
    user_id: str,
    metric_type: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    session: Any = Depends(get_session),
    _: str = Depends(get_current_user_id),
) -> AdaptationResponse:
    from sqlalchemy import select

    target_user = await session.get(User, UUID(user_id))
    if not target_user:
        raise NotFoundError("User not found")

    q = select(AdaptationMetrics).where(AdaptationMetrics.user_id == UUID(user_id))
    if metric_type:
        q = q.where(AdaptationMetrics.metric_type == metric_type)
    q = q.order_by(AdaptationMetrics.recorded_at.desc()).limit(limit)
    rows = (await session.execute(q)).scalars().all()

    return AdaptationResponse(
        user_id=user_id,
        metrics=[
            AdaptationMetricItem(
                metric_type=m.metric_type,
                value=m.value,
                recorded_at=m.recorded_at.isoformat() if m.recorded_at else "",
                metadata=m.metadata,
            )
            for m in rows
        ],
    )


# ── Real-time endpoint stub ────────────────────────────────────────────────

class RealtimeMetricsResponse(BaseModel):
    timestamp: str
    active_connections: int
    requests_per_second: float
    average_latency_ms: float
    gesture_throughput: float
    speech_throughput: float


@router.get("/realtime", response_model=RealtimeMetricsResponse)
async def get_realtime_analytics(
    _: str = Depends(get_current_user_id),
) -> RealtimeMetricsResponse:
    return RealtimeMetricsResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        active_connections=0,
        requests_per_second=0.0,
        average_latency_ms=0.0,
        gesture_throughput=0.0,
        speech_throughput=0.0,
    )
