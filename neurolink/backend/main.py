from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest

from neurolink.backend.api.middleware import setup_middleware
from neurolink.backend.api.v1 import api_v1_router
from neurolink.backend.core.config import settings
from neurolink.backend.core.exceptions import register_exception_handlers
from neurolink.backend.core.logging import get_logger, setup_logging
from neurolink.backend.db import close_db, health_check as db_health_check, init_db
from neurolink.backend.websocket.handlers import websocket_endpoint_handler
from neurolink.backend.websocket.manager import connection_manager

# ── Prometheus metrics ─────────────────────────────────────────────────────

HTTP_REQUESTS_TOTAL = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
HTTP_REQUEST_DURATION = Histogram("http_request_duration_seconds", "HTTP request duration", ["method", "path"])
ACTIVE_WS_CONNECTIONS = Gauge("active_ws_connections", "Active WebSocket connections")
DB_HEALTH = Gauge("db_health", "Database health (1=healthy, 0=unhealthy)")

logger = get_logger("main")

# App startup time for uptime tracking
_STARTUP_TIME = time.monotonic()


# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    """Handle startup and shutdown events."""
    setup_logging()
    logger.info(
        "app_starting",
        name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )

    # ── Startup ────────────────────────────────────────────────────────────
    try:
        await init_db()
        logger.info("database_initialized")
    except Exception as exc:
        logger.error("database_init_failed", error=str(exc))

    try:
        if settings.ENABLE_ANALYTICS:
            await connection_manager.start_heartbeat()
            logger.info("websocket_heartbeat_started")
    except Exception as exc:
        logger.error("websocket_heartbeat_failed", error=str(exc))

    # Pre-load ML models
    if settings.ENABLE_SPEECH_PROCESSING:
        try:
            from neurolink.backend.speech.stt_engine import STTEngine
            _stt = STTEngine()
            await _stt.load()
            logger.info("stt_model_loaded")
        except Exception as exc:
            logger.warning("stt_model_load_failed", error=str(exc))

    if settings.ENABLE_GESTURE_RECOGNITION:
        try:
            from neurolink.backend.ml.gesture_recognizer import GestureRecognizer
            _gesture = GestureRecognizer()
            await _gesture.load()
            logger.info("gesture_model_loaded")
        except Exception as exc:
            logger.warning("gesture_model_load_failed", error=str(exc))

    logger.info("app_startup_complete")
    yield
    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("app_shutting_down")

    try:
        await connection_manager.stop_heartbeat()
    except Exception as exc:
        logger.error("heartbeat_stop_failed", error=str(exc))

    try:
        await close_db()
        logger.info("database_connections_closed")
    except Exception as exc:
        logger.error("db_close_failed", error=str(exc))

    logger.info("app_shutdown_complete")


# ── App factory ────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Adaptive Multimodal Communication Intelligence System",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    openapi_url="/openapi.json" if settings.ENVIRONMENT != "production" else None,
)

# ── Middleware ──────────────────────────────────────────────────────────────

setup_middleware(app)
register_exception_handlers(app)


# ── Metrics middleware ──────────────────────────────────────────────────────

@app.middleware("http")
async def metrics_middleware(request: Request, call_next: Any) -> Any:
    start = time.monotonic()
    response = await call_next(request)
    duration = time.monotonic() - start

    HTTP_REQUESTS_TOTAL.labels(method=request.method, path=request.url.path, status=response.status_code).inc()
    HTTP_REQUEST_DURATION.labels(method=request.method, path=request.url.path).observe(duration)
    return response


# ── Routers ─────────────────────────────────────────────────────────────────

app.include_router(api_v1_router)


# ── Health check ───────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    db_status = await db_health_check()
    is_healthy = db_status.get("status") == "healthy"
    DB_HEALTH.set(1 if is_healthy else 0)

    uptime = (time.monotonic() - _STARTUP_TIME) / 3600

    return JSONResponse(
        content={
            "status": "healthy" if is_healthy else "degraded",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "uptime_hours": round(uptime, 2),
            "database": db_status,
            "websocket_connections": connection_manager.active_connections,
            "active_users": connection_manager.active_users,
        },
        status_code=200 if is_healthy else 503,
    )


# ── Prometheus metrics endpoint ────────────────────────────────────────────

@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type="text/plain",
    )


# ── WebSocket endpoint ─────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: Any) -> None:
    ACTIVE_WS_CONNECTIONS.inc()
    try:
        await websocket_endpoint_handler(websocket)
    finally:
        ACTIVE_WS_CONNECTIONS.dec()


# ── Root ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        content={
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
        }
    )
