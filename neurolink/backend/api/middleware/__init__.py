from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from neurolink.backend.core.config import settings
from neurolink.backend.core.exceptions import AuthenticationError, RateLimitError
from neurolink.backend.core.logging import get_logger, RequestIDMiddleware
from neurolink.backend.core.security import SecurityManager

logger = get_logger("middleware")


# ── CORS ───────────────────────────────────────────────────────────────────

def setup_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_CREDENTIALS,
        allow_methods=settings.CORS_METHODS,
        allow_headers=settings.CORS_HEADERS,
    )
    logger.info("cors_configured", origins=settings.CORS_ORIGINS)


# ── Authentication Middleware ──────────────────────────────────────────────

class AuthMiddleware(BaseHTTPMiddleware):
    """Extract and validate JWT from Authorization header; attach user_id to request.state."""

    PUBLIC_PATHS: set[str] = {
        "/api/v1/auth/register",
        "/api/v1/auth/login",
        "/api/v1/auth/refresh",
        "/health",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
    }

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path

        # Skip auth for public paths
        if any(path.startswith(p) for p in self.PUBLIC_PATHS):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        api_key = request.headers.get(settings.API_KEY_HEADER)

        user_id: str | None = None

        if auth_header:
            token = SecurityManager.sanitize_token_header(auth_header)
            if token:
                user_id = SecurityManager.get_subject_from_token(token)

        if not user_id and api_key:
            # API key validation would go here against stored hashes
            pass

        if not user_id:
            raise AuthenticationError("Missing or invalid authentication token")

        request.state.user_id = user_id
        return await call_next(request)


# ── Rate Limiting ─────────────────────────────────────────────────────────

class RateLimitStore:
    """Simple in-memory store for rate limiting (fallback when Redis is unavailable)."""

    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, key: str, max_requests: int, window_seconds: int) -> bool:
        now = time.monotonic()
        cutoff = now - window_seconds
        async with self._lock:
            timestamps = self._store.get(key, [])
            timestamps = [t for t in timestamps if t > cutoff]
            if len(timestamps) >= max_requests:
                return False
            timestamps.append(now)
            self._store[key] = timestamps
            return True


_rate_limit_store = RateLimitStore()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce per-IP rate limits on non-public paths."""

    PUBLIC_PATHS: set[str] = {"/health", "/metrics"}

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)

        path = request.url.path
        if any(path.startswith(p) for p in self.PUBLIC_PATHS):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        key = SecurityManager.build_rate_limit_key(client_ip)

        allowed = await _rate_limit_store.check(
            key,
            settings.RATE_LIMIT_DEFAULT,
            settings.RATE_LIMIT_WINDOW,
        )
        if not allowed:
            raise RateLimitError("Too many requests")

        return await call_next(request)


# ── Logging Middleware ─────────────────────────────────────────────────────

class LoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with timing, status, and metadata."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        request.state.request_id = request_id

        with structlog.contextvars.bound_contextvars(request_id=request_id):
            start = time.monotonic()
            response = await call_next(request)
            elapsed_ms = (time.monotonic() - start) * 1000

            response.headers["X-Request-ID"] = request_id
            logger.info(
                "request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                elapsed_ms=round(elapsed_ms, 2),
                client=request.client.host if request.client else None,
            )

        return response


# ── Setup all middleware ───────────────────────────────────────────────────

def setup_middleware(app: FastAPI) -> None:
    """Register all middleware in order (last added = first executed)."""
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)
    setup_cors(app)
    logger.info("middleware_configured")


# ── Dependencies used by route handlers ────────────────────────────────────

async def get_current_user_id(request: Request) -> str:
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise AuthenticationError("Not authenticated")
    return user_id


async def rate_limit_dependency(request: Request) -> None:
    """Optional per-endpoint rate limit dependency."""
    pass
