from __future__ import annotations

import logging
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from neurolink.backend.core.config import settings


def _build_processor_list() -> list[Callable[..., Any]]:
    procs: list[Callable[..., Any]] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.LOG_FORMAT == "json":
        procs.append(structlog.processors.JSONRenderer(serializer=None))
    else:
        procs.append(structlog.dev.ConsoleRenderer())

    return procs


def setup_logging() -> None:
    """Configure structlog and stdlib logging once at startup."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handler: logging.Handler
    if settings.LOG_FILE:
        handler = logging.handlers.RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=settings.LOG_MAX_BYTES,
            backupCount=settings.LOG_BACKUP_COUNT,
        )
    else:
        handler = logging.StreamHandler(sys.stdout)

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=_build_processor_list(),
        foreign_pre_chain=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
        ],
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Quiet noisy third-party loggers
    for name in ("uvicorn.access", "httpx", "httpcore", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name or __name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request_id into every request's state and log context."""

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
            response: Response = await call_next(request)
            elapsed_ms = (time.monotonic() - start) * 1000

            response.headers["X-Request-ID"] = request_id
            get_logger("http").info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                elapsed_ms=round(elapsed_ms, 2),
                client_host=request.client.host if request.client else None,
            )

        return response


class PerformanceLogger:
    """Context manager / decorator for timing code blocks."""

    def __init__(self, logger: structlog.stdlib.BoundLogger, operation: str, **context: Any) -> None:
        self._logger = logger
        self._operation = operation
        self._context = context
        self._start: float | None = None

    def __enter__(self) -> PerformanceLogger:
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = (time.monotonic() - self._start) * 1000 if self._start else 0
        self._logger.info(
            "perf",
            operation=self._operation,
            elapsed_ms=round(elapsed, 2),
            **self._context,
        )
