from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppException(Exception):
    """Base exception for all application-level errors."""

    status_code: int = 500
    detail: str = "An internal error occurred"
    error_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        detail: str | None = None,
        status_code: int | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.detail = detail or self.detail
        self.status_code = status_code or self.status_code
        self.error_code = error_code or self.error_code
        self.context = context or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.error_code,
                "message": self.detail,
                "context": self.context,
            }
        }


class GestureProcessingError(AppException):
    status_code = 422
    detail = "Gesture processing failed"
    error_code = "GESTURE_PROCESSING_ERROR"


class SpeechProcessingError(AppException):
    status_code = 422
    detail = "Speech processing failed"
    error_code = "SPEECH_PROCESSING_ERROR"


class EmotionDetectionError(AppException):
    status_code = 422
    detail = "Emotion detection failed"
    error_code = "EMOTION_DETECTION_ERROR"


class MultimodalFusionError(AppException):
    status_code = 500
    detail = "Multimodal fusion failed"
    error_code = "MULTIMODAL_FUSION_ERROR"


class PersonalizationError(AppException):
    status_code = 422
    detail = "Personalization operation failed"
    error_code = "PERSONALIZATION_ERROR"


class EdgeDeploymentError(AppException):
    status_code = 500
    detail = "Edge deployment operation failed"
    error_code = "EDGE_DEPLOYMENT_ERROR"


class AuthenticationError(AppException):
    status_code = 401
    detail = "Authentication failed"
    error_code = "AUTHENTICATION_ERROR"


class AuthorizationError(AppException):
    status_code = 403
    detail = "Not authorized"
    error_code = "AUTHORIZATION_ERROR"


class NotFoundError(AppException):
    status_code = 404
    detail = "Resource not found"
    error_code = "NOT_FOUND"


class ValidationError(AppException):
    status_code = 422
    detail = "Validation failed"
    error_code = "VALIDATION_ERROR"


class RateLimitError(AppException):
    status_code = 429
    detail = "Rate limit exceeded"
    error_code = "RATE_LIMIT_ERROR"


class DatabaseError(AppException):
    status_code = 500
    detail = "Database operation failed"
    error_code = "DATABASE_ERROR"


# ── FastAPI exception handlers ──────────────────────────────────────────────


async def _app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


async def _generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "context": {},
            }
        },
    )


async def _validation_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    errors = getattr(exc, "errors", lambda: [{"msg": str(exc)}])()
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "context": {"details": errors},
            }
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all custom exception handlers on a FastAPI instance."""
    from fastapi.exceptions import RequestValidationError

    app.add_exception_handler(AppException, _app_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(Exception, _generic_exception_handler)
