from fastapi import APIRouter

from neurolink.backend.api.v1.auth import router as auth_router
from neurolink.backend.api.v1.gestures import router as gestures_router
from neurolink.backend.api.v1.speech import router as speech_router
from neurolink.backend.api.v1.communication import router as communication_router
from neurolink.backend.api.v1.analytics import router as analytics_router

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_v1_router.include_router(gestures_router, prefix="/gestures", tags=["Gestures"])
api_v1_router.include_router(speech_router, prefix="/speech", tags=["Speech"])
api_v1_router.include_router(communication_router, prefix="/communication", tags=["Communication"])
api_v1_router.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])

__all__ = ["api_v1_router"]
