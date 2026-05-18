from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field

from neurolink.backend.api.middleware import get_current_user_id, rate_limit_dependency
from neurolink.backend.core.config import settings
from neurolink.backend.core.exceptions import AuthenticationError, ValidationError
from neurolink.backend.core.logging import get_logger
from neurolink.backend.core.security import SecurityManager
from neurolink.backend.db import get_session
from neurolink.backend.db.models import User

logger = get_logger("api.auth")
router = APIRouter()


# ── Schemas ────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    is_active: bool
    is_verified: bool
    preferences: dict[str, Any] | None
    accessibility_settings: dict[str, Any] | None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


# ── Dependencies ───────────────────────────────────────────────────────────

async def get_auth_user(request: Request) -> str:
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise AuthenticationError("Not authenticated")
    return user_id


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    session: Any = Depends(get_session),
) -> TokenResponse:
    existing = await session.get(User, body.email)
    if existing:
        raise ValidationError("Email already registered")

    user = User(
        email=body.email,
        name=body.name,
        hashed_password=SecurityManager.hash_password(body.password),
    )
    session.add(user)
    await session.flush()
    await session.refresh(user)

    logger.info("user_registered", user_id=str(user.id), email=user.email)

    return TokenResponse(
        access_token=SecurityManager.create_access_token(subject=str(user.id)),
        refresh_token=SecurityManager.create_refresh_token(subject=str(user.id)),
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    request: Request,
    session: Any = Depends(get_session),
) -> TokenResponse:
    from sqlalchemy import select

    result = await session.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    if not user or not SecurityManager.verify_password(body.password, user.hashed_password):
        logger.warning("login_failed", email=body.email, client=request.client.host if request.client else None)
        raise AuthenticationError("Invalid email or password")

    if not user.is_active:
        raise AuthenticationError("Account is disabled")

    logger.info("user_logged_in", user_id=str(user.id))
    return TokenResponse(
        access_token=SecurityManager.create_access_token(subject=str(user.id)),
        refresh_token=SecurityManager.create_refresh_token(subject=str(user.id)),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(body: RefreshRequest) -> TokenResponse:
    new_access = SecurityManager.refresh_access_token(body.refresh_token)
    if not new_access:
        raise AuthenticationError("Invalid or expired refresh token")

    return TokenResponse(
        access_token=new_access,
        refresh_token=body.refresh_token,
    )


@router.get("/me", response_model=UserResponse)
async def get_me(
    user_id: str = Depends(get_auth_user),
    session: Any = Depends(get_session),
) -> User:
    from uuid import UUID

    user = await session.get(User, UUID(user_id))
    if not user:
        raise AuthenticationError("User not found")
    return user
