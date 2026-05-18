from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    APP_NAME: str = "Neurolink"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False

    @field_validator("DEBUG", mode="before")
    @classmethod
    def derive_debug(cls, v: bool | None, info) -> bool:
        if v is not None:
            return v
        return info.data.get("ENVIRONMENT", "development") == "development"

    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # ── Database ─────────────────────────────────────────────────────────────
    POSTGRES_USER: str = "neurolink"
    POSTGRES_PASSWORD: str = "neurolink"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "neurolink"
    DATABASE_URL: PostgresDsn | None = None
    DB_POOL_MIN_SIZE: int = 5
    DB_POOL_MAX_SIZE: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_ECHO: bool = False

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_url(cls, v: str | None, info) -> str:
        if v:
            return v
        return str(
            PostgresDsn.build(
                scheme="postgresql+asyncpg",
                username=info.data.get("POSTGRES_USER", "neurolink"),
                password=info.data.get("POSTGRES_PASSWORD", "neurolink"),
                host=info.data.get("POSTGRES_HOST", "localhost"),
                port=info.data.get("POSTGRES_PORT", 5432),
                path=info.data.get("POSTGRES_DB", "neurolink"),
            )
        )

    # ── Redis ────────────────────────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None
    REDIS_URL: RedisDsn | None = None

    @field_validator("REDIS_URL", mode="before")
    @classmethod
    def assemble_redis_url(cls, v: str | None, info) -> str:
        if v:
            return v
        password = info.data.get("REDIS_PASSWORD")
        pw_part = f":{password}@" if password else ""
        return f"redis://{pw_part}{info.data.get('REDIS_HOST', 'localhost')}:{info.data.get('REDIS_PORT', 6379)}/{info.data.get('REDIS_DB', 0)}"

    REDIS_SESSION_TTL: int = 3600  # 1 hour
    REDIS_RATE_LIMIT_TTL: int = 60
    REDIS_MAX_CONNECTIONS: int = 20

    # ── WebSocket ────────────────────────────────────────────────────────────
    WS_MAX_SIZE: int = 1_048_576  # 1 MB
    WS_COMPRESSION: bool = True
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_HEARTBEAT_TIMEOUT: int = 10
    WS_MAX_CONNECTIONS_PER_USER: int = 5

    # ── JWT ──────────────────────────────────────────────────────────────────
    JWT_SECRET: str = Field(default="change-me-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── CORS ─────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list[str] = ["*"]
    CORS_HEADERS: list[str] = ["*"]

    # ── Model paths ──────────────────────────────────────────────────────────
    MODELS_DIR: Path = Field(default_factory=lambda: Path(os.getenv("MODELS_DIR", "models")))
    WHISPER_MODEL_SIZE: str = "base"
    TTS_MODEL_NAME: str = "tts_models/en/ljspeech/tacotron2-DDC"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    GESTURE_MODEL_PATH: str = ""
    EMOTION_MODEL_PATH: str = ""

    # ── Feature flags ────────────────────────────────────────────────────────
    ENABLE_GESTURE_RECOGNITION: bool = True
    ENABLE_SPEECH_PROCESSING: bool = True
    ENABLE_EMOTION_DETECTION: bool = True
    ENABLE_TRANSLATION: bool = True
    ENABLE_PERSONALIZATION: bool = True
    ENABLE_ANALYTICS: bool = True
    ENABLE_EDGE_DEPLOYMENT: bool = False

    # ── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "json"
    LOG_FILE: str | None = None
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
    LOG_BACKUP_COUNT: int = 5

    # ── Rate limiting ────────────────────────────────────────────────────────
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT: int = 100  # requests per window
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # ── Monitoring ───────────────────────────────────────────────────────────
    ENABLE_METRICS: bool = True
    ENABLE_OPENTELEMETRY: bool = True
    OTLP_ENDPOINT: str = "http://localhost:4318"
    PROMETHEUS_PORT: int = 9090

    # ── Security ─────────────────────────────────────────────────────────────
    BCRYPT_ROUNDS: int = 12
    API_KEY_HEADER: str = "X-API-Key"
    MAX_LOGIN_ATTEMPTS: int = 5
    LOGIN_LOCKOUT_MINUTES: int = 15


settings = Settings()
