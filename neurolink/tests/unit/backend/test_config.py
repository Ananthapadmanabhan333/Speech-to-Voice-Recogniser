from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from neurolink.backend.core.config import Settings, settings


class TestSettingsDefaults:
    """Test default values for all settings."""

    def test_app_defaults(self) -> None:
        assert settings.APP_NAME == "Neurolink"
        assert settings.APP_VERSION == "1.0.0"
        assert settings.ENVIRONMENT in ("development", "staging", "production")
        assert isinstance(settings.DEBUG, bool)

    def test_database_defaults(self) -> None:
        assert settings.POSTGRES_USER == "neurolink"
        assert settings.POSTGRES_PORT == 5432
        assert settings.DB_POOL_MIN_SIZE == 5
        assert settings.DB_POOL_MAX_SIZE == 20
        assert settings.DATABASE_URL is not None
        assert "postgresql+asyncpg" in str(settings.DATABASE_URL)

    def test_redis_defaults(self) -> None:
        assert settings.REDIS_HOST == "localhost"
        assert settings.REDIS_PORT == 6379
        assert settings.REDIS_DB == 0
        assert settings.REDIS_SESSION_TTL == 3600
        assert "redis://" in str(settings.REDIS_URL)

    def test_jwt_defaults(self) -> None:
        assert settings.JWT_ALGORITHM == "HS256"
        assert settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES == 30
        assert settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS == 7

    def test_websocket_defaults(self) -> None:
        assert settings.WS_MAX_SIZE == 1_048_576
        assert settings.WS_COMPRESSION is True
        assert settings.WS_HEARTBEAT_INTERVAL == 30
        assert settings.WS_MAX_CONNECTIONS_PER_USER == 5

    def test_rate_limit_defaults(self) -> None:
        assert settings.RATE_LIMIT_ENABLED is True
        assert settings.RATE_LIMIT_DEFAULT == 100
        assert settings.RATE_LIMIT_WINDOW == 60

    def test_security_defaults(self) -> None:
        assert settings.BCRYPT_ROUNDS == 12
        assert settings.API_KEY_HEADER == "X-API-Key"
        assert settings.MAX_LOGIN_ATTEMPTS == 5

    def test_model_path_defaults(self) -> None:
        assert settings.WHISPER_MODEL_SIZE == "base"
        assert settings.EMBEDDING_MODEL_NAME == "all-MiniLM-L6-v2"

    def test_feature_flags_defaults(self) -> None:
        assert settings.ENABLE_GESTURE_RECOGNITION is True
        assert settings.ENABLE_SPEECH_PROCESSING is True
        assert settings.ENABLE_EMOTION_DETECTION is True
        assert settings.ENABLE_TRANSLATION is True
        assert settings.ENABLE_PERSONALIZATION is True
        assert settings.ENABLE_ANALYTICS is True

    def test_logging_defaults(self) -> None:
        assert settings.LOG_LEVEL == "INFO"
        assert settings.LOG_FORMAT in ("json", "console")
        assert settings.LOG_MAX_BYTES == 10 * 1024 * 1024
        assert settings.LOG_BACKUP_COUNT == 5


class TestEnvironmentOverrides:
    """Test environment variable overrides."""

    @pytest.fixture(autouse=True)
    def _setup_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_NAME", "Neurolink-Test")
        monkeypatch.setenv("ENVIRONMENT", "staging")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("POSTGRES_USER", "test_user")
        monkeypatch.setenv("POSTGRES_DB", "test_db")
        monkeypatch.setenv("REDIS_HOST", "test-redis.example.com")
        monkeypatch.setenv("JWT_SECRET", "test-secret-key")
        monkeypatch.setenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "15")
        monkeypatch.setenv("WS_MAX_SIZE", "2097152")
        monkeypatch.setenv("RATE_LIMIT_DEFAULT", "500")
        monkeypatch.setenv("BCRYPT_ROUNDS", "10")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("ENABLE_EDGE_DEPLOYMENT", "true")
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("REDIS_URL", raising=False)

    def test_env_overrides_app(self) -> None:
        test_settings = Settings()
        assert test_settings.APP_NAME == "Neurolink-Test"
        assert test_settings.ENVIRONMENT == "staging"
        assert test_settings.DEBUG is True

    def test_env_overrides_database(self) -> None:
        test_settings = Settings()
        assert test_settings.POSTGRES_USER == "test_user"
        assert test_settings.POSTGRES_DB == "test_db"
        assert "test_user" in str(test_settings.DATABASE_URL)
        assert "test_db" in str(test_settings.DATABASE_URL)

    def test_env_overrides_redis(self) -> None:
        test_settings = Settings()
        assert test_settings.REDIS_HOST == "test-redis.example.com"
        assert "test-redis.example.com" in str(test_settings.REDIS_URL)

    def test_env_overrides_jwt(self) -> None:
        test_settings = Settings()
        assert test_settings.JWT_SECRET == "test-secret-key"
        assert test_settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES == 15

    def test_env_overrides_websocket(self) -> None:
        test_settings = Settings()
        assert test_settings.WS_MAX_SIZE == 2_097_152

    def test_env_overrides_rate_limit(self) -> None:
        test_settings = Settings()
        assert test_settings.RATE_LIMIT_DEFAULT == 500

    def test_env_overrides_security(self) -> None:
        test_settings = Settings()
        assert test_settings.BCRYPT_ROUNDS == 10

    def test_env_overrides_logging(self) -> None:
        test_settings = Settings()
        assert test_settings.LOG_LEVEL == "DEBUG"

    def test_env_overrides_feature_flags(self) -> None:
        test_settings = Settings()
        assert test_settings.ENABLE_EDGE_DEPLOYMENT is True


class TestValidation:
    """Test configuration validation."""

    def test_invalid_environment(self) -> None:
        with pytest.raises(ValidationError):
            Settings(ENVIRONMENT="invalid")  # type: ignore[arg-type]

    def test_invalid_log_format(self) -> None:
        with pytest.raises(ValidationError):
            Settings(LOG_FORMAT="xml")  # type: ignore[arg-type]

    def test_invalid_port_range(self) -> None:
        with pytest.raises(ValidationError):
            Settings(POSTGRES_PORT=99999)

    def test_debug_derived_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.delenv("DEBUG", raising=False)
        test_settings = Settings()
        assert test_settings.DEBUG is False

    def test_debug_explicit_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("DEBUG", "true")
        test_settings = Settings()
        assert test_settings.DEBUG is True

    def test_database_url_custom(self) -> None:
        custom_url = "postgresql+asyncpg://custom:pass@host:5432/custom_db"
        s = Settings(DATABASE_URL=custom_url)
        assert str(s.DATABASE_URL) == custom_url

    def test_redis_url_custom(self) -> None:
        custom_url = "redis://:pass@custom-host:9999/5"
        s = Settings(REDIS_URL=custom_url)
        assert str(s.REDIS_URL) == custom_url


class TestSettingsInstance:
    """Test the global settings singleton."""

    def test_settings_is_singleton(self) -> None:
        from neurolink.backend.core.config import settings as s1
        from neurolink.backend.core.config import settings as s2
        assert s1 is s2

    def test_settings_has_required_fields(self) -> None:
        required = [
            "APP_NAME", "APP_VERSION", "ENVIRONMENT", "DEBUG",
            "DATABASE_URL", "REDIS_URL",
            "JWT_SECRET", "JWT_ALGORITHM",
        ]
        for field in required:
            assert hasattr(settings, field), f"Missing field: {field}"

    def test_settings_types(self) -> None:
        assert isinstance(settings.APP_NAME, str)
        assert isinstance(settings.DEBUG, bool)
        assert isinstance(settings.POSTGRES_PORT, int)
        assert isinstance(settings.CORS_ORIGINS, list)
        assert isinstance(settings.MODELS_DIR, Path)

    def test_models_dir_creatable(self) -> None:
        assert settings.MODELS_DIR is not None
        # Just verify it points to a valid path
        assert isinstance(settings.MODELS_DIR, Path)
