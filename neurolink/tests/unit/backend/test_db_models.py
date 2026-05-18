from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import pytest
from sqlalchemy import Column, inspect, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from neurolink.backend.db.models import (
    AdaptationMetrics,
    Base,
    CommunicationSession,
    EmotionalAnalytics,
    GestureHistory,
    PersonalizationMemory,
    PhrasePrediction,
    TranslationHistory,
    User,
    _new_uuid,
    _utcnow,
)


@pytest.fixture
async def in_memory_db() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session

    await engine.dispose()


class TestUtilityFunctions:
    """Test utility functions used by models."""

    def test_new_uuid(self) -> None:
        uid = _new_uuid()
        assert isinstance(uid, uuid.UUID)
        assert uid.version == 4

    def test_utcnow(self) -> None:
        now = _utcnow()
        assert isinstance(now, datetime)
        assert now.tzinfo is not None


class TestUserModel:
    """Test User model creation and validation."""

    async def test_create_user(self, in_memory_db: AsyncSession) -> None:
        user = User(
            email="test@example.com",
            name="Test User",
            hashed_password="hashed_pw_123",
        )
        in_memory_db.add(user)
        await in_memory_db.commit()
        await in_memory_db.refresh(user)

        assert user.id is not None
        assert isinstance(user.id, uuid.UUID)
        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert user.is_active is True
        assert user.is_verified is False
        assert user.preferences == {}
        assert user.created_at is not None
        assert user.updated_at is not None

    async def test_user_defaults(self, in_memory_db: AsyncSession) -> None:
        user = User(
            email="defaults@example.com",
            name="Defaults",
            hashed_password="pw",
        )
        in_memory_db.add(user)
        await in_memory_db.commit()

        assert user.is_active is True
        assert user.is_verified is False
        assert user.preferences == {}
        assert user.accessibility_settings == {}

    async def test_user_unique_email(self, in_memory_db: AsyncSession) -> None:
        user1 = User(email="dup@example.com", name="User1", hashed_password="pw1")
        user2 = User(email="dup@example.com", name="User2", hashed_password="pw2")
        in_memory_db.add_all([user1, user2])
        with pytest.raises(Exception):
            await in_memory_db.commit()

    async def test_user_relationships(self, in_memory_db: AsyncSession) -> None:
        user = User(email="rel@example.com", name="Rel", hashed_password="pw")
        gesture = GestureHistory(user=user, gesture_type="thumbs_up")
        session = CommunicationSession(user=user, session_type="test")
        emotion = EmotionalAnalytics(user=user, emotion_type="happy")
        memory = PersonalizationMemory(user=user, memory_type="pref", key="theme", value={"val": "dark"})

        in_memory_db.add_all([user, gesture, session, emotion, memory])
        await in_memory_db.commit()
        await in_memory_db.refresh(user)

        assert len(user.gesture_histories) == 1
        assert len(user.communication_sessions) == 1
        assert len(user.emotional_analytics) == 1
        assert len(user.personalization_memories) == 1

    async def test_user_cascade_delete(self, in_memory_db: AsyncSession) -> None:
        user = User(email="cascade@example.com", name="Cascade", hashed_password="pw")
        GestureHistory(user=user, gesture_type="wave")
        GestureHistory(user=user, gesture_type="point")
        in_memory_db.add(user)
        await in_memory_db.commit()

        await in_memory_db.delete(user)
        await in_memory_db.commit()

        result = await in_memory_db.execute(select(GestureHistory))
        assert result.scalars().all() == []

    async def test_user_serialization(self, in_memory_db: AsyncSession) -> None:
        user = User(
            email="serial@example.com",
            name="Serial",
            hashed_password="pw",
            preferences={"theme": "dark", "lang": "en"},
        )
        in_memory_db.add(user)
        await in_memory_db.commit()

        assert user.preferences["theme"] == "dark"
        assert user.preferences["lang"] == "en"

    async def test_user_indexes(self) -> None:
        mapper = inspect(User)
        cols = {c.name: c for c in mapper.columns}
        assert cols["email"].unique is True
        assert cols["email"].index is True


class TestGestureHistoryModel:
    """Test GestureHistory model."""

    async def test_create_gesture_history(self, in_memory_db: AsyncSession) -> None:
        user = User(email="gesture@example.com", name="Gesture", hashed_password="pw")
        gesture = GestureHistory(
            user=user,
            gesture_type="thumbs_up",
            confidence=0.95,
            landmarks={"wrist": {"x": 0.5, "y": 0.3}},
        )
        in_memory_db.add_all([user, gesture])
        await in_memory_db.commit()
        await in_memory_db.refresh(gesture)

        assert gesture.gesture_type == "thumbs_up"
        assert gesture.confidence == 0.95
        assert gesture.landmarks["wrist"]["x"] == 0.5
        assert gesture.timestamp is not None
        assert gesture.user_id == user.id

    async def test_gesture_relationship(self, in_memory_db: AsyncSession) -> None:
        user = User(email="grel@example.com", name="GRel", hashed_password="pw")
        gesture = GestureHistory(user=user, gesture_type="ok")
        in_memory_db.add_all([user, gesture])
        await in_memory_db.commit()
        await in_memory_db.refresh(gesture)

        assert gesture.user is not None
        assert gesture.user.email == "grel@example.com"


class TestCommunicationSessionModel:
    """Test CommunicationSession model."""

    async def test_create_session(self, in_memory_db: AsyncSession) -> None:
        user = User(email="session@example.com", name="Session", hashed_password="pw")
        session = CommunicationSession(
            user=user,
            session_type="multimodal",
            metadata={"device": "web"},
        )
        in_memory_db.add_all([user, session])
        await in_memory_db.commit()
        await in_memory_db.refresh(session)

        assert session.session_type == "multimodal"
        assert session.is_active is True
        assert session.start_time is not None
        assert session.end_time is None
        assert session.metadata["device"] == "web"

    async def test_session_defaults(self, in_memory_db: AsyncSession) -> None:
        user = User(email="sdef@example.com", name="SDef", hashed_password="pw")
        session = CommunicationSession(user=user, session_type="gesture")
        in_memory_db.add_all([user, session])
        await in_memory_db.commit()

        assert session.is_active is True
        assert session.start_time is not None


class TestEmotionalAnalyticsModel:
    """Test EmotionalAnalytics model."""

    async def test_create_emotional_analytics(self, in_memory_db: AsyncSession) -> None:
        user = User(email="emo@example.com", name="Emo", hashed_password="pw")
        ea = EmotionalAnalytics(
            user=user,
            emotion_type="happy",
            confidence=0.87,
            facial_data={"smile": 0.9},
            vocal_data={"pitch": 200},
        )
        in_memory_db.add_all([user, ea])
        await in_memory_db.commit()
        await in_memory_db.refresh(ea)

        assert ea.emotion_type == "happy"
        assert ea.confidence == 0.87
        assert ea.facial_data["smile"] == 0.9
        assert ea.timestamp is not None

    async def test_emotion_timestamp_indexed(self) -> None:
        mapper = inspect(EmotionalAnalytics)
        cols = {c.name: c for c in mapper.columns}
        assert cols["timestamp"].index is True


class TestPersonalizationMemoryModel:
    """Test PersonalizationMemory model."""

    async def test_create_personalization_memory(self, in_memory_db: AsyncSession) -> None:
        user = User(email="pmem@example.com", name="PMem", hashed_password="pw")
        pm = PersonalizationMemory(
            user=user,
            memory_type="preference",
            key="theme",
            value={"setting": "dark"},
        )
        in_memory_db.add_all([user, pm])
        await in_memory_db.commit()
        await in_memory_db.refresh(pm)

        assert pm.memory_type == "preference"
        assert pm.key == "theme"
        assert pm.value["setting"] == "dark"
        assert pm.created_at is not None

    async def test_unique_constraint(self, in_memory_db: AsyncSession) -> None:
        user = User(email="uniq@example.com", name="Uniq", hashed_password="pw")
        pm1 = PersonalizationMemory(user=user, memory_type="pref", key="k1", value={"v": 1})
        pm2 = PersonalizationMemory(user=user, memory_type="pref", key="k1", value={"v": 2})
        in_memory_db.add_all([user, pm1, pm2])
        with pytest.raises(Exception):
            await in_memory_db.commit()

    async def test_embedding_column(self) -> None:
        mapper = inspect(PersonalizationMemory)
        cols = {c.name: c for c in mapper.columns}
        assert "embedding" in cols


class TestPhrasePredictionModel:
    """Test PhrasePrediction model."""

    async def test_create_phrase_prediction(self, in_memory_db: AsyncSession) -> None:
        user = User(email="phrase@example.com", name="Phrase", hashed_password="pw")
        pp = PhrasePrediction(
            user=user,
            predicted_phrase="I need help",
            confidence=0.92,
            context={"activity": "eating"},
        )
        in_memory_db.add_all([user, pp])
        await in_memory_db.commit()
        await in_memory_db.refresh(pp)

        assert pp.predicted_phrase == "I need help"
        assert pp.confidence == 0.92
        assert pp.frequency == 1
        assert pp.last_used is not None

    async def test_frequency_increment(self, in_memory_db: AsyncSession) -> None:
        user = User(email="freq@example.com", name="Freq", hashed_password="pw")
        pp = PhrasePrediction(user=user, predicted_phrase="hello")
        in_memory_db.add_all([user, pp])
        await in_memory_db.commit()
        await in_memory_db.refresh(pp)

        pp.frequency += 1
        await in_memory_db.commit()
        assert pp.frequency == 2


class TestTranslationHistoryModel:
    """Test TranslationHistory model."""

    async def test_create_translation(self, in_memory_db: AsyncSession) -> None:
        user = User(email="trans@example.com", name="Trans", hashed_password="pw")
        th = TranslationHistory(
            user=user,
            source_text="Hello",
            target_text="Hola",
            source_lang="en",
            target_lang="es",
            confidence=0.98,
        )
        in_memory_db.add_all([user, th])
        await in_memory_db.commit()
        await in_memory_db.refresh(th)

        assert th.source_text == "Hello"
        assert th.target_text == "Hola"
        assert th.source_lang == "en"
        assert th.target_lang == "es"
        assert th.confidence == 0.98


class TestAdaptationMetricsModel:
    """Test AdaptationMetrics model."""

    async def test_create_adaptation_metric(self, in_memory_db: AsyncSession) -> None:
        user = User(email="adapt@example.com", name="Adapt", hashed_password="pw")
        am = AdaptationMetrics(
            user=user,
            metric_type="gesture_accuracy",
            value=0.85,
            metadata={"samples": 100},
        )
        in_memory_db.add_all([user, am])
        await in_memory_db.commit()
        await in_memory_db.refresh(am)

        assert am.metric_type == "gesture_accuracy"
        assert am.value == 0.85
        assert am.metadata["samples"] == 100
        assert am.recorded_at is not None


class TestModelRelationships:
    """Test cross-model relationships."""

    async def test_user_multiple_relations(self, in_memory_db: AsyncSession) -> None:
        user = User(email="multi@example.com", name="Multi", hashed_password="pw")
        GestureHistory(user=user, gesture_type="wave")
        CommunicationSession(user=user, session_type="speech")
        EmotionalAnalytics(user=user, emotion_type="neutral")
        PersonalizationMemory(user=user, memory_type="habit", key="morning", value={})
        PhrasePrediction(user=user, predicted_phrase="good morning")
        TranslationHistory(user=user, source_text="Hello", target_text="Bonjour",
                           source_lang="en", target_lang="fr")
        AdaptationMetrics(user=user, metric_type="accuracy", value=0.9)

        in_memory_db.add(user)
        await in_memory_db.commit()
        await in_memory_db.refresh(user)

        assert len(user.gesture_histories) == 1
        assert len(user.communication_sessions) == 1
        assert len(user.emotional_analytics) == 1
        assert len(user.personalization_memories) == 1
        assert len(user.phrase_predictions) == 1
        assert len(user.translation_histories) == 1
        assert len(user.adaptation_metrics) == 1
