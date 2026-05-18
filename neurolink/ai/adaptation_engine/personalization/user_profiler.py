from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class UserProfilerError(Exception):
    """Raised when user profiling fails."""


@dataclass
class AccessibilityPreferences:
    """User accessibility preferences."""

    visual_aid: bool = False  # Need visual cues
    hearing_aid: bool = False  # Need hearing assistance
    motor_impairment: bool = False  # Fine motor skill limitations
    cognitive_assistance: bool = False  # Need simplified interface
    text_size: str = "normal"  # "small", "normal", "large"
    contrast_mode: str = "normal"  # "normal", "high"
    response_speed: str = "normal"  # "slow", "normal", "fast"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "visual_aid": self.visual_aid,
            "hearing_aid": self.hearing_aid,
            "motor_impairment": self.motor_impairment,
            "cognitive_assistance": self.cognitive_assistance,
            "text_size": self.text_size,
            "contrast_mode": self.contrast_mode,
            "response_speed": self.response_speed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AccessibilityPreferences":
        return cls(
            visual_aid=data.get("visual_aid", False),
            hearing_aid=data.get("hearing_aid", False),
            motor_impairment=data.get("motor_impairment", False),
            cognitive_assistance=data.get("cognitive_assistance", False),
            text_size=data.get("text_size", "normal"),
            contrast_mode=data.get("contrast_mode", "normal"),
            response_speed=data.get("response_speed", "normal"),
        )


@dataclass
class UserProfile:
    """Complete user profile with learned patterns and preferences."""

    user_id: str
    name: str = ""
    gestural_vocabulary: Dict[str, int] = field(default_factory=dict)  # gesture -> frequency
    speech_patterns: Dict[str, int] = field(default_factory=dict)  # phrase -> frequency
    common_intents: Dict[str, float] = field(default_factory=dict)  # intent -> probability
    preferred_gestures: List[str] = field(default_factory=list)
    communication_speed: float = 1.0  # Relative speed factor
    emotional_baseline: str = "neutral"
    accessibility: AccessibilityPreferences = field(default_factory=AccessibilityPreferences)
    adaptive_vocabulary: Set[str] = field(default_factory=set)
    language_preference: str = "en"
    last_active: float = field(default_factory=time.time)
    session_count: int = 0
    total_interactions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "gestural_vocabulary": self.gestural_vocabulary,
            "speech_patterns": self.speech_patterns,
            "common_intents": self.common_intents,
            "preferred_gestures": self.preferred_gestures,
            "communication_speed": self.communication_speed,
            "emotional_baseline": self.emotional_baseline,
            "accessibility": self.accessibility.to_dict(),
            "adaptive_vocabulary": list(self.adaptive_vocabulary),
            "language_preference": self.language_preference,
            "last_active": self.last_active,
            "session_count": self.session_count,
            "total_interactions": self.total_interactions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        profile = cls(user_id=data["user_id"])
        profile.name = data.get("name", "")
        profile.gestural_vocabulary = data.get("gestural_vocabulary", {})
        profile.speech_patterns = data.get("speech_patterns", {})
        profile.common_intents = data.get("common_intents", {})
        profile.preferred_gestures = data.get("preferred_gestures", [])
        profile.communication_speed = data.get("communication_speed", 1.0)
        profile.emotional_baseline = data.get("emotional_baseline", "neutral")
        profile.accessibility = AccessibilityPreferences.from_dict(
            data.get("accessibility", {})
        )
        profile.adaptive_vocabulary = set(data.get("adaptive_vocabulary", []))
        profile.language_preference = data.get("language_preference", "en")
        profile.last_active = data.get("last_active", time.time())
        profile.session_count = data.get("session_count", 0)
        profile.total_interactions = data.get("total_interactions", 0)
        profile.metadata = data.get("metadata", {})
        return profile


class UserProfiler:
    """User personalization engine that builds and maintains user profiles.

    Learns from user interactions to build comprehensive profiles including:
    - Gesture preferences and vocabulary
    - Communication patterns and speed
    - Common intents and phrases
    - Accessibility requirements
    - Emotional baselines

    Profiles are persisted to disk for long-term learning.
    """

    PROFILES_DIR: str = "user_profiles"

    def __init__(
        self,
        profiles_dir: Optional[Path] = None,
        auto_save_interval: int = 300,  # 5 minutes
    ):
        """Initialize user profiler.

        Args:
            profiles_dir: Directory to store user profiles.
            auto_save_interval: Auto-save interval in seconds.
        """
        self._profiles_dir = profiles_dir or Path(self.PROFILES_DIR)
        self._profiles_dir.mkdir(parents=True, exist_ok=True)

        self._auto_save_interval = auto_save_interval
        self._last_save_time: float = time.time()

        # In-memory cache of loaded profiles
        self._profiles: Dict[str, UserProfile] = {}

        # Learning state
        self._interaction_buffer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        logger.info(
            "user_profiler_initialized",
            profiles_dir=str(self._profiles_dir),
            auto_save_interval=auto_save_interval,
        )

    def build_user_profile(self, user_id: str) -> UserProfile:
        """Build or retrieve a user profile.

        Loads existing profile from disk or creates a new one.

        Args:
            user_id: Unique user identifier.

        Returns:
            UserProfile for the given user.

        Raises:
            UserProfilerError: If profile creation fails.
        """
        if user_id in self._profiles:
            return self._profiles[user_id]

        try:
            # Try to load from disk
            profile_path = self._get_profile_path(user_id)
            if profile_path.exists():
                with open(profile_path) as f:
                    data = json.load(f)
                profile = UserProfile.from_dict(data)
                logger.info("user_profile_loaded", user_id=user_id, path=str(profile_path))
            else:
                # Create new profile
                profile = UserProfile(user_id=user_id)
                logger.info("user_profile_created", user_id=user_id)

            self._profiles[user_id] = profile
            return profile

        except Exception as e:
            logger.error("failed_to_build_profile", user_id=user_id, error=str(e))
            raise UserProfilerError(f"Failed to build profile for {user_id}: {e}") from e

    def update_from_interaction(
        self,
        user_id: str,
        gesture_label: Optional[str] = None,
        speech_text: Optional[str] = None,
        intent: Optional[str] = None,
        emotion: Optional[str] = None,
        success: bool = True,
        **kwargs: Any,
    ) -> UserProfile:
        """Update user profile based on a single interaction.

        Args:
            user_id: User identifier.
            gesture_label: Recognized gesture label.
            speech_text: Transcribed speech text.
            intent: Classified intent.
            emotion: Detected emotion.
            success: Whether the interaction was successful.
            **kwargs: Additional interaction data.

        Returns:
            Updated UserProfile.
        """
        profile = self.build_user_profile(user_id)

        # Update gesture vocabulary
        if gesture_label:
            profile.gestural_vocabulary[gesture_label] = (
                profile.gestural_vocabulary.get(gesture_label, 0) + 1
            )
            if gesture_label not in profile.preferred_gestures:
                profile.preferred_gestures.append(gesture_label)

        # Update speech patterns
        if speech_text:
            normalized = speech_text.lower().strip()
            profile.speech_patterns[normalized] = (
                profile.speech_patterns.get(normalized, 0) + 1
            )

        # Update intents
        if intent:
            profile.common_intents[intent] = (
                profile.common_intents.get(intent, 0.0) + 0.1
            )

        # Update emotional baseline
        if emotion and profile.total_interactions > 0:
            # Exponential moving average of emotional baseline
            profile.emotional_baseline = emotion

        # Update communication speed based on interaction timing
        if "processing_time" in kwargs:
            # Normalize: faster processing = faster communication speed
            pt = kwargs["processing_time"]
            target_speed = min(2.0, max(0.5, 1.0 / (pt + 0.1)))
            profile.communication_speed = 0.9 * profile.communication_speed + 0.1 * target_speed

        # Update adaptive vocabulary
        if speech_text:
            words = set(speech_text.lower().split())
            profile.adaptive_vocabulary.update(words)

        # Update session tracking
        profile.total_interactions += 1
        profile.last_active = time.time()

        # Buffer interaction for batch processing
        self._interaction_buffer[user_id].append({
            "gesture_label": gesture_label,
            "speech_text": speech_text,
            "intent": intent,
            "emotion": emotion,
            "success": success,
            "timestamp": time.time(),
            **kwargs,
        })

        # Auto-save
        if time.time() - self._last_save_time > self._auto_save_interval:
            self._save_all_profiles()

        return profile

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get a user's profile.

        Args:
            user_id: User identifier.

        Returns:
            UserProfile if found, None otherwise.
        """
        return self._profiles.get(user_id) or (
            self.build_user_profile(user_id)
            if self._get_profile_path(user_id).exists()
            else None
        )

    def set_accessibility_preferences(
        self,
        user_id: str,
        preferences: AccessibilityPreferences,
    ) -> UserProfile:
        """Set accessibility preferences for a user.

        Args:
            user_id: User identifier.
            preferences: Accessibility preferences.

        Returns:
            Updated UserProfile.
        """
        profile = self.build_user_profile(user_id)
        profile.accessibility = preferences
        self._save_profile(user_id)
        logger.info("accessibility_preferences_updated", user_id=user_id, preferences=preferences.to_dict())
        return profile

    def save_profile(self, user_id: str) -> None:
        """Save a single user profile to disk.

        Args:
            user_id: User identifier.
        """
        self._save_profile(user_id)

    def save_all(self) -> None:
        """Save all cached profiles to disk."""
        self._save_all_profiles()

    def delete_profile(self, user_id: str) -> None:
        """Delete a user profile.

        Args:
            user_id: User identifier.
        """
        self._profiles.pop(user_id, None)
        profile_path = self._get_profile_path(user_id)
        if profile_path.exists():
            profile_path.unlink()
        logger.info("user_profile_deleted", user_id=user_id)

    def get_all_user_ids(self) -> List[str]:
        """Get all known user IDs.

        Returns:
            List of user IDs.
        """
        # From disk
        user_ids = set()
        for f in self._profiles_dir.glob("*.json"):
            user_ids.add(f.stem)
        # From cache
        user_ids.update(self._profiles.keys())
        return sorted(user_ids)

    def _save_profile(self, user_id: str) -> None:
        """Save a single profile to disk.

        Args:
            user_id: User identifier.
        """
        profile = self._profiles.get(user_id)
        if not profile:
            return

        try:
            profile_path = self._get_profile_path(user_id)
            with open(profile_path, "w") as f:
                json.dump(profile.to_dict(), f, indent=2)
            self._last_save_time = time.time()
        except Exception as e:
            logger.error("failed_to_save_profile", user_id=user_id, error=str(e))

    def _save_all_profiles(self) -> None:
        """Save all cached profiles to disk."""
        for user_id in list(self._profiles.keys()):
            self._save_profile(user_id)
        logger.debug("all_profiles_saved", count=len(self._profiles))

    def _get_profile_path(self, user_id: str) -> Path:
        """Get the file path for a user's profile.

        Args:
            user_id: User identifier.

        Returns:
            Path to the profile JSON file.
        """
        safe_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in user_id)
        return self._profiles_dir / f"{safe_id}.json"
