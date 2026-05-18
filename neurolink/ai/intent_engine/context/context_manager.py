from __future__ import annotations

import json
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

logger = structlog.get_logger(__name__)


class ContextError(Exception):
    """Raised when context management fails."""


@dataclass
class Utterance:
    """A single utterance in the conversation."""

    text: str
    intent: str
    emotion: str
    timestamp: float
    session_id: str
    speaker: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Topic:
    """A tracked conversation topic."""

    name: str
    confidence: float
    first_mentioned: float
    last_mentioned: float
    mention_count: int = 1
    keywords: Set[str] = field(default_factory=set)

    def update(self, confidence: float) -> None:
        self.confidence = max(self.confidence, confidence)
        self.last_mentioned = time.time()
        self.mention_count += 1


@dataclass
class Context:
    """Full conversation context for a session."""

    session_id: str
    short_term: List[Utterance]
    long_term: Dict[str, Any]
    current_topic: Optional[Topic] = None
    topics: List[Topic] = field(default_factory=list)
    active_intents: List[str] = field(default_factory=list)
    turn_count: int = 0
    last_active: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextManager:
    """Communication context management with short-term and long-term memory.

    Maintains per-session conversation context including:
    - Short-term memory: recent utterances (last N)
    - Long-term memory: user patterns and preferences
    - Topic tracking with decay
    - Contextual disambiguation
    - Active intent tracking

    Context persistence is configurable for memory efficiency.
    """

    SHORT_TERM_SIZE: int = 20  # Recent utterances to keep
    TOPIC_DECAY_SECONDS: float = 300.0  # 5 minutes
    SESSION_TIMEOUT: float = 1800.0  # 30 minutes
    MAX_TOPICS: int = 10

    def __init__(
        self,
        short_term_size: int = 20,
        topic_decay: float = 300.0,
        session_timeout: float = 1800.0,
        persistence_path: Optional[Path] = None,
    ):
        """Initialize context manager.

        Args:
            short_term_size: Max recent utterances to keep.
            topic_decay: Seconds before topic confidence decays.
            session_timeout: Seconds before session is considered stale.
            persistence_path: Path to persist long-term memories.
        """
        self.SHORT_TERM_SIZE = short_term_size
        self.TOPIC_DECAY_SECONDS = topic_decay
        self.SESSION_TIMEOUT = session_timeout
        self._persistence_path = persistence_path

        # Active sessions
        self._sessions: Dict[str, Context] = {}

        # Long-term user profiles (session_id -> profile data)
        self._long_term_memory: Dict[str, Dict[str, Any]] = {}

        # Load persisted long-term memory
        if persistence_path and persistence_path.exists():
            self._load_long_term_memory()

        logger.info(
            "context_manager_initialized",
            short_term_size=short_term_size,
            topic_decay=topic_decay,
        )

    def maintain_conversation_context(
        self,
        session_id: str,
        utterance_text: str,
        intent: str = "unknown",
        emotion: str = "neutral",
        speaker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Context:
        """Update conversation context with a new utterance.

        Creates session if it doesn't exist, updates short-term memory,
        tracks topics, and maintains long-term patterns.

        Args:
            session_id: Unique session identifier.
            utterance_text: The current utterance text.
            intent: Classified intent for this utterance.
            emotion: Detected emotion for this utterance.
            speaker: Optional speaker identifier.
            metadata: Additional metadata for this utterance.

        Returns:
            Updated Context for the session.

        Raises:
            ContextError: If context maintenance fails.
        """
        if not session_id:
            raise ValueError("session_id is required")
        if not utterance_text:
            raise ValueError("utterance_text is required")

        try:
            # Get or create session context
            if session_id not in self._sessions:
                self._create_session(session_id)

            context = self._sessions[session_id]

            # Check session timeout
            if self._is_session_stale(context):
                self._archive_session(session_id)
                self._create_session(session_id)
                context = self._sessions[session_id]

            # Create utterance
            utterance = Utterance(
                text=utterance_text,
                intent=intent,
                emotion=emotion,
                timestamp=time.time(),
                session_id=session_id,
                speaker=speaker,
                metadata=metadata or {},
            )

            # Update short-term memory
            context.short_term.append(utterance)
            if len(context.short_term) > self.SHORT_TERM_SIZE:
                context.short_term = context.short_term[-self.SHORT_TERM_SIZE:]

            # Update turn count
            context.turn_count += 1
            context.last_active = time.time()

            # Track topics
            self._update_topics(context, utterance_text, utterance)

            # Track active intents
            if intent not in context.active_intents:
                context.active_intents.append(intent)
            # Keep only last 5 intents
            if len(context.active_intents) > 5:
                context.active_intents = context.active_intents[-5:]

            # Update long-term memory
            self._update_long_term(session_id, utterance)

            return context

        except Exception as e:
            logger.error("context_maintenance_failed", session_id=session_id, error=str(e))
            raise ContextError(f"Context maintenance failed: {e}") from e

    def get_context(self, session_id: str) -> Optional[Context]:
        """Get the current context for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Context if session exists, None otherwise.
        """
        return self._sessions.get(session_id)

    def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a human-readable summary of the conversation context.

        Args:
            session_id: Session identifier.

        Returns:
            Dict with context summary.
        """
        context = self.get_context(session_id)
        if not context:
            return {"session_id": session_id, "active": False}

        recent_texts = [u.text for u in context.short_term[-5:]]
        return {
            "session_id": session_id,
            "active": True,
            "turn_count": context.turn_count,
            "current_topic": context.current_topic.name if context.current_topic else None,
            "active_intents": context.active_intents,
            "recent_utterances": recent_texts,
            "topics": [t.name for t in context.topics],
            "last_active": context.last_active,
        }

    def disambiguate(
        self, session_id: str, ambiguous_text: str, candidates: List[str]
    ) -> List[Tuple[str, float]]:
        """Disambiguate an ambiguous utterance using context.

        Uses recent topics and intents to score interpretation candidates.

        Args:
            session_id: Session identifier.
            ambiguous_text: Ambiguous input text.
            candidates: List of possible interpretations.

        Returns:
            List of (candidate, score) sorted by relevance.
        """
        context = self.get_context(session_id)
        if not context:
            return [(c, 0.5) for c in candidates]

        scored: List[Tuple[str, float]] = []
        current_topic = context.current_topic

        for candidate in candidates:
            score = 0.5  # Base score

            # Boost if candidate matches current topic
            if current_topic and any(
                kw in candidate.lower() for kw in current_topic.keywords
            ):
                score += 0.3

            # Boost if candidate matches recent intents
            if any(intent in candidate.lower() for intent in context.active_intents):
                score += 0.2

            scored.append((candidate, min(1.0, score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def close_session(self, session_id: str) -> None:
        """Archive and close a session.

        Args:
            session_id: Session to close.
        """
        if session_id in self._sessions:
            self._archive_session(session_id)
            logger.info("session_closed", session_id=session_id)

    def cleanup_stale_sessions(self, max_age: Optional[float] = None) -> int:
        """Remove sessions that have been inactive beyond timeout.

        Args:
            max_age: Maximum session age in seconds. Defaults to SESSION_TIMEOUT.

        Returns:
            Number of cleaned-up sessions.
        """
        timeout = max_age or self.SESSION_TIMEOUT
        now = time.time()
        stale_ids = [
            sid
            for sid, ctx in self._sessions.items()
            if (now - ctx.last_active) > timeout
        ]

        for sid in stale_ids:
            self._archive_session(sid)

        if stale_ids:
            logger.info("stale_sessions_cleaned", count=len(stale_ids))

        return len(stale_ids)

    def get_long_term_memory(self, session_id: str) -> Dict[str, Any]:
        """Get long-term memory for a user/session.

        Args:
            session_id: Session or user identifier.

        Returns:
            Long-term memory dict.
        """
        return self._long_term_memory.get(session_id, {})

    def persist(self) -> None:
        """Persist long-term memory to disk."""
        if not self._persistence_path:
            return

        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "long_term_memory": self._long_term_memory,
            "saved_at": time.time(),
        }
        with open(self._persistence_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("context_persisted", path=str(self._persistence_path))

    def _create_session(self, session_id: str) -> Context:
        """Create a new session context.

        Args:
            session_id: Session identifier.
        """
        context = Context(
            session_id=session_id,
            short_term=[],
            long_term=self._long_term_memory.get(session_id, {}),
        )
        self._sessions[session_id] = context

        logger.debug("session_created", session_id=session_id)
        return context

    def _is_session_stale(self, context: Context) -> bool:
        """Check if a session has timed out.

        Args:
            context: Session context.

        Returns:
            True if session has timed out.
        """
        return (time.time() - context.last_active) > self.SESSION_TIMEOUT

    def _archive_session(self, session_id: str) -> None:
        """Archive session data to long-term memory.

        Args:
            session_id: Session to archive.
        """
        context = self._sessions.get(session_id)
        if not context:
            return

        # Summarize session for long-term memory
        summary = {
            "last_active": context.last_active,
            "total_turns": context.turn_count,
            "topics": [t.name for t in context.topics],
            "common_intents": list(set(context.active_intents)),
            "utterance_count": len(context.short_term),
        }

        # Merge into long-term memory
        if session_id not in self._long_term_memory:
            self._long_term_memory[session_id] = {}

        ltm = self._long_term_memory[session_id]
        ltm["sessions"] = ltm.get("sessions", [])
        ltm["sessions"].append(summary)
        ltm["last_updated"] = time.time()

        # Remove from active sessions
        self._sessions.pop(session_id, None)

        logger.debug("session_archived", session_id=session_id)

    def _update_topics(self, context: Context, text: str, utterance: Utterance) -> None:
        """Update topic tracking with new utterance.

        Extracts keywords and updates or creates topics.

        Args:
            context: Session context.
            text: Utterance text.
            utterance: Utterance object.
        """
        # Extract potential keywords (simple approach)
        keywords = set(
            word.lower().strip(".,!?;:")
            for word in text.split()
            if len(word) > 3 and word[0].isalpha()
        )

        if not keywords:
            return

        # Check if utterance matches existing topics
        best_match = None
        best_score = 0.0

        for topic in context.topics:
            overlap = keywords & topic.keywords
            if overlap:
                score = len(overlap) / len(keywords)
                if score > best_score:
                    best_score = score
                    best_match = topic

        if best_match and best_score > 0.3:
            # Update existing topic
            best_match.update(best_score)
            best_match.keywords.update(keywords)
            context.current_topic = best_match
        else:
            # Create new topic
            new_topic = Topic(
                name=" ".join(sorted(keywords)[:3]),
                confidence=0.5,
                first_mentioned=utterance.timestamp,
                last_mentioned=utterance.timestamp,
                keywords=keywords,
            )
            context.topics.append(new_topic)
            context.current_topic = new_topic

            # Limit number of topics
            if len(context.topics) > self.MAX_TOPICS:
                # Remove oldest topic
                context.topics.sort(key=lambda t: t.last_mentioned)
                context.topics = context.topics[-self.MAX_TOPICS:]

    def _update_long_term(self, session_id: str, utterance: Utterance) -> None:
        """Update long-term memory with utterance patterns.

        Args:
            session_id: Session identifier.
            utterance: Current utterance.
        """
        if session_id not in self._long_term_memory:
            self._long_term_memory[session_id] = {
                "first_seen": time.time(),
                "utterance_count": 0,
                "intent_frequency": defaultdict(int),
                "topics_of_interest": [],
            }

        ltm = self._long_term_memory[session_id]
        ltm["utterance_count"] = ltm.get("utterance_count", 0) + 1
        ltm["intent_frequency"][utterance.intent] = ltm["intent_frequency"].get(utterance.intent, 0) + 1

    def _load_long_term_memory(self) -> None:
        """Load long-term memory from persistence file."""
        try:
            with open(self._persistence_path) as f:
                data = json.load(f)
            self._long_term_memory = data.get("long_term_memory", {})
            logger.info(
                "long_term_memory_loaded",
                sessions=len(self._long_term_memory),
                path=str(self._persistence_path),
            )
        except Exception as e:
            logger.warning("failed_to_load_long_term_memory", error=str(e))
