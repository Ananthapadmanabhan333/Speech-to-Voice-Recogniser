from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from vector_memory.embeddings.embedding_generator import EmbeddingGenerator
from vector_memory.retrieval.semantic_retriever import RetrievedItem, RetrievalMetrics, SemanticRetriever
from vector_memory.store.memory_manager import MemoryManager, UserPatterns
from vector_memory.store.vector_store import VectorStore

logger = structlog.get_logger(__name__)


class ContextAssemblyError(Exception):
    """Raised when context assembly fails."""


@dataclass
class ActiveTopic:
    """An active conversation topic."""

    name: str
    relevance: float
    first_mentioned: datetime
    last_mentioned: datetime
    mention_count: int
    related_entities: List[str] = field(default_factory=list)


@dataclass
class CommunicationContext:
    """Complete communication context for LLM/multimodal input."""

    user_id: str
    current_input: str
    conversation_history: List[Dict[str, Any]]
    relevant_memories: List[RetrievedItem]
    user_preferences: Dict[str, Any]
    active_topics: List[ActiveTopic]
    user_patterns: Optional[UserPatterns]
    retrieval_metrics: Optional[RetrievalMetrics]
    formatted_prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    assembled_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to a serializable dict."""
        return {
            "user_id": self.user_id,
            "current_input": self.current_input,
            "conversation_history": self.conversation_history[-10:],
            "relevant_memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "score": m.final_score,
                    "modality": m.modality,
                    "source": m.source_collection,
                }
                for m in self.relevant_memories[:5]
            ],
            "user_preferences": self.user_preferences,
            "active_topics": [
                {
                    "name": t.name,
                    "relevance": t.relevance,
                    "mention_count": t.mention_count,
                }
                for t in self.active_topics[:5]
            ],
            "patterns": {
                "preferred_modality": self.user_patterns.preferred_modality if self.user_patterns else "text",
                "emotion_trend": self.user_patterns.emotion_trend if self.user_patterns else "stable",
            } if self.user_patterns else {},
            "formatted_prompt": self.formatted_prompt,
            "assembled_at": self.assembled_at,
        }


class ContextAssembler:
    """Assembles rich communication context from multiple memory sources.

    Combines:
    - Current user input
    - Relevant semantic/episodic/procedural memories
    - Conversation history
    - User preferences
    - Active topics
    - User behavioral patterns

    Outputs a formatted CommunicationContext ready for:
    - LLM prompt construction
    - Multimodal model input
    - Response generation
    - Personalization
    """

    MAX_HISTORY_LENGTH: int = 20
    MAX_MEMORIES: int = 10
    MAX_TOPICS: int = 10
    TOPIC_DECAY_HOURS: float = 48.0

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        semantic_retriever: SemanticRetriever,
        memory_manager: MemoryManager,
    ) -> None:
        """Initialize the context assembler.

        Args:
            vector_store: VectorStore instance.
            embedding_generator: EmbeddingGenerator instance.
            semantic_retriever: SemanticRetriever instance.
            memory_manager: MemoryManager instance.
        """
        self._vs = vector_store
        self._embedder = embedding_generator
        self._retriever = semantic_retriever
        self._memory_manager = memory_manager

        # In-memory conversation history and topics per user
        self._conversation_histories: Dict[str, List[Dict[str, Any]]] = {}
        self._active_topics: Dict[str, List[ActiveTopic]] = {}

        logger.info("context_assembler_initialized")

    def assemble_context(
        self,
        user_id: str,
        current_input: str,
        k: int = 10,
        include_patterns: bool = True,
        include_preferences: bool = True,
    ) -> CommunicationContext:
        """Assemble the full communication context for a user.

        Args:
            user_id: User identifier.
            current_input: Current user input text.
            k: Number of memories to retrieve.
            include_patterns: Include user behavioral patterns.
            include_preferences: Include user preferences.

        Returns:
            CommunicationContext with all assembled data.

        Raises:
            ContextAssemblyError: If assembly fails.
        """
        if not user_id:
            raise ValueError("user_id is required")
        if not current_input:
            raise ValueError("current_input cannot be empty")

        try:
            # 1. Retrieve relevant memories
            memories, retrieval_metrics = self._retriever.retrieve(
                query=current_input,
                user_id=user_id,
                k=k,
            )

            # 2. Build conversation history
            history = self._get_conversation_history(user_id, current_input)

            # 3. Extract user preferences
            preferences: Dict[str, Any] = {}
            if include_preferences:
                preferences = self._extract_preferences(user_id)

            # 4. Identify active topics
            topics = self._update_topics(user_id, current_input, memories)

            # 5. Get user patterns
            patterns: Optional[UserPatterns] = None
            if include_patterns:
                patterns = self._memory_manager.get_user_patterns(user_id)

            # 6. Format for LLM/multimodal input
            formatted = self._format_prompt(
                current_input=current_input,
                history=history,
                memories=memories,
                preferences=preferences,
                topics=topics,
                patterns=patterns,
            )

            return CommunicationContext(
                user_id=user_id,
                current_input=current_input,
                conversation_history=history,
                relevant_memories=memories,
                user_preferences=preferences,
                active_topics=topics,
                user_patterns=patterns,
                retrieval_metrics=retrieval_metrics,
                formatted_prompt=formatted,
                metadata={
                    "memories_retrieved": len(memories),
                    "history_length": len(history),
                    "topics_active": len(topics),
                },
            )

        except Exception as e:
            logger.error(
                "context_assembly_failed",
                user_id=user_id,
                error=str(e),
            )
            raise ContextAssemblyError(f"Context assembly failed: {e}") from e

    def add_to_history(
        self,
        user_id: str,
        entry: Dict[str, Any],
    ) -> None:
        """Add an entry to the conversation history.

        Args:
            user_id: User identifier.
            entry: History entry dict with 'role', 'content', 'timestamp', etc.
        """
        if user_id not in self._conversation_histories:
            self._conversation_histories[user_id] = []

        entry["timestamp"] = entry.get("timestamp", datetime.now(timezone.utc).isoformat())
        self._conversation_histories[user_id].append(entry)

        # Trim to max length
        if len(self._conversation_histories[user_id]) > self.MAX_HISTORY_LENGTH:
            self._conversation_histories[user_id] = \
                self._conversation_histories[user_id][-self.MAX_HISTORY_LENGTH:]

    def clear_user_context(self, user_id: str) -> None:
        """Clear all tracked context for a user.

        Args:
            user_id: User identifier.
        """
        self._conversation_histories.pop(user_id, None)
        self._active_topics.pop(user_id, None)
        logger.info("user_context_cleared", user_id=user_id)

    def _get_conversation_history(
        self,
        user_id: str,
        current_input: str,
    ) -> List[Dict[str, Any]]:
        """Get and update conversation history for a user.

        Args:
            user_id: User identifier.
            current_input: Current input to append.

        Returns:
            List of history entries.
        """
        if user_id not in self._conversation_histories:
            self._conversation_histories[user_id] = []

        # Add current input as latest user message
        now = datetime.now(timezone.utc).isoformat()

        # Avoid duplicates of the exact same input
        history = self._conversation_histories[user_id]
        if not history or history[-1].get("content") != current_input:
            self.add_to_history(user_id, {
                "role": "user",
                "content": current_input,
                "timestamp": now,
            })

        return self._conversation_histories.get(user_id, [])

    def _extract_preferences(self, user_id: str) -> Dict[str, Any]:
        """Extract user preferences from memory.

        Args:
            user_id: User identifier.

        Returns:
            Dict of preference key -> value.
        """
        preferences: Dict[str, Any] = {}

        try:
            # Search for preference entries
            query_emb = self._embedder.generate_text_embedding(
                "preference setting", use_cache=True
            )
            results = self._vs.search_similar(
                "user_preferences",
                query_emb.vector,
                k=20,
                filters={"user_id": user_id},
            )
            for r in results:
                key = r.metadata.get("preference_key", "")
                value = r.metadata.get("preference_value", "")
                if key:
                    preferences[key] = value
        except Exception as e:
            logger.warning("preference_extraction_failed", user_id=user_id, error=str(e))

        return preferences

    def _update_topics(
        self,
        user_id: str,
        current_input: str,
        memories: List[RetrievedItem],
    ) -> List[ActiveTopic]:
        """Update active topics based on current input and memories.

        Args:
            user_id: User identifier.
            current_input: Current user input.
            memories: Retrieved relevant memories.

        Returns:
            Updated list of active topics.
        """
        if user_id not in self._active_topics:
            self._active_topics[user_id] = []

        topics = self._active_topics[user_id]
        now = datetime.now(timezone.utc)

        # Decay old topics
        topics = [
            t for t in topics
            if (now - t.last_mentioned).total_seconds() / 3600 < self.TOPIC_DECAY_HOURS
        ]

        # Extract topic candidates from input
        input_lower = current_input.lower()
        topic_words = [
            w for w in input_lower.split()
            if len(w) > 3 and w not in self._STOP_WORDS
        ]

        # Extract topics from memories
        for mem in memories:
            for field_key in ["intent", "utterance_type", "session_type"]:
                val = mem.metadata.get(field_key, "")
                if val and val.lower() not in self._STOP_WORDS:
                    topic_words.append(val.lower())

        # Update topic tracking
        for word in topic_words:
            existing = next((t for t in topics if t.name.lower() == word.lower()), None)
            if existing:
                existing.relevance = min(1.0, existing.relevance + 0.1)
                existing.last_mentioned = now
                existing.mention_count += 1
            else:
                topics.append(ActiveTopic(
                    name=word,
                    relevance=0.3,
                    first_mentioned=now,
                    last_mentioned=now,
                    mention_count=1,
                ))

        # Sort by relevance and limit
        topics.sort(key=lambda t: t.relevance, reverse=True)
        self._active_topics[user_id] = topics[:self.MAX_TOPICS]

        return self._active_topics[user_id]

    def _format_prompt(
        self,
        current_input: str,
        history: List[Dict[str, Any]],
        memories: List[RetrievedItem],
        preferences: Dict[str, Any],
        topics: List[ActiveTopic],
        patterns: Optional[UserPatterns],
    ) -> str:
        """Format the assembled context into a prompt string.

        Constructs a structured prompt suitable for LLM or multimodal model input.

        Args:
            current_input: Current user input.
            history: Conversation history.
            memories: Relevant memories.
            preferences: User preferences.
            topics: Active topics.
            patterns: User patterns.

        Returns:
            Formatted prompt string.
        """
        parts: List[str] = []

        # System context
        parts.append("=== System Context ===")
        parts.append("You are a communication assistant for a multimodal AAC system.")
        parts.append("")

        # User profile
        if patterns:
            parts.append("=== User Profile ===")
            parts.append(f"Preferred modality: {patterns.preferred_modality}")
            parts.append(f"Emotion trend: {patterns.emotion_trend}")
            parts.append(f"Learning progress: {patterns.learning_progress:.1%}")
            parts.append("")

        # Active topics
        if topics:
            parts.append("=== Active Topics ===")
            for t in topics[:5]:
                parts.append(f"- {t.name} (relevance: {t.relevance:.2f}, mentioned {t.mention_count}x)")
            parts.append("")

        # Conversation history
        if history:
            parts.append("=== Recent Conversation ===")
            for entry in history[-6:]:
                role = entry.get("role", "unknown")
                content = entry.get("content", "")
                if content:
                    parts.append(f"{role}: {content}")
            parts.append("")

        # Relevant memories
        if memories:
            parts.append("=== Relevant Memories ===")
            for mem in memories[:5]:
                parts.append(f"- [{mem.modality}] {mem.content} (confidence: {mem.final_score:.2f})")
            parts.append("")

        # User preferences
        if preferences:
            parts.append("=== User Preferences ===")
            for k, v in preferences.items():
                parts.append(f"- {k}: {v}")
            parts.append("")

        # Current input
        parts.append("=== Current Input ===")
        parts.append(f"user: {current_input}")
        parts.append("")

        # Assistant prefix
        parts.append("=== Response ===")
        parts.append("assistant: ")

        return "\n".join(parts)

    _STOP_WORDS: set = {
        "this", "that", "with", "from", "have", "been", "were",
        "what", "when", "where", "which", "their", "there",
        "about", "would", "could", "should", "because", "just",
        "very", "also", "some", "them", "then", "than", "into",
        "over", "after", "other", "more", "such", "only", "even",
        "still", "back", "here", "well", "much", "many", "down",
        "your", "make", "like", "know", "take", "come", "look",
        "want", "tell", "think", "need", "feel", "give", "use",
        "find", "ask", "work", "seem", "keep", "try", "call",
        "done", "said", "got", "set", "put", "let", "begin",
        "show", "hear", "play", "run", "move", "live", "believe",
        "hold", "bring", "happen", "write", "provide", "suggest",
    }

    def to_llm_messages(
        self,
        context: CommunicationContext,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Convert context to OpenAI-compatible chat messages.

        Args:
            context: Assembled communication context.
            system_prompt: Optional custom system prompt.

        Returns:
            List of message dicts with 'role' and 'content'.
        """
        messages: List[Dict[str, str]] = []

        # System message
        sys = system_prompt or (
            "You are a multimodal communication assistant for AAC. "
            "Use the provided context, memories, and user preferences "
            "to generate appropriate responses."
        )
        messages.append({"role": "system", "content": sys})

        # Conversation history
        for entry in context.conversation_history:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            if content:
                messages.append({"role": role, "content": content})

        # Current input
        if context.current_input:
            messages.append({"role": "user", "content": context.current_input})

        return messages
