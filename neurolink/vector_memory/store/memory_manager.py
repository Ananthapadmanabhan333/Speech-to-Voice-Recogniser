from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import structlog

from vector_memory.embeddings.embedding_generator import EmbeddingGenerator
from vector_memory.store.vector_store import SearchResult, VectorStore

logger = structlog.get_logger(__name__)


class MemoryManagerError(Exception):
    """Raised when memory management operations fail."""


@dataclass
class MemoryItem:
    """A single memory item retrieved from the vector store."""

    id: str
    content: str
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]
    score: float
    memory_type: str  # "semantic", "episodic", "procedural"
    timestamp: datetime
    age_hours: float

    @property
    def is_recent(self) -> bool:
        return self.age_hours < 24.0


@dataclass
class UserPatterns:
    """Aggregated user patterns extracted from memory."""

    user_id: str
    frequent_gestures: List[Dict[str, Any]]
    common_intents: List[Dict[str, Any]]
    preferred_modality: str
    average_confidence: float
    emotion_trend: str
    recent_topics: List[str]
    peak_usage_hours: List[int]
    learning_progress: float
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CommunicationSessionData:
    """Data for a communication session to be stored."""

    user_id: str
    session_type: str
    utterances: List[Dict[str, Any]]
    gestures: List[Dict[str, Any]]
    emotions: List[Dict[str, Any]]
    intent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryManager:
    """Manages the full memory lifecycle including storage, retrieval,
    consolidation, and forgetting curves.

    Implements three memory types:
    - Semantic memory: Long-term factual knowledge about the user
    - Episodic memory: Recent interaction history
    - Procedural memory: User habits and behavioral patterns

    Features:
    - Forgetting curve management with decay-based scoring
    - Automatic consolidation from episodic to semantic memory
    - User preference extraction and storage
    - Pattern recognition across sessions
    - Configurable retention policies
    """

    # Forgetting curve parameters (Ebbinghaus-based)
    FORGETTING_DECAY_RATE: float = 0.5  # per day
    FORGETTING_REPETITION_BOOST: float = 1.5
    EPISODIC_TTL_DAYS: float = 7.0
    SEMANTIC_TTL_DAYS: float = 365.0
    CONSOLIDATION_INTERVAL: int = 50  # episodes before consolidation

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        semantic_collection: str = "semantic_memory",
        episodic_collection: str = "episodic_memory",
        procedural_collection: str = "procedural_memory",
        preferences_collection: str = "user_preferences",
    ) -> None:
        """Initialize the memory manager.

        Args:
            vector_store: VectorStore instance for persistence.
            embedding_generator: EmbeddingGenerator for creating embeddings.
            semantic_collection: Collection name for semantic memory.
            episodic_collection: Collection name for episodic memory.
            procedural_collection: Collection name for procedural memory.
            preferences_collection: Collection name for user preferences.
        """
        self._vs = vector_store
        self._embedder = embedding_generator

        self._semantic_collection = semantic_collection
        self._episodic_collection = episodic_collection
        self._procedural_collection = procedural_collection
        self._preferences_collection = preferences_collection

        # Ensure collections exist
        for col in [semantic_collection, episodic_collection, procedural_collection, preferences_collection]:
            self._vs.create_collection(col)

        # In-memory session tracking for consolidation
        self._session_counters: Dict[str, int] = {}
        self._recent_episodes: Dict[str, List[Dict[str, Any]]] = {}

        logger.info("memory_manager_initialized")

    def store_communication_session(
        self,
        session_data: CommunicationSessionData,
    ) -> str:
        """Store a complete communication session into memory.

        Args:
            session_data: Session data with utterances, gestures, emotions.

        Returns:
            Memory ID for the stored session.

        Raises:
            MemoryManagerError: If storage fails.
        """
        session_id = str(uuid.uuid4())
        user_id = session_data.user_id
        timestamp = datetime.now(timezone.utc)

        try:
            # Store as episodic memory
            for utterance in session_data.utterances:
                text = utterance.get("text", "")
                if not text:
                    continue

                embedding = self._embedder.generate_text_embedding(text, use_cache=True)
                metadata = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "type": "episodic",
                    "utterance_type": utterance.get("type", "user"),
                    "session_type": session_data.session_type,
                    "timestamp": timestamp.isoformat(),
                    "intent": session_data.intent or "",
                    "emotion": utterance.get("emotion", ""),
                    "gesture": utterance.get("gesture", ""),
                }
                self._vs.store_embedding(
                    self._episodic_collection,
                    embedding.vector,
                    metadata=metadata,
                )

            # Store gesture patterns as procedural memory
            for gesture_data in session_data.gestures:
                gest_text = gesture_data.get("type", "unknown")
                gest_emb = self._embedder.generate_text_embedding(gest_text, use_cache=True)
                gest_metadata = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "type": "procedural",
                    "gesture_type": gesture_data.get("type", ""),
                    "confidence": gesture_data.get("confidence", 0.0),
                    "timestamp": timestamp.isoformat(),
                    "repetitions": gesture_data.get("repetitions", 1),
                }
                self._vs.store_embedding(
                    self._procedural_collection,
                    gest_emb.vector,
                    metadata=gest_metadata,
                )

            # Track for consolidation
            counter = self._session_counters.get(user_id, 0) + 1
            self._session_counters[user_id] = counter

            if user_id not in self._recent_episodes:
                self._recent_episodes[user_id] = []
            self._recent_episodes[user_id].append({
                "id": session_id,
                "data": session_data,
                "timestamp": timestamp,
            })

            # Trigger consolidation if threshold reached
            if counter % self.CONSOLIDATION_INTERVAL == 0:
                self._consolidate_memories(user_id)

            logger.info(
                "session_stored",
                user_id=user_id,
                session_id=session_id,
                utterances=len(session_data.utterances),
                gestures=len(session_data.gestures),
            )
            return session_id

        except Exception as e:
            logger.error("session_store_failed", user_id=user_id, error=str(e))
            raise MemoryManagerError(f"Failed to store session: {e}") from e

    def retrieve_relevant_context(
        self,
        user_id: str,
        query: str,
        k: int = 10,
    ) -> List[MemoryItem]:
        """Retrieve relevant memories across all memory types.

        Args:
            user_id: User identifier.
            query: Query string for semantic search.
            k: Maximum results to return.

        Returns:
            List of MemoryItem sorted by relevance score.
        """
        if not query or not query.strip():
            return []

        try:
            query_emb = self._embedder.generate_text_embedding(query, use_cache=True)

            all_results: List[MemoryItem] = []

            # Search each memory type
            for collection, memory_type in [
                (self._episodic_collection, "episodic"),
                (self._semantic_collection, "semantic"),
                (self._procedural_collection, "procedural"),
            ]:
                user_filter = {"user_id": user_id} if user_id else None
                results = self._vs.search_similar(
                    collection,
                    query_emb.vector,
                    k=k // 3 + 1,
                    filters=user_filter,
                )

                for r in results:
                    mem_item = self._result_to_memory_item(r, memory_type)
                    if mem_item:
                        all_results.append(mem_item)

            # Apply forgetting curve scoring
            for item in all_results:
                item.score *= self._forgetting_curve_factor(item)

            # Sort by adjusted score and take top k
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:k]

        except Exception as e:
            logger.error("context_retrieval_failed", user_id=user_id, error=str(e))
            return []

    def store_user_preference(
        self,
        user_id: str,
        key: str,
        value: Any,
    ) -> str:
        """Store a user preference in memory.

        Args:
            user_id: User identifier.
            key: Preference key.
            value: Preference value (JSON-serializable).

        Returns:
            Preference ID.

        Raises:
            MemoryManagerError: If storage fails.
        """
        pref_id = str(uuid.uuid4())

        try:
            embedding = self._embedder.generate_text_embedding(f"{key}: {value}", use_cache=True)
            metadata = {
                "user_id": user_id,
                "preference_key": key,
                "preference_value": str(value),
                "type": "preference",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._vs.store_embedding(
                self._preferences_collection,
                embedding.vector,
                metadata=metadata,
                id=pref_id,
            )
            logger.info("preference_stored", user_id=user_id, key=key)
            return pref_id

        except Exception as e:
            logger.error("preference_store_failed", user_id=user_id, key=key, error=str(e))
            raise MemoryManagerError(f"Failed to store preference: {e}") from e

    def get_user_preference(
        self,
        user_id: str,
        key: str,
    ) -> Optional[Any]:
        """Retrieve a specific user preference.

        Args:
            user_id: User identifier.
            key: Preference key.

        Returns:
            Preference value or None.
        """
        try:
            results = self._vs.search_similar(
                self._preferences_collection,
                self._embedder.generate_text_embedding(key, use_cache=True).vector,
                k=1,
                filters={"user_id": user_id, "preference_key": key},
            )
            if results:
                return results[0].metadata.get("preference_value")
        except Exception:
            pass
        return None

    def get_user_patterns(
        self,
        user_id: str,
    ) -> UserPatterns:
        """Extract behavioral patterns for a user.

        Args:
            user_id: User identifier.

        Returns:
            UserPatterns with aggregated statistics.
        """
        try:
            # Query procedural memory for gesture patterns
            gesture_query = self._embedder.generate_text_embedding(
                "gesture pattern habit", use_cache=True
            )
            gesture_results = self._vs.search_similar(
                self._procedural_collection,
                gesture_query.vector,
                k=20,
                filters={"user_id": user_id},
            )

            frequent_gestures: Dict[str, int] = {}
            for r in gesture_results:
                gtype = r.metadata.get("gesture_type", "unknown")
                frequent_gestures[gtype] = frequent_gestures.get(gtype, 0) + 1

            # Query semantic for intents
            intent_results = self._vs.search_similar(
                self._semantic_collection,
                self._embedder.generate_text_embedding("intent goal", use_cache=True).vector,
                k=20,
                filters={"user_id": user_id},
            )

            common_intents: Dict[str, float] = {}
            for r in intent_results:
                intent = r.metadata.get("intent", "unknown")
                common_intents[intent] = common_intents.get(intent, 0) + r.score

            # Determine preferred modality
            modality_counts = {"text": 0, "gesture": 0, "speech": 0}
            episodes = self._recent_episodes.get(user_id, [])
            for ep in episodes:
                for utt in ep["data"].utterances:
                    utt_type = utt.get("type", "text")
                    if "gesture" in str(utt_type).lower() or utt.get("gesture"):
                        modality_counts["gesture"] += 1
                    elif "speech" in str(utt_type).lower():
                        modality_counts["speech"] += 1
                    else:
                        modality_counts["text"] += 1

            preferred_modality = max(modality_counts, key=modality_counts.get)

            # Emotion trend
            emotion_trend = self._analyze_emotion_trend(user_id)

            # Recent topics
            recent_topics = self._extract_recent_topics(user_id)

            # Learning progress
            progress = self._estimate_learning_progress(user_id)

            return UserPatterns(
                user_id=user_id,
                frequent_gestures=[
                    {"type": k, "count": v}
                    for k, v in sorted(frequent_gestures.items(), key=lambda x: -x[1])[:10]
                ],
                common_intents=[
                    {"intent": k, "score": round(v, 4)}
                    for k, v in sorted(common_intents.items(), key=lambda x: -x[1])[:10]
                ],
                preferred_modality=preferred_modality,
                average_confidence=self._compute_average_confidence(user_id),
                emotion_trend=emotion_trend,
                recent_topics=recent_topics,
                peak_usage_hours=self._compute_peak_hours(user_id),
                learning_progress=progress,
            )

        except Exception as e:
            logger.error("pattern_extraction_failed", user_id=user_id, error=str(e))
            return UserPatterns(
                user_id=user_id,
                frequent_gestures=[],
                common_intents=[],
                preferred_modality="text",
                average_confidence=0.0,
                emotion_trend="stable",
                recent_topics=[],
                peak_usage_hours=[],
                learning_progress=0.0,
            )

    def delete_user_data(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
    ) -> int:
        """Delete all memory data for a user.

        Args:
            user_id: User identifier.
            memory_type: Optional filter ('episodic', 'semantic', 'procedural').

        Returns:
            Number of deleted items.
        """
        deleted = 0
        collections_map = {
            "episodic": self._episodic_collection,
            "semantic": self._semantic_collection,
            "procedural": self._procedural_collection,
        }

        targets = [col for name, col in collections_map.items()
                   if memory_type is None or name == memory_type]

        for col in targets:
            try:
                results = self._vs.search_similar(
                    col,
                    np.zeros(self._embedder._embedding_dim, dtype=np.float32),
                    k=1000,
                    filters={"user_id": user_id},
                )
                for r in results:
                    if self._vs.delete_embedding(col, r.id):
                        deleted += 1
            except Exception:
                pass

        self._session_counters.pop(user_id, None)
        self._recent_episodes.pop(user_id, None)
        logger.info("user_data_deleted", user_id=user_id, deleted=deleted)
        return deleted

    def run_forgetting_curve_maintenance(self) -> Dict[str, int]:
        """Run maintenance to remove expired memories based on forgetting curves.

        Returns:
            Dict with count of removed items per collection.
        """
        removed = {}
        for collection, ttl_days in [
            (self._episodic_collection, self.EPISODIC_TTL_DAYS),
            (self._semantic_collection, self.SEMANTIC_TTL_DAYS),
        ]:
            count = self._prune_old_memories(collection, ttl_days)
            removed[collection] = count
        logger.info("forgetting_curve_maintenance_complete", removed=removed)
        return removed

    def _consolidate_memories(self, user_id: str) -> None:
        """Consolidate episodic memories into semantic memory.

        Extracts patterns and important facts from recent episodes
        and stores them as semantic memories.

        Args:
            user_id: User identifier.
        """
        episodes = self._recent_episodes.get(user_id, [])
        if len(episodes) < self.CONSOLIDATION_INTERVAL // 2:
            return

        logger.info("consolidating_memories", user_id=user_id, episodes=len(episodes))

        try:
            # Aggregate frequent patterns
            all_utterances: List[str] = []
            intent_counts: Dict[str, int] = {}

            for ep in episodes:
                for utt in ep["data"].utterances:
                    text = utt.get("text", "")
                    if text:
                        all_utterances.append(text)
                    intent = utt.get("intent", "")
                    if intent:
                        intent_counts[intent] = intent_counts.get(intent, 0) + 1

            # Store consolidated semantic memories
            if all_utterances:
                summary = " ".join(all_utterances[-20:])
                emb = self._embedder.generate_text_embedding(summary, use_cache=True)
                self._vs.store_embedding(
                    self._semantic_collection,
                    emb.vector,
                    metadata={
                        "user_id": user_id,
                        "type": "semantic",
                        "consolidated": True,
                        "source_count": len(all_utterances),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "top_intents": dict(sorted(intent_counts.items(), key=lambda x: -x[1])[:5]),
                    },
                )

            # Keep only recent episodes
            self._recent_episodes[user_id] = episodes[-self.CONSOLIDATION_INTERVAL:]

        except Exception as e:
            logger.error("memory_consolidation_failed", user_id=user_id, error=str(e))

    def _forgetting_curve_factor(self, item: MemoryItem) -> float:
        """Compute a forgetting curve multiplier based on memory age.

        Uses Ebbinghaus-inspired exponential decay.

        Args:
            item: Memory item with age.

        Returns:
            Decay factor in [0, 1].
        """
        age_days = item.age_hours / 24.0
        decay = np.exp(-self.FORGETTING_DECAY_RATE * age_days)

        # Repetition boost: memories seen more often decay slower
        repetitions = item.metadata.get("repetitions", 1)
        boost = 1.0 + (repetitions - 1) * (self.FORGETTING_REPETITION_BOOST - 1.0) / 10.0
        boost = min(boost, self.FORGETTING_REPETITION_BOOST)

        return float(min(decay * boost, 1.0))

    def _result_to_memory_item(
        self,
        result: SearchResult,
        memory_type: str,
    ) -> Optional[MemoryItem]:
        """Convert a SearchResult to a MemoryItem.

        Args:
            result: Search result from vector store.
            memory_type: Type of memory ('semantic', 'episodic', 'procedural').

        Returns:
            MemoryItem or None if conversion fails.
        """
        try:
            ts_str = result.metadata.get("timestamp", "")
            ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)
            age = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0

            # Extract content from metadata
            content = result.metadata.get("utterance_type", "") or \
                      result.metadata.get("gesture_type", "") or \
                      result.metadata.get("preference_key", "") or ""

            return MemoryItem(
                id=result.id,
                content=content,
                embedding=result.embedding,
                metadata=result.metadata,
                score=result.score,
                memory_type=memory_type,
                timestamp=ts,
                age_hours=age,
            )
        except Exception:
            return None

    def _analyze_emotion_trend(self, user_id: str) -> str:
        """Analyze emotion trend from recent memories.

        Args:
            user_id: User identifier.

        Returns:
            'improving', 'declining', or 'stable'.
        """
        try:
            results = self._vs.search_similar(
                self._episodic_collection,
                self._embedder.generate_text_embedding("emotion feeling", use_cache=True).vector,
                k=10,
                filters={"user_id": user_id},
            )
            if len(results) < 3:
                return "stable"

            valence_values = []
            for r in results:
                v = r.metadata.get("valence", 0.5)
                try:
                    valence_values.append(float(v))
                except (ValueError, TypeError):
                    valence_values.append(0.5)

            if len(valence_values) >= 2:
                mid = len(valence_values) // 2
                first_half = sum(valence_values[:mid]) / mid
                second_half = sum(valence_values[mid:]) / (len(valence_values) - mid)
                diff = second_half - first_half
                if diff > 0.1:
                    return "improving"
                elif diff < -0.1:
                    return "declining"
            return "stable"
        except Exception:
            return "stable"

    def _extract_recent_topics(self, user_id: str) -> List[str]:
        """Extract recent conversation topics.

        Args:
            user_id: User identifier.

        Returns:
            List of topic strings.
        """
        try:
            results = self._vs.search_similar(
                self._episodic_collection,
                self._embedder.generate_text_embedding("topic about", use_cache=True).vector,
                k=5,
                filters={"user_id": user_id},
            )
            topics = []
            for r in results:
                intent = r.metadata.get("intent", "")
                utype = r.metadata.get("utterance_type", "")
                if intent:
                    topics.append(intent)
                if utype:
                    topics.append(utype)
            return list(dict.fromkeys(topics))[:5]
        except Exception:
            return []

    def _estimate_learning_progress(self, user_id: str) -> float:
        """Estimate user's learning progress over time.

        Based on confidence score trends from semantic memories.

        Args:
            user_id: User identifier.

        Returns:
            Progress score in [0, 1].
        """
        try:
            results = self._vs.search_similar(
                self._semantic_collection,
                self._embedder.generate_text_embedding("learning progress", use_cache=True).vector,
                k=10,
                filters={"user_id": user_id},
            )
            if not results:
                return 0.0

            confidences = []
            for r in results:
                c = r.metadata.get("confidence", 0.5)
                try:
                    confidences.append(float(c))
                except (ValueError, TypeError):
                    confidences.append(0.5)

            if not confidences:
                return 0.0

            # Progress = average of recent confidence scores
            avg = sum(confidences) / len(confidences)
            return float(np.clip(avg, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_average_confidence(self, user_id: str) -> float:
        """Compute average confidence across all memory types.

        Args:
            user_id: User identifier.

        Returns:
            Average confidence in [0, 1].
        """
        confidences = []
        try:
            for collection in [self._semantic_collection, self._episodic_collection]:
                results = self._vs.search_similar(
                    collection,
                    np.zeros(self._embedder._embedding_dim, dtype=np.float32),
                    k=50,
                    filters={"user_id": user_id},
                )
                for r in results:
                    c = r.metadata.get("confidence", 0.0)
                    try:
                        confidences.append(float(c))
                    except (ValueError, TypeError):
                        pass

            if not confidences:
                return 0.0
            return round(sum(confidences) / len(confidences), 4)
        except Exception:
            return 0.0

    def _compute_peak_hours(self, user_id: str) -> List[int]:
        """Compute peak usage hours from episodic memory timestamps.

        Args:
            user_id: User identifier.

        Returns:
            List of hour integers (0-23) sorted by activity.
        """
        hours: Dict[int, int] = {}
        try:
            episodes = self._recent_episodes.get(user_id, [])
            for ep in episodes:
                hour = ep["timestamp"].hour
                hours[hour] = hours.get(hour, 0) + 1

            sorted_hours = sorted(hours.items(), key=lambda x: -x[1])
            return [h for h, _ in sorted_hours[:5]]
        except Exception:
            return []

    def _prune_old_memories(self, collection: str, ttl_days: float) -> int:
        """Remove memories older than TTL from a collection.

        Args:
            collection: Collection name.
            ttl_days: Maximum age in days.

        Returns:
            Number of removed items.
        """
        removed = 0
        try:
            # Retrieve all items (approximate - limited to 1000 for safety)
            dummy_emb = np.zeros(self._embedder._embedding_dim, dtype=np.float32)
            results = self._vs.search_similar(collection, dummy_emb, k=1000)

            cutoff = datetime.now(timezone.utc).timestamp() - ttl_days * 86400
            for r in results:
                ts_str = r.metadata.get("timestamp", "")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str).timestamp()
                        if ts < cutoff:
                            if self._vs.delete_embedding(collection, r.id):
                                removed += 1
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            logger.warning("pruning_failed", collection=collection, error=str(e))

        return removed
