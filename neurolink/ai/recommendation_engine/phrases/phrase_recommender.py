from __future__ import annotations

import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import structlog

from ai.adaptation_engine.personalization.user_profiler import UserProfile

logger = structlog.get_logger(__name__)


class PhraseRecommendationError(Exception):
    """Raised when phrase recommendation fails."""


@dataclass
class RecommendedPhrase:
    """A recommended phrase with context and relevance."""

    text: str
    score: float
    category: str  # "greeting", "request", "emergency", "small_talk", etc.
    source: str  # "semantic", "frequency", "context", "user_pattern"
    context_relevance: float  # 0 to 1
    diversity_bonus: float = 0.0

    @property
    def final_score(self) -> float:
        return self.score * (0.7 + 0.3 * self.diversity_bonus)


class PhraseRecommender:
    """Phrase recommendation engine with semantic similarity and context awareness.

    Recommends relevant phrases based on conversation context, user profile,
    and semantic similarity. Supports diversity in recommendations and
    context-aware filtering.

    Features:
    - Semantic similarity search using embeddings
    - Usage frequency weighting
    - Context-aware filtering (topic-based)
    - Diversity in recommendations
    - Multi-category phrase database
    """

    # Internal phrase database with categories
    PHRASE_DATABASE: Dict[str, List[Tuple[str, float]]] = {
        "greeting": [
            ("Hello!", 1.0), ("Hi there!", 0.9), ("Hey!", 0.8),
            ("Good morning!", 0.9), ("Good afternoon!", 0.85),
            ("Good evening!", 0.85), ("Nice to meet you!", 0.8),
            ("How are you?", 0.9), ("How's it going?", 0.7),
        ],
        "farewell": [
            ("Goodbye!", 1.0), ("See you later!", 0.9), ("Take care!", 0.85),
            ("Bye!", 0.8), ("See you soon!", 0.85), ("Have a good day!", 0.9),
        ],
        "request": [
            ("I need help", 1.0), ("Can you help me?", 0.95),
            ("I would like...", 0.8), ("Please assist me", 0.85),
            ("Could you please...", 0.8), ("I need assistance", 0.85),
            ("Help me with this", 0.9),
        ],
        "question": [
            ("What is this?", 0.9), ("How do I do this?", 0.85),
            ("What's happening?", 0.8), ("Can you explain?", 0.85),
            ("Why is that?", 0.75), ("Where is...?", 0.8),
            ("When will...?", 0.75), ("Who is...?", 0.75),
        ],
        "affirmation": [
            ("Yes", 1.0), ("Yes, please", 0.9), ("Sure", 0.85),
            ("Okay", 0.8), ("I understand", 0.85), ("That's correct", 0.8),
            ("Absolutely", 0.75), ("Of course", 0.8),
        ],
        "negation": [
            ("No", 1.0), ("No, thank you", 0.9), ("Not right now", 0.8),
            ("I don't think so", 0.75), ("I disagree", 0.7),
            ("That's not right", 0.75),
        ],
        "emergency": [
            ("Emergency!", 1.0), ("Help!", 0.95), ("Call for help!", 0.95),
            ("I need a doctor!", 0.9), ("It's an emergency!", 0.95),
            ("Someone help me!", 0.9), ("I'm in danger!", 0.9),
        ],
        "pain": [
            ("I'm in pain", 1.0), ("It hurts", 0.9), ("I need medicine", 0.85),
            ("I need a doctor", 0.9), ("It's very painful", 0.85),
        ],
        "needs": [
            ("I need water", 0.9), ("I need food", 0.9),
            ("I'm hungry", 0.85), ("I'm thirsty", 0.85),
            ("I need to rest", 0.8), ("I need the bathroom", 0.85),
        ],
        "emotion": [
            ("I'm happy", 0.9), ("I'm sad", 0.85), ("I'm tired", 0.85),
            ("I'm scared", 0.8), ("I'm confused", 0.8),
            ("I'm frustrated", 0.75), ("I'm grateful", 0.8),
        ],
        "small_talk": [
            ("Thank you", 0.9), ("You're welcome", 0.85),
            ("I'm sorry", 0.85), ("That's great!", 0.8),
            ("No problem", 0.8), ("Great job!", 0.75),
            ("Well done!", 0.75), ("Good luck!", 0.7),
        ],
        "clarification": [
            ("I don't understand", 0.9), ("Can you repeat that?", 0.85),
            ("What did you say?", 0.8), ("Please slow down", 0.8),
            ("Can you explain again?", 0.85), ("I'm not sure", 0.75),
        ],
    }

    # Context-to-category mapping
    CONTEXT_CATEGORY_MAP: Dict[str, List[str]] = {
        "hello": ["greeting", "small_talk"],
        "hi": ["greeting", "small_talk"],
        "help": ["request", "emergency"],
        "pain": ["pain", "emergency", "needs"],
        "hurt": ["pain", "emergency"],
        "emergency": ["emergency"],
        "thank": ["small_talk", "farewell"],
        "bye": ["farewell", "small_talk"],
        "yes": ["affirmation"],
        "no": ["negation"],
        "question": ["question", "clarification"],
        "what": ["question", "clarification"],
        "why": ["question"],
        "hungry": ["needs"],
        "thirsty": ["needs"],
        "tired": ["emotion", "needs"],
        "sad": ["emotion"],
        "happy": ["emotion", "small_talk"],
        "scared": ["emotion", "emergency"],
        "confused": ["clarification", "emotion"],
        "sorry": ["small_talk", "clarification"],
    }

    def __init__(
        self,
        num_recommendations: int = 5,
        diversity_factor: float = 0.3,
        min_score: float = 0.1,
        enable_semantic_search: bool = False,
    ):
        """Initialize phrase recommender.

        Args:
            num_recommendations: Number of recommendations to return.
            diversity_factor: How much to penalize same-category repeats (0-1).
            min_score: Minimum score for recommendation.
            enable_semantic_search: Enable embedding-based semantic search.
        """
        self._num_recommendations = num_recommendations
        self._diversity_factor = diversity_factor
        self._min_score = min_score
        self._enable_semantic_search = enable_semantic_search

        # Build embedding cache for semantic search
        self._phrase_embeddings: Dict[str, np.ndarray] = {}
        if self._enable_semantic_search:
            self._build_semantic_embeddings()

        # Track recent recommendations for diversity
        self._recent_recommendations: Dict[str, List[str]] = defaultdict(list)

        logger.info(
            "phrase_recommender_initialized",
            num_recommendations=num_recommendations,
            diversity_factor=diversity_factor,
            semantic_search=enable_semantic_search,
        )

    def recommend_phrases(
        self,
        context: str = "",
        user_profile: Optional[UserProfile] = None,
        current_intent: str = "unknown",
        num_recommendations: Optional[int] = None,
        exclude_phrases: Optional[List[str]] = None,
    ) -> List[RecommendedPhrase]:
        """Get phrase recommendations based on context and user profile.

        Args:
            context: Current conversation context.
            user_profile: Optional user profile for personalization.
            current_intent: Current detected intent.
            num_recommendations: Override default count.
            exclude_phrases: Phrases to exclude from recommendations.

        Returns:
            List of RecommendedPhrase sorted by score.

        Raises:
            PhraseRecommendationError: If recommendation fails.
        """
        try:
            n = num_recommendations or self._num_recommendations
            exclude = set(p.lower().strip() for p in (exclude_phrases or []))

            # Collect candidate phrases with scores
            candidates: List[RecommendedPhrase] = []

            # 1. Context-based recommendations
            context_candidates = self._get_context_based_phrases(context)
            candidates.extend(context_candidates)

            # 2. Intent-based recommendations
            intent_candidates = self._get_intent_based_phrases(current_intent)
            candidates.extend(intent_candidates)

            # 3. User pattern-based recommendations
            if user_profile:
                user_candidates = self._get_user_pattern_phrases(user_profile, context)
                candidates.extend(user_candidates)

            # 4. Semantic similarity recommendations
            if self._enable_semantic_search and context:
                semantic_candidates = self._get_semantic_phrases(context)
                candidates.extend(semantic_candidates)

            # 5. Fallback: popular/default phrases
            if not candidates:
                default_candidates = self._get_default_phrases()
                candidates.extend(default_candidates)

            # Filter exclusions
            candidates = [
                c for c in candidates
                if c.text.lower().strip() not in exclude
            ]

            # Score, diversify, and sort
            recommendations = self._score_and_diversify(
                candidates, context, user_profile
            )

            # Return top N
            return recommendations[:n]

        except Exception as e:
            logger.error("phrase_recommendation_failed", error=str(e))
            raise PhraseRecommendationError(f"Phrase recommendation failed: {e}") from e

    def _get_context_based_phrases(self, context: str) -> List[RecommendedPhrase]:
        """Get phrases based on conversation context keywords.

        Args:
            context: Current context text.

        Returns:
            List of context-based recommended phrases.
        """
        if not context:
            return []

        context_lower = context.lower()
        candidates: List[RecommendedPhrase] = []
        seen_categories: Set[str] = set()

        # Find matching categories based on context keywords
        matching_categories: Set[str] = set()
        for keyword, categories in self.CONTEXT_CATEGORY_MAP.items():
            if keyword in context_lower:
                matching_categories.update(categories)

        # Get phrases from matching categories
        for category in matching_categories:
            if category in self.PHRASE_DATABASE:
                seen_categories.add(category)
                for phrase_text, base_score in self.PHRASE_DATABASE[category]:
                    candidates.append(RecommendedPhrase(
                        text=phrase_text,
                        score=base_score * 0.9,  # Slightly discounted
                        category=category,
                        source="context",
                        context_relevance=self._compute_context_relevance(
                            phrase_text, context_lower
                        ),
                    ))

        # If no matching categories, return general phrases
        if not candidates:
            for category in ["greeting", "small_talk", "clarification"]:
                if category in self.PHRASE_DATABASE:
                    for phrase_text, base_score in self.PHRASE_DATABASE[category][:3]:
                        candidates.append(RecommendedPhrase(
                            text=phrase_text,
                            score=base_score * 0.5,
                            category=category,
                            source="context",
                            context_relevance=0.3,
                        ))

        return candidates

    def _get_intent_based_phrases(self, intent: str) -> List[RecommendedPhrase]:
        """Get phrases based on the current classified intent.

        Args:
            intent: Current intent label.

        Returns:
            List of intent-based recommended phrases.
        """
        intent_to_category: Dict[str, str] = {
            "greeting": "greeting",
            "farewell": "farewell",
            "request": "request",
            "question": "question",
            "affirmation": "affirmation",
            "negation": "negation",
            "emergency": "emergency",
            "help": "request",
            "pain": "pain",
            "thanks": "small_talk",
            "apology": "small_talk",
            "clarification": "clarification",
        }

        category = intent_to_category.get(intent, "small_talk")
        candidates: List[RecommendedPhrase] = []

        if category in self.PHRASE_DATABASE:
            for phrase_text, base_score in self.PHRASE_DATABASE[category]:
                candidates.append(RecommendedPhrase(
                    text=phrase_text,
                    score=base_score * 1.0,
                    category=category,
                    source="intent",
                    context_relevance=0.8,
                ))

        return candidates

    def _get_user_pattern_phrases(
        self, profile: UserProfile, context: str
    ) -> List[RecommendedPhrase]:
        """Get phrases based on user's historical patterns.

        Args:
            profile: User profile with interaction history.
            context: Current context.

        Returns:
            List of user-specific recommended phrases.
        """
        candidates: List[RecommendedPhrase] = []

        # Get frequently used phrases
        if profile.speech_patterns:
            total = sum(profile.speech_patterns.values())
            sorted_phrases = sorted(
                profile.speech_patterns.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]

            for phrase_text, count in sorted_phrases:
                frequency = count / total
                candidates.append(RecommendedPhrase(
                    text=phrase_text,
                    score=frequency * 0.8,
                    category="user_pattern",
                    source="user_pattern",
                    context_relevance=self._compute_context_relevance(phrase_text, context.lower() if context else ""),
                ))

        # Add preferred gestures as phrase suggestions
        if profile.preferred_gestures:
            for gesture in profile.preferred_gestures[:5]:
                phrase = gesture.replace("_", " ").title()
                candidates.append(RecommendedPhrase(
                    text=phrase,
                    score=0.4,
                    category="user_pattern",
                    source="gesture_preference",
                    context_relevance=0.5,
                ))

        return candidates

    def _get_semantic_phrases(self, context: str) -> List[RecommendedPhrase]:
        """Get phrases based on semantic similarity to context.

        Args:
            context: Current context text.

        Returns:
            List of semantically similar recommended phrases.
        """
        if not self._enable_semantic_search or not self._phrase_embeddings:
            return []

        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            context_emb = model.encode(context)

            candidates: List[RecommendedPhrase] = []
            for category, phrases in self.PHRASE_DATABASE.items():
                for phrase_text, base_score in phrases:
                    if phrase_text in self._phrase_embeddings:
                        phrase_emb = self._phrase_embeddings[phrase_text]
                        similarity = float(np.dot(context_emb, phrase_emb) / (
                            np.linalg.norm(context_emb) * np.linalg.norm(phrase_emb) + 1e-8
                        ))
                        if similarity > 0.3:
                            candidates.append(RecommendedPhrase(
                                text=phrase_text,
                                score=base_score * similarity,
                                category=category,
                                source="semantic",
                                context_relevance=similarity,
                            ))
            return candidates

        except ImportError:
            self._enable_semantic_search = False
            return []
        except Exception as e:
            logger.warning("semantic_search_failed", error=str(e))
            return []

    def _get_default_phrases(self) -> List[RecommendedPhrase]:
        """Get default/fallback phrases.

        Returns:
            List of default recommended phrases.
        """
        candidates: List[RecommendedPhrase] = []
        for category in ["greeting", "small_talk", "clarification"]:
            if category in self.PHRASE_DATABASE:
                for phrase_text, base_score in self.PHRASE_DATABASE[category][:2]:
                    candidates.append(RecommendedPhrase(
                        text=phrase_text,
                        score=base_score * 0.3,
                        category=category,
                        source="default",
                        context_relevance=0.2,
                    ))
        return candidates

    def _score_and_diversify(
        self,
        candidates: List[RecommendedPhrase],
        context: str,
        user_profile: Optional[UserProfile],
    ) -> List[RecommendedPhrase]:
        """Score candidates and apply diversity bonus.

        Promotes diversity by penalizing multiple candidates from the same category
        and boosting underrepresented categories.

        Args:
            candidates: Raw candidate phrases.
            context: Current context.
            user_profile: User profile for personalization.

        Returns:
            Scored and diversified recommendations.
        """
        if not candidates:
            return []

        # Count categories for diversity
        category_counts = Counter(c.category for c in candidates)

        # Apply diversity bonus and compute final scores
        scored = []
        for c in candidates:
            # Diversity bonus: penalize repeated categories
            if category_counts[c.category] > 1:
                c.diversity_bonus = 0.0
            else:
                c.diversity_bonus = self._diversity_factor

            # User profile personalization bonus
            if user_profile and c.text.lower() in set(
                p.lower() for p in user_profile.speech_patterns
            ):
                c.diversity_bonus += 0.1

            scored.append(c)

        # Sort by final score
        scored.sort(key=lambda x: x.final_score, reverse=True)

        # Ensure category diversity in top results
        diverse: List[RecommendedPhrase] = []
        seen_categories: Set[str] = set()

        for c in scored:
            if len(diverse) >= self._num_recommendations:
                break
            if c.category not in seen_categories:
                diverse.append(c)
                seen_categories.add(c.category)
            elif len(diverse) < self._num_recommendations // 2:
                diverse.append(c)

        # Fill remaining slots
        for c in scored:
            if len(diverse) >= self._num_recommendations:
                break
            if c not in diverse:
                diverse.append(c)

        return diverse

    def _compute_context_relevance(self, phrase: str, context: str) -> float:
        """Compute relevance of a phrase to the current context.

        Uses keyword overlap as a simple relevance metric.

        Args:
            phrase: Phrase to evaluate.
            context: Current context text (lowercase).

        Returns:
            Relevance score in [0, 1].
        """
        if not context:
            return 0.0

        phrase_words = set(phrase.lower().split())
        context_words = set(context.split())

        if not phrase_words:
            return 0.0

        overlap = phrase_words & context_words
        if overlap:
            return min(1.0, len(overlap) / max(len(phrase_words), 1))
        return 0.0

    def _build_semantic_embeddings(self) -> None:
        """Build embedding cache for all phrases in database."""
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            all_phrases = [
                phrase
                for phrases in self.PHRASE_DATABASE.values()
                for phrase, _ in phrases
            ]
            embeddings = model.encode(all_phrases)

            self._phrase_embeddings = {
                phrase: embeddings[i]
                for i, phrase in enumerate(all_phrases)
            }

            logger.info("semantic_embeddings_built", count=len(self._phrase_embeddings))

        except ImportError:
            logger.warning("sentence_transformers not available, semantic search disabled")
            self._enable_semantic_search = False
        except Exception as e:
            logger.warning("failed_to_build_embeddings", error=str(e))
            self._enable_semantic_search = False
