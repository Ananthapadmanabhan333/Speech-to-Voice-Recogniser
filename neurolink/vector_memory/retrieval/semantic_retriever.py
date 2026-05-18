from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import structlog

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError:
    CrossEncoder = None
    SentenceTransformer = None

from vector_memory.embeddings.embedding_generator import EmbeddingGenerator
from vector_memory.store.vector_store import SearchResult, VectorStore

logger = structlog.get_logger(__name__)


class SemanticRetrievalError(Exception):
    """Raised when semantic retrieval fails."""


@dataclass
class RetrievedItem:
    """A single retrieved item with relevance scores."""

    id: str
    content: str
    score: float
    rerank_score: float
    final_score: float
    modality: str
    source_collection: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    timestamp: Optional[str] = None


@dataclass
class RetrievalMetrics:
    """Metrics for a retrieval operation."""

    total_candidates: int
    final_count: int
    embedding_search_ms: float
    rerank_ms: float
    total_ms: float
    diversity_score: float
    filter_breakdown: Dict[str, int]


class SemanticRetriever:
    """Multi-stage semantic retriever with cross-encoder reranking.

    Implements a production-grade retrieval pipeline:
    1. Embedding-based retrieval (dense search)
    2. Optional sparse keyword search (hybrid)
    3. Cross-encoder reranking for precision
    4. Context-aware weighting
    5. Diversity enforcement via MMR
    6. Filtering by modality, time range, confidence

    Features:
    - Multi-stage retrieval for high precision/recall
    - Cross-encoder reranking with sentence-transformers
    - Hybrid dense + sparse search
    - MMR (Maximal Marginal Relevance) for diversity
    - Context-aware relevance weighting
    - Comprehensive filtering
    """

    DEFAULT_RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    DEFAULT_SPARSE_WEIGHT: float = 0.3
    MMR_LAMBDA: float = 0.7
    MAX_CANDIDATES: int = 100

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        rerank_model_name: str = DEFAULT_RERANK_MODEL,
        enable_reranking: bool = True,
        enable_hybrid: bool = True,
        dense_weight: float = 0.7,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
        mmr_lambda: float = MMR_LAMBDA,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the semantic retriever.

        Args:
            vector_store: VectorStore instance.
            embedding_generator: EmbeddingGenerator instance.
            rerank_model_name: Cross-encoder model for reranking.
            enable_reranking: Enable cross-encoder reranking stage.
            enable_hybrid: Enable hybrid dense + sparse search.
            dense_weight: Weight for dense embedding scores.
            sparse_weight: Weight for sparse keyword scores.
            mmr_lambda: MMR diversity parameter (0=max diversity, 1=max relevance).
            device: Device for model inference.
        """
        self._vs = vector_store
        self._embedder = embedding_generator
        self._rerank_model_name = rerank_model_name
        self._enable_reranking = enable_reranking
        self._enable_hybrid = enable_hybrid
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._mmr_lambda = mmr_lambda
        self._device = device or "cpu"

        # Lazy-loaded reranking model
        self._rerank_model: Optional[CrossEncoder] = None

        logger.info(
            "semantic_retriever_initialized",
            rerank=enable_reranking,
            hybrid=enable_hybrid,
            mmr_lambda=mmr_lambda,
        )

    def retrieve(
        self,
        query: str,
        user_id: Optional[str] = None,
        modality: Optional[str] = None,
        k: int = 10,
        collections: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        time_range_hours: Optional[Tuple[float, float]] = None,
        min_confidence: float = 0.0,
        diversity: bool = True,
    ) -> Tuple[List[RetrievedItem], RetrievalMetrics]:
        """Multi-stage retrieval pipeline.

        Args:
            query: Natural language query.
            user_id: Filter by user.
            modality: Filter by modality ('gesture', 'speech', 'text', 'emotion').
            k: Number of final results.
            collections: Collections to search (default: all).
            filters: Additional metadata filters.
            time_range_hours: (start_hours_ago, end_hours_ago) tuple.
            min_confidence: Minimum confidence threshold.
            diversity: Apply MMR diversity to results.

        Returns:
            Tuple of (list of RetrievedItem, RetrievalMetrics).

        Raises:
            SemanticRetrievalError: If retrieval fails.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        metrics_start = time.perf_counter()
        filter_breakdown: Dict[str, int] = {}

        try:
            # Stage 1: Dense embedding search
            embed_start = time.perf_counter()
            candidates = self._dense_search(
                query=query,
                user_id=user_id,
                modality=modality,
                collections=collections,
                filters=filters,
            )
            embed_time = (time.perf_counter() - embed_start) * 1000.0
            filter_breakdown["dense_candidates"] = len(candidates)

            # Stage 1b: Sparse search (hybrid)
            if self._enable_hybrid:
                sparse_candidates = self._sparse_search(
                    query=query,
                    user_id=user_id,
                    collections=collections,
                )
                filter_breakdown["sparse_candidates"] = len(sparse_candidates)
                candidates = self._merge_dense_sparse(candidates, sparse_candidates)
                filter_breakdown["merged_candidates"] = len(candidates)

            # Apply time range filter
            if time_range_hours:
                candidates = self._filter_by_time_range(candidates, time_range_hours)
                filter_breakdown["after_time_filter"] = len(candidates)

            # Apply min confidence filter
            if min_confidence > 0:
                candidates = [c for c in candidates if c.get("confidence", 1.0) >= min_confidence]
                filter_breakdown["after_confidence_filter"] = len(candidates)

            # Stage 2: Cross-encoder reranking
            rerank_start = time.perf_counter()
            if self._enable_reranking and candidates:
                candidates = self._rerank(query, candidates)

            rerank_time = (time.perf_counter() - rerank_start) * 1000.0

            # Stage 3: MMR diversity
            if diversity and candidates:
                candidates = self._apply_mmr(query, candidates, k)

            # Limit to final k
            top = candidates[:k]

            # Convert to RetrievedItem
            results = []
            for i, c in enumerate(top):
                rerank_score = c.get("rerank_score", c.get("score", 0.0))
                dense_score = c.get("score", 0.0)
                final_score = rerank_score if self._enable_reranking else dense_score
                results.append(RetrievedItem(
                    id=c.get("id", ""),
                    content=c.get("content", ""),
                    score=dense_score,
                    rerank_score=rerank_score,
                    final_score=final_score,
                    modality=c.get("metadata", {}).get("modality", "text"),
                    source_collection=c.get("collection", ""),
                    metadata=c.get("metadata", {}),
                    embedding=c.get("embedding"),
                    timestamp=c.get("metadata", {}).get("timestamp"),
                ))

            metrics_total = (time.perf_counter() - metrics_start) * 1000.0
            diversity_score = self._compute_diversity_score(results)
            total_candidates = filter_breakdown.get("dense_candidates", 0)

            metrics = RetrievalMetrics(
                total_candidates=total_candidates,
                final_count=len(results),
                embedding_search_ms=round(embed_time, 2),
                rerank_ms=round(rerank_time, 2),
                total_ms=round(metrics_total, 2),
                diversity_score=round(diversity_score, 4),
                filter_breakdown=filter_breakdown,
            )

            logger.debug(
                "retrieval_complete",
                query=query[:50],
                candidates=total_candidates,
                final=len(results),
                metrics=f"{metrics_total:.1f}ms",
            )
            return results, metrics

        except Exception as e:
            logger.error("retrieval_failed", query=query[:50], error=str(e))
            raise SemanticRetrievalError(f"Semantic retrieval failed: {e}") from e

    def _dense_search(
        self,
        query: str,
        user_id: Optional[str],
        modality: Optional[str],
        collections: Optional[List[str]],
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Stage 1: Dense embedding similarity search.

        Args:
            query: Search query.
            user_id: Optional user filter.
            modality: Optional modality filter.
            collections: Collections to search.
            filters: Additional filters.

        Returns:
            List of candidate dicts.
        """
        query_emb = self._embedder.generate_text_embedding(query, use_cache=True)

        search_filters: Dict[str, Any] = {}
        if user_id:
            search_filters["user_id"] = user_id
        if modality:
            search_filters["modality"] = modality
        if filters:
            search_filters.update(filters)

        targets = collections or [
            "semantic_memory", "episodic_memory", "procedural_memory",
        ]

        all_candidates: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        for col in targets:
            try:
                results = self._vs.search_similar(
                    collection=col,
                    embedding=query_emb.vector,
                    k=self.MAX_CANDIDATES // len(targets),
                    filters=search_filters or None,
                )
                for r in results:
                    if r.id not in seen_ids:
                        seen_ids.add(r.id)
                        all_candidates.append({
                            "id": r.id,
                            "score": r.score,
                            "metadata": r.metadata,
                            "embedding": r.embedding,
                            "collection": col,
                            "content": self._extract_content(r.metadata),
                        })
            except Exception as e:
                logger.warning("dense_search_failed", collection=col, error=str(e))

        # Sort by score descending
        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        return all_candidates[:self.MAX_CANDIDATES]

    def _sparse_search(
        self,
        query: str,
        user_id: Optional[str],
        collections: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Stage 1b: Sparse keyword search for hybrid retrieval.

        Uses simple keyword matching as a sparse signal.

        Args:
            query: Search query.
            user_id: Optional user filter.
            collections: Collections to search.

        Returns:
            List of candidate dicts with BM25-like scores.
        """
        keywords = set(query.lower().split())
        if not keywords:
            return []

        targets = collections or [
            "semantic_memory", "episodic_memory", "procedural_memory",
        ]

        candidates: List[Dict[str, Any]] = []

        for col in targets:
            try:
                # Use search with text-like query via embedding
                dummy_emb = np.zeros(self._embedder._embedding_dim, dtype=np.float32)
                results = self._vs.search_similar(
                    collection=col,
                    embedding=dummy_emb,
                    k=self.MAX_CANDIDATES,
                    filters={"user_id": user_id} if user_id else None,
                )
                for r in results:
                    content = self._extract_content(r.metadata).lower()
                    match_count = sum(1 for kw in keywords if kw in content)
                    if match_count > 0:
                        candidates.append({
                            "id": r.id,
                            "score": match_count / len(keywords),
                            "metadata": r.metadata,
                            "embedding": r.embedding,
                            "collection": col,
                            "content": self._extract_content(r.metadata),
                            "keyword_matches": match_count,
                        })
            except Exception as e:
                logger.warning("sparse_search_failed", collection=col, error=str(e))

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:self.MAX_CANDIDATES // 2]

    def _merge_dense_sparse(
        self,
        dense: List[Dict[str, Any]],
        sparse: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge dense and sparse results using reciprocal rank fusion.

        Args:
            dense: Dense search candidates.
            sparse: Sparse search candidates.

        Returns:
            Merged and re-ranked candidates.
        """
        merged: Dict[str, Dict[str, Any]] = {}
        K = 60  # Constant for RRF

        for rank, item in enumerate(dense):
            item["dense_rank"] = rank
            item["sparse_rank"] = None
            merged[item["id"]] = item

        for rank, item in enumerate(sparse):
            if item["id"] in merged:
                merged[item["id"]]["sparse_rank"] = rank
            else:
                item["dense_rank"] = None
                item["sparse_rank"] = rank
                merged[item["id"]] = item

        # Compute RRF scores
        for item in merged.values():
            dense_rank = item.get("dense_rank")
            sparse_rank = item.get("sparse_rank")
            dense_rrf = 1.0 / (K + dense_rank + 1) if dense_rank is not None else 0
            sparse_rrf = 1.0 / (K + sparse_rank + 1) if sparse_rank is not None else 0
            item["rrf_score"] = (
                self._dense_weight * dense_rrf +
                self._sparse_weight * sparse_rrf
            )
            item["score"] = item["rrf_score"]

        merged_list = list(merged.values())
        merged_list.sort(key=lambda x: x["rrf_score"], reverse=True)
        return merged_list[:self.MAX_CANDIDATES]

    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Stage 2: Cross-encoder reranking.

        Args:
            query: Original query.
            candidates: Candidate items to rerank.

        Returns:
            Reranked candidates with rerank_score set.
        """
        if not candidates:
            return candidates

        model = self._get_rerank_model()
        if model is None:
            # Fall through: use dense scores
            for c in candidates:
                c["rerank_score"] = c.get("score", 0.0)
            return candidates

        try:
            pairs = []
            for c in candidates:
                content = c.get("content", "")
                if not content:
                    content = str(c.get("metadata", {}))
                pairs.append((query, content[:512]))

            rerank_scores = model.predict(pairs)
            if isinstance(rerank_scores, np.ndarray):
                rerank_scores = rerank_scores.tolist()

            for i, c in enumerate(candidates):
                c["rerank_score"] = float(rerank_scores[i]) if i < len(rerank_scores) else c.get("score", 0.0)

        except Exception as e:
            logger.warning("reranking_failed", error=str(e))
            for c in candidates:
                c["rerank_score"] = c.get("score", 0.0)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates

    def _apply_mmr(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        k: int,
    ) -> List[Dict[str, Any]]:
        """Stage 3: Apply Maximal Marginal Relevance for diversity.

        Args:
            query: Original query.
            candidates: Ranked candidates.
            k: Number to select.

        Returns:
            Diverse subset of candidates.
        """
        if not candidates or len(candidates) <= k:
            return candidates[:k]

        selected: List[Dict[str, Any]] = []
        remaining = list(candidates)

        query_emb = self._embedder.generate_text_embedding(query, use_cache=True).vector

        while len(selected) < k and remaining:
            best_idx = 0
            best_score = -float("inf")

            for i, cand in enumerate(remaining):
                relevance = cand.get("rerank_score", cand.get("score", 0.0))

                # Diversity penalty: similarity to already selected
                diversity_penalty = 0.0
                if selected:
                    cand_emb = cand.get("embedding")
                    if cand_emb is not None:
                        similarities = []
                        for sel in selected:
                            sel_emb = sel.get("embedding")
                            if sel_emb is not None:
                                sim = float(np.dot(cand_emb, sel_emb) /
                                            (np.linalg.norm(cand_emb) * np.linalg.norm(sel_emb) + 1e-8))
                                similarities.append(sim)
                        if similarities:
                            diversity_penalty = max(similarities)
                    else:
                        # Fallback: use relative position as proxy
                        diversity_penalty = 1.0 - (i / len(remaining))

                mmr_score = (
                    self._mmr_lambda * relevance -
                    (1 - self._mmr_lambda) * diversity_penalty
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _filter_by_time_range(
        self,
        candidates: List[Dict[str, Any]],
        time_range: Tuple[float, float],
    ) -> List[Dict[str, Any]]:
        """Filter candidates by time range.

        Args:
            candidates: Candidate items.
            time_range: (start_hours_ago, end_hours_ago).

        Returns:
            Filtered candidates.
        """
        import time as time_module
        now = time_module.time()
        start_sec = time_range[0] * 3600 if time_range[0] > 1000 else time_range[0]
        end_sec = time_range[1] * 3600 if time_range[1] > 1000 else time_range[1]

        filtered = []
        for c in candidates:
            ts_str = c.get("metadata", {}).get("timestamp", "")
            if ts_str:
                try:
                    from datetime import datetime
                    ts = datetime.fromisoformat(ts_str).timestamp()
                    age = now - ts
                    if start_sec <= age <= end_sec:
                        filtered.append(c)
                except (ValueError, TypeError):
                    filtered.append(c)
            else:
                filtered.append(c)

        return filtered

    def _compute_diversity_score(self, items: List[RetrievedItem]) -> float:
        """Compute diversity score among retrieved items.

        Uses average pairwise cosine distance of content hashes.

        Args:
            items: Retrieved items.

        Returns:
            Diversity score in [0, 1] (higher = more diverse).
        """
        if len(items) < 2:
            return 1.0

        embeddings = []
        for item in items:
            if item.embedding is not None:
                embeddings.append(item.embedding)

        if len(embeddings) < 2:
            return 0.5

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = float(np.dot(embeddings[i], embeddings[j]) /
                            (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8))
                similarities.append(sim)

        if not similarities:
            return 0.5

        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - float(np.clip(avg_similarity, 0.0, 1.0))

    def _extract_content(self, metadata: Dict[str, Any]) -> str:
        """Extract a text representation from metadata.

        Args:
            metadata: Item metadata.

        Returns:
            Text content string.
        """
        fields = [
            metadata.get("utterance_type", ""),
            metadata.get("gesture_type", ""),
            metadata.get("preference_key", ""),
            metadata.get("preference_value", ""),
            metadata.get("intent", ""),
            metadata.get("session_type", ""),
        ]
        return " ".join(f for f in fields if f)

    def _get_rerank_model(self) -> Optional[CrossEncoder]:
        """Get or load the cross-encoder reranker model (lazy).

        Returns:
            CrossEncoder model or None if unavailable.
        """
        if self._rerank_model is not None:
            return self._rerank_model

        if CrossEncoder is None:
            logger.warning("sentence_transformers not installed, reranking disabled")
            return None

        try:
            logger.info("loading_rerank_model", model=self._rerank_model_name)
            self._rerank_model = CrossEncoder(
                self._rerank_model_name,
                device=self._device,
            )
            logger.info("rerank_model_loaded")
        except Exception as e:
            logger.error("rerank_model_load_failed", error=str(e))
            return None

        return self._rerank_model
