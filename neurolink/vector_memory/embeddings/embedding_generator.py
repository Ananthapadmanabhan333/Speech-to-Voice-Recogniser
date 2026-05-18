from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)


class EmbeddingGenerationError(Exception):
    """Raised when embedding generation fails."


class ModelNotFoundError(EmbeddingGenerationError):
    """Raised when a requested model is not available."""


@dataclass
class EmbeddingResult:
    """Result of a single embedding generation."""

    vector: np.ndarray
    model_name: str
    dimension: int
    normalized: bool
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Entry in the embedding cache."""

    embedding: np.ndarray
    model_name: str
    created_at: float
    ttl: float

    @property
    def is_expired(self) -> bool:
        return time.monotonic() - self.created_at > self.ttl


class EmbeddingGenerator:
    """Generates embeddings for text, gesture landmarks, and multimodal inputs.

    Uses sentence-transformers for text embeddings and learned projections
    for gesture and emotion modalities. Supports model lazy loading, caching
    with TTL, batch generation, and multiple embedding models.

    Features:
    - Text embedding via sentence-transformers
    - Gesture landmark embedding via configurable projection
    - Multimodal fusion embedding
    - Batch generation with automatic batching
    - LRU cache with configurable TTL
    - Model lazy loading for memory efficiency
    - Multiple model support with fallback
    """

    DEFAULT_TEXT_MODEL: str = "all-MiniLM-L6-v2"
    DEFAULT_GESTURE_MODEL: str = "gesture_projection"
    DEFAULT_EMOTION_MODEL: str = "emotion_projection"
    EMBEDDING_DIM: int = 384
    GESTURE_LANDMARK_DIM: int = 63
    GESTURE_EMBEDDING_DIM: int = 128
    EMOTION_EMBEDDING_DIM: int = 64

    def __init__(
        self,
        text_model_name: str = DEFAULT_TEXT_MODEL,
        models_dir: Optional[Path] = None,
        device: Optional[str] = None,
        cache_ttl: float = 300.0,
        cache_maxsize: int = 10000,
        normalize_embeddings: bool = True,
        lazy_loading: bool = True,
        batch_size: int = 32,
    ) -> None:
        """Initialize the embedding generator.

        Args:
            text_model_name: sentence-transformers model name.
            models_dir: Directory for model caching.
            device: Device for model inference ('cpu', 'cuda', 'mps').
            cache_ttl: Cache TTL in seconds (default 5 min).
            cache_maxsize: Maximum cache entries.
            normalize_embeddings: L2-normalize all embeddings.
            lazy_loading: Load models on first use.
            batch_size: Default batch size for generation.
        """
        self._text_model_name = text_model_name
        self._models_dir = models_dir or Path("models/embeddings")
        self._device = device or ("cuda" if self._has_torch_cuda() else "cpu")
        self._cache_ttl = cache_ttl
        self._cache_maxsize = cache_maxsize
        self._normalize = normalize_embeddings
        self._lazy_loading = lazy_loading
        self._batch_size = batch_size

        # Model instances (lazy loaded)
        self._text_model: Optional[SentenceTransformer] = None
        self._gesture_projection: Optional[np.ndarray] = None
        self._emotion_projection: Optional[np.ndarray] = None
        self._fusion_projection: Optional[np.ndarray] = None

        # Cache: key_hash -> CacheEntry
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._total_requests: int = 0

        logger.info(
            "embedding_generator_initialized",
            text_model=text_model_name,
            device=self._device,
            normalize=normalize_embeddings,
            lazy_loading=lazy_loading,
            batch_size=batch_size,
        )

    def generate_text_embedding(
        self,
        text: str,
        model_name: Optional[str] = None,
        use_cache: bool = True,
    ) -> EmbeddingResult:
        """Generate embedding for a text input.

        Args:
            text: Input text string.
            model_name: Override default text model name.
            use_cache: Enable cache lookup.

        Returns:
            EmbeddingResult with vector and metadata.

        Raises:
            EmbeddingGenerationError: If generation fails.
        """
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        model = model_name or self._text_model_name
        cache_key = self._make_cache_key(text, model)

        if use_cache and cache_key in self._cache:
            entry = self._cache[cache_key]
            if not entry.is_expired:
                self._cache_hits += 1
                return EmbeddingResult(
                    vector=entry.embedding.copy(),
                    model_name=entry.model_name,
                    dimension=entry.embedding.shape[0],
                    normalized=self._normalize,
                    processing_time_ms=0.0,
                    metadata={"cached": True},
                )

        self._cache_misses += 1
        self._total_requests += 1
        start = time.perf_counter()

        try:
            model_obj = self._get_text_model(model)
            embedding = model_obj.encode(text, normalize_embeddings=self._normalize)
            vector = np.array(embedding, dtype=np.float32)

            if vector.ndim > 1:
                vector = vector.flatten()

        except Exception as e:
            logger.error("text_embedding_failed", error=str(e), text=text[:50])
            raise EmbeddingGenerationError(f"Text embedding failed: {e}") from e

        elapsed = (time.perf_counter() - start) * 1000.0

        result = EmbeddingResult(
            vector=vector,
            model_name=model,
            dimension=vector.shape[0],
            normalized=self._normalize,
            processing_time_ms=elapsed,
        )

        if use_cache:
            self._add_to_cache(cache_key, vector, model)

        return result

    def generate_gesture_embedding(
        self,
        landmarks: np.ndarray,
        use_cache: bool = True,
    ) -> EmbeddingResult:
        """Generate embedding from hand landmark coordinates.

        Args:
            landmarks: Array of shape (21, 3) or (seq_len, 21, 3).
            use_cache: Enable cache lookup.

        Returns:
            EmbeddingResult with projected gesture embedding.

        Raises:
            EmbeddingGenerationError: If generation fails.
        """
        if landmarks is None or landmarks.size == 0:
            raise ValueError("Landmarks cannot be empty")

        original_shape = landmarks.shape
        if landmarks.ndim == 2:
            landmarks = landmarks[np.newaxis, ...]

        seq_len, num_pts, coords = landmarks.shape
        if num_pts != 21 or coords != 3:
            raise ValueError(f"Invalid landmarks shape: {original_shape}, expected (..., 21, 3)")

        cache_key = self._make_cache_key(landmarks.tobytes(), "gesture")
        if use_cache and cache_key in self._cache:
            entry = self._cache[cache_key]
            if not entry.is_expired:
                self._cache_hits += 1
                return EmbeddingResult(
                    vector=entry.embedding.copy(),
                    model_name=self.DEFAULT_GESTURE_MODEL,
                    dimension=entry.embedding.shape[0],
                    normalized=self._normalize,
                    processing_time_ms=0.0,
                    metadata={"cached": True, "seq_len": seq_len},
                )

        self._cache_misses += 1
        self._total_requests += 1
        start = time.perf_counter()

        try:
            projection = self._get_gesture_projection()

            # Flatten landmarks: (seq_len, 63)
            flat = landmarks.reshape(seq_len, -1).astype(np.float32)

            # Mean pool over sequence then project
            pooled = flat.mean(axis=0)
            vector = pooled @ projection
            vector = vector.astype(np.float32)

            if self._normalize:
                norm = np.linalg.norm(vector)
                if norm > 1e-8:
                    vector = vector / norm

        except Exception as e:
            logger.error("gesture_embedding_failed", error=str(e))
            raise EmbeddingGenerationError(f"Gesture embedding failed: {e}") from e

        elapsed = (time.perf_counter() - start) * 1000.0

        result = EmbeddingResult(
            vector=vector,
            model_name=self.DEFAULT_GESTURE_MODEL,
            dimension=vector.shape[0],
            normalized=self._normalize,
            processing_time_ms=elapsed,
            metadata={"seq_len": seq_len},
        )

        if use_cache:
            self._add_to_cache(cache_key, vector, self.DEFAULT_GESTURE_MODEL)

        return result

    def generate_multimodal_embedding(
        self,
        text: Optional[str] = None,
        gesture: Optional[np.ndarray] = None,
        emotion: Optional[np.ndarray] = None,
        use_cache: bool = True,
    ) -> EmbeddingResult:
        """Generate fused multimodal embedding from available modalities.

        Args:
            text: Text input for text embedding.
            gesture: Gesture landmarks of shape (21, 3) or (seq_len, 21, 3).
            emotion: Emotion feature vector.
            use_cache: Enable cache lookup.

        Returns:
            EmbeddingResult with fused multimodal embedding.

        Raises:
            EmbeddingGenerationError: If no modalities provided or fusion fails.
        """
        if text is None and gesture is None and emotion is None:
            raise ValueError("At least one modality must be provided")

        cache_parts = []
        if text:
            cache_parts.append(text)
        if gesture is not None:
            cache_parts.append(gesture.tobytes())
        if emotion is not None:
            cache_parts.append(emotion.tobytes())

        cache_key = self._make_cache_key("|".join(cache_parts), "multimodal")
        if use_cache and cache_key in self._cache:
            entry = self._cache[cache_key]
            if not entry.is_expired:
                self._cache_hits += 1
                return EmbeddingResult(
                    vector=entry.embedding.copy(),
                    model_name="multimodal_fusion",
                    dimension=entry.embedding.shape[0],
                    normalized=self._normalize,
                    processing_time_ms=0.0,
                    metadata={"cached": True},
                )

        self._cache_misses += 1
        self._total_requests += 1
        start = time.perf_counter()

        try:
            embeddings: List[np.ndarray] = []
            dims: List[int] = []

            if text:
                text_result = self.generate_text_embedding(text, use_cache=False)
                embeddings.append(text_result.vector)
                dims.append(text_result.dimension)

            if gesture is not None:
                gesture_result = self.generate_gesture_embedding(gesture, use_cache=False)
                embeddings.append(gesture_result.vector)
                dims.append(gesture_result.dimension)

            if emotion is not None:
                emotion_emb = self._embed_emotion(emotion)
                embeddings.append(emotion_emb)
                dims.append(emotion_emb.shape[0])

            if not embeddings:
                raise EmbeddingGenerationError("No embeddings generated from provided modalities")

            if len(embeddings) == 1:
                vector = embeddings[0]
            else:
                vector = self._fuse_embeddings(embeddings)

            if self._normalize:
                norm = np.linalg.norm(vector)
                if norm > 1e-8:
                    vector = vector / norm

        except Exception as e:
            logger.error("multimodal_embedding_failed", error=str(e))
            raise EmbeddingGenerationError(f"Multimodal embedding failed: {e}") from e

        elapsed = (time.perf_counter() - start) * 1000.0

        result = EmbeddingResult(
            vector=vector,
            model_name="multimodal_fusion",
            dimension=vector.shape[0],
            normalized=self._normalize,
            processing_time_ms=elapsed,
            metadata={"modalities_used": len(embeddings), "modality_dims": dims},
        )

        if use_cache:
            self._add_to_cache(cache_key, vector, "multimodal_fusion")

        return result

    def generate_batch(
        self,
        texts: Optional[List[str]] = None,
        landmarks_batch: Optional[List[np.ndarray]] = None,
        batch_size: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[EmbeddingResult]:
        """Generate embeddings for a batch of inputs.

        Args:
            texts: List of text strings to embed.
            landmarks_batch: List of landmark arrays to embed.
            batch_size: Override batch size.
            use_cache: Enable cache lookup.

        Returns:
            List of EmbeddingResult in the same order as inputs.
        """
        results: List[EmbeddingResult] = []
        bs = batch_size or self._batch_size

        if texts:
            for i in range(0, len(texts), bs):
                batch = texts[i : i + bs]
                for text in batch:
                    results.append(self.generate_text_embedding(text, use_cache=use_cache))

        if landmarks_batch:
            for i in range(0, len(landmarks_batch), bs):
                batch = landmarks_batch[i : i + bs]
                for lm in batch:
                    results.append(self.generate_gesture_embedding(lm, use_cache=use_cache))

        return results

    def normalize_embedding(
        self,
        embedding: np.ndarray,
        norm_type: str = "l2",
    ) -> np.ndarray:
        """Normalize an embedding vector in place.

        Args:
            embedding: Input embedding vector.
            norm_type: 'l2', 'unit', or 'minmax'.

        Returns:
            Normalized embedding.
        """
        if norm_type == "l2":
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                return embedding / norm
        elif norm_type == "unit":
            max_abs = np.max(np.abs(embedding))
            if max_abs > 1e-8:
                return embedding / max_abs
        elif norm_type == "minmax":
            emin, emax = embedding.min(), embedding.max()
            if emax - emin > 1e-8:
                return (embedding - emin) / (emax - emin)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
        return embedding

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dict with hit rate, size, etc.
        """
        hit_rate = self._cache_hits / max(self._total_requests, 1)
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_requests": self._total_requests,
            "hit_rate": round(hit_rate, 4),
            "cache_size": len(self._cache),
            "cache_maxsize": self._cache_maxsize,
            "cache_ttl": self._cache_ttl,
        }

    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_requests = 0
        logger.info("embedding_cache_cleared")

    def get_embedding_dimension(self, model_name: Optional[str] = None) -> int:
        """Get the embedding dimension for a model.

        Args:
            model_name: Model name to check.

        Returns:
            Embedding dimension.
        """
        if model_name is None or model_name == self.DEFAULT_TEXT_MODEL:
            return self.EMBEDDING_DIM
        elif model_name == self.DEFAULT_GESTURE_MODEL:
            return self.GESTURE_EMBEDDING_DIM
        elif model_name == self.DEFAULT_EMOTION_MODEL:
            return self.EMOTION_EMBEDDING_DIM
        elif model_name == "multimodal_fusion":
            return self.EMBEDDING_DIM
        return self.EMBEDDING_DIM

    def unload_models(self) -> None:
        """Unload all models to free memory."""
        self._text_model = None
        self._gesture_projection = None
        self._emotion_projection = None
        self._fusion_projection = None
        self.clear_cache()
        logger.info("embedding_models_unloaded")

    def _get_text_model(self, model_name: str) -> SentenceTransformer:
        """Get or load a text embedding model (lazy loading).

        Args:
            model_name: Name of the sentence-transformers model.

        Returns:
            Loaded SentenceTransformer model.
        """
        if self._text_model is not None and model_name == self._text_model_name:
            return self._text_model

        if self._lazy_loading and self._text_model is None:
            logger.info("loading_text_model", model=model_name)
            try:
                self._text_model = SentenceTransformer(
                    model_name,
                    device=self._device,
                    cache_folder=str(self._models_dir),
                )
                logger.info("text_model_loaded", model=model_name)
            except Exception as e:
                logger.error("text_model_load_failed", model=model_name, error=str(e))
                raise ModelNotFoundError(f"Failed to load text model '{model_name}': {e}") from e

        if self._text_model is None:
            raise ModelNotFoundError(f"Text model '{model_name}' not loaded")

        return self._text_model

    def _get_gesture_projection(self) -> np.ndarray:
        """Get or create gesture projection matrix.

        Returns:
            Projection matrix of shape (63, 128).
        """
        if self._gesture_projection is not None:
            return self._gesture_projection

        logger.info("initializing_gesture_projection")
        rng = np.random.RandomState(42)
        projection = rng.randn(63, self.GESTURE_EMBEDDING_DIM).astype(np.float32)
        projection /= np.linalg.norm(projection, axis=0, keepdims=True) + 1e-8
        self._gesture_projection = projection
        return projection

    def _embed_emotion(self, emotion_features: np.ndarray) -> np.ndarray:
        """Project emotion features to embedding space.

        Args:
            emotion_features: Emotion feature vector.

        Returns:
            Emotion embedding vector.
        """
        if self._emotion_projection is None:
            rng = np.random.RandomState(42)
            proj = rng.randn(emotion_features.shape[0], self.EMOTION_EMBEDDING_DIM).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8
            self._emotion_projection = proj

        emb = emotion_features.astype(np.float32) @ self._emotion_projection
        return emb

    def _fuse_embeddings(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Fuse multiple embeddings into a single vector.

        Uses learned weighted averaging via projection.

        Args:
            embeddings: List of embedding vectors.

        Returns:
            Fused embedding vector.
        """
        if len(embeddings) == 0:
            raise ValueError("No embeddings to fuse")

        if len(embeddings) == 1:
            return embeddings[0]

        # Project all to common dimension
        if self._fusion_projection is None:
            target_dim = self.EMBEDDING_DIM
            projections = []
            for emb in embeddings:
                rng = np.random.RandomState(hash(emb.shape[0]) % (2**31))
                proj = rng.randn(emb.shape[0], target_dim).astype(np.float32)
                projections.append(proj)
            self._fusion_projection = projections

        projected = []
        for i, emb in enumerate(embeddings):
            p = emb @ self._fusion_projection[i] if self._fusion_projection else emb
            projected.append(p)

        # Weighted average based on embedding norms
        weights = np.array([np.linalg.norm(e) for e in embeddings])
        weights = weights / (weights.sum() + 1e-8)

        fused = np.sum(
            [w * p for w, p in zip(weights, projected)],
            axis=0,
        )

        return fused.astype(np.float32)

    def _make_cache_key(self, data: str, prefix: str) -> str:
        """Create a deterministic cache key from input data.

        Args:
            data: Input data string.
            prefix: Cache key prefix.

        Returns:
            SHA-256 hash key.
        """
        raw = f"{prefix}:{data}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _add_to_cache(self, key: str, embedding: np.ndarray, model_name: str) -> None:
        """Add an embedding to the LRU cache.

        Args:
            key: Cache key.
            embedding: Embedding vector.
            model_name: Model name that generated it.
        """
        if len(self._cache) >= self._cache_maxsize:
            # Evict oldest entry
            oldest = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
            del self._cache[oldest]

        self._cache[key] = CacheEntry(
            embedding=embedding.copy(),
            model_name=model_name,
            created_at=time.monotonic(),
            ttl=self._cache_ttl,
        )

    @staticmethod
    def _has_torch_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
