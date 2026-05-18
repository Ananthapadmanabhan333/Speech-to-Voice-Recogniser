from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.api.types import QueryResult
except ImportError:
    chromadb = None

logger = structlog.get_logger(__name__)


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


class CollectionNotFoundError(VectorStoreError):
    """Raised when a collection does not exist."""


class EmbeddingDimensionMismatchError(VectorStoreError):
    """Raised when embedding dimension does not match collection."""


@dataclass
class SearchResult:
    """Result of a similarity search."""

    id: str
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]
    distance: float
    score: float


class VectorStore:
    """ChromaDB-based vector store for embedding storage and retrieval.

    Provides a robust wrapper around ChromaDB with retry logic, health
    checks, batch operations, and collection management.

    Features:
    - Embedding storage with metadata
    - Similarity search with filters
    - Collection CRUD
    - Batch operations with progress tracking
    - Automatic retry with exponential backoff
    - Health check and diagnostics
    - Configurable distance metrics
    """

    DEFAULT_COLLECTION_NAME: str = "neurolink_memory"
    DEFAULT_DISTANCE_METRIC: str = "cosine"
    MAX_RETRIES: int = 3
    RETRY_BACKOFF: float = 0.5

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_dim: int = 384,
        distance_metric: str = DEFAULT_DISTANCE_METRIC,
        persist_directory: Optional[str] = None,
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None,
        batch_size: int = 100,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        """Initialize the vector store.

        Args:
            collection_name: Default collection name.
            embedding_dim: Expected embedding dimension.
            distance_metric: Distance metric ('cosine', 'l2', 'ip').
            persist_directory: Local directory for persistence.
            chroma_host: ChromaDB server host (for client-server mode).
            chroma_port: ChromaDB server port.
            batch_size: Batch size for bulk operations.
            max_retries: Maximum retry attempts for failed operations.

        Raises:
            VectorStoreError: If ChromaDB is not installed.
        """
        if chromadb is None:
            raise VectorStoreError(
                "chromadb is not installed. Install with: pip install chromadb"
            )

        self._collection_name = collection_name
        self._embedding_dim = embedding_dim
        self._distance_metric = distance_metric
        self._batch_size = batch_size
        self._max_retries = max_retries

        self._client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None
        self._initialized = False

        # Statistics
        self._total_stored: int = 0
        self._total_searched: int = 0
        self._total_deleted: int = 0
        self._errors: int = 0

        try:
            if chroma_host and chroma_port:
                self._client = chromadb.HttpClient(
                    host=chroma_host,
                    port=chroma_port,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                logger.info("connecting_to_chroma_server", host=chroma_host, port=chroma_port)
            else:
                self._client = chromadb.Client(
                    settings=ChromaSettings(
                        persist_directory=persist_directory,
                        anonymized_telemetry=False,
                    )
                )
                logger.info("chroma_client_initialized", persist_directory=persist_directory)

            self._ensure_collection()
            self._initialized = True
            logger.info(
                "vector_store_initialized",
                collection=collection_name,
                dimension=embedding_dim,
                metric=distance_metric,
            )

        except Exception as e:
            logger.error("vector_store_init_failed", error=str(e))
            raise VectorStoreError(f"Failed to initialize vector store: {e}") from e

    def store_embedding(
        self,
        collection: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> str:
        """Store a single embedding in the specified collection.

        Args:
            collection: Collection name.
            embedding: Embedding vector.
            metadata: Optional metadata dict.
            id: Optional document ID (auto-generated if not provided).

        Returns:
            Document ID of the stored embedding.

        Raises:
            VectorStoreError: If storage fails.
            EmbeddingDimensionMismatchError: If dimension doesn't match.
        """
        self._validate_embedding(embedding)

        import uuid
        doc_id = id or str(uuid.uuid4())

        def _store() -> None:
            col = self._get_or_create_collection(collection)
            col.add(
                embeddings=[embedding.tolist()],
                metadatas=[metadata or {}],
                ids=[doc_id],
            )

        self._retry_operation(_store)
        self._total_stored += 1
        return doc_id

    def search_similar(
        self,
        collection: str,
        embedding: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar embeddings in a collection.

        Args:
            collection: Collection name.
            embedding: Query embedding vector.
            k: Number of results to return.
            filters: Metadata filters (e.g., {'user_id': 'abc'}).

        Returns:
            List of SearchResult ordered by similarity (highest first).

        Raises:
            CollectionNotFoundError: If collection doesn't exist.
            VectorStoreError: If search fails.
        """
        self._validate_embedding(embedding)

        def _search() -> QueryResult:
            col = self._get_or_create_collection(collection)
            return col.query(
                query_embeddings=[embedding.tolist()],
                n_results=k,
                where=filters,
                include=["embeddings", "metadatas", "distances"],
            )

        try:
            result = self._retry_operation(_search)
        except CollectionNotFoundError:
            return []

        self._total_searched += 1
        return self._parse_query_result(result)

    def delete_embedding(
        self,
        collection: str,
        id: str,
    ) -> bool:
        """Delete an embedding by ID from a collection.

        Args:
            collection: Collection name.
            id: Document ID to delete.

        Returns:
            True if deleted, False if not found.

        Raises:
            VectorStoreError: If deletion fails.
        """
        def _delete() -> None:
            col = self._get_or_create_collection(collection)
            try:
                col.delete(ids=[id])
            except Exception as e:
                if "not found" in str(e).lower():
                    return
                raise

        try:
            self._retry_operation(_delete)
            self._total_deleted += 1
            return True
        except Exception:
            return False

    def update_embedding(
        self,
        collection: str,
        id: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an existing embedding and/or metadata.

        Args:
            collection: Collection name.
            id: Document ID to update.
            embedding: New embedding vector (None to keep existing).
            metadata: New metadata dict (None to keep existing).

        Returns:
            True if updated, False if not found.

        Raises:
            VectorStoreError: If update fails.
        """
        if embedding is not None:
            self._validate_embedding(embedding)

        def _update() -> None:
            col = self._get_or_create_collection(collection)
            col.update(
                ids=[id],
                embeddings=[embedding.tolist()] if embedding is not None else None,
                metadatas=[metadata] if metadata is not None else None,
            )

        try:
            self._retry_operation(_update)
            return True
        except Exception:
            return False

    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a new collection.

        Args:
            name: Collection name.
            metadata: Optional collection metadata.

        Returns:
            True if created, False if already exists.

        Raises:
            VectorStoreError: If creation fails.
        """
        try:
            self._client.create_collection(
                name=name,
                metadata=metadata or {},
            )
            logger.info("collection_created", name=name)
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                return False
            raise VectorStoreError(f"Failed to create collection '{name}': {e}") from e

    def list_collections(self) -> List[str]:
        """List all collection names.

        Returns:
            List of collection name strings.
        """
        collections = self._client.list_collections()
        return [c.name for c in collections]

    def delete_collection(self, name: str) -> bool:
        """Delete a collection and all its data.

        Args:
            name: Collection name.

        Returns:
            True if deleted, False if not found.
        """
        try:
            self._client.delete_collection(name)
            logger.info("collection_deleted", name=name)
            return True
        except Exception:
            return False

    def get_collection_info(self, name: str) -> Dict[str, Any]:
        """Get metadata about a collection.

        Args:
            name: Collection name.

        Returns:
            Dict with collection metadata and count.

        Raises:
            CollectionNotFoundError: If collection doesn't exist.
        """
        try:
            col = self._client.get_collection(name)
            count = col.count()
            return {
                "name": name,
                "count": count,
                "metadata": col.metadata,
            }
        except Exception as e:
            raise CollectionNotFoundError(f"Collection '{name}' not found: {e}") from e

    def store_batch(
        self,
        collection: str,
        embeddings: List[np.ndarray],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Store multiple embeddings in a batch.

        Args:
            collection: Collection name.
            embeddings: List of embedding vectors.
            metadatas: List of metadata dicts (same length).
            ids: List of document IDs (auto-generated if None).

        Returns:
            List of stored document IDs.

        Raises:
            VectorStoreError: If batch storage fails.
        """
        if not embeddings:
            return []

        for emb in embeddings:
            self._validate_embedding(emb)

        import uuid
        doc_ids = ids or [str(uuid.uuid4()) for _ in range(len(embeddings))]
        metas = metadatas or [{} for _ in range(len(embeddings))]

        if len(embeddings) != len(metas) or len(embeddings) != len(doc_ids):
            raise ValueError("embeddings, metadatas, and ids must have the same length")

        def _store_batch_subset(
            start: int,
            end: int,
        ) -> None:
            col = self._get_or_create_collection(collection)
            col.add(
                embeddings=[e.tolist() for e in embeddings[start:end]],
                metadatas=metas[start:end],
                ids=doc_ids[start:end],
            )

        for i in range(0, len(embeddings), self._batch_size):
            end = min(i + self._batch_size, len(embeddings))
            self._retry_operation(lambda: _store_batch_subset(i, end))
            self._total_stored += end - i

        return doc_ids

    def search_batch(
        self,
        collection: str,
        query_embeddings: List[np.ndarray],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[List[SearchResult]]:
        """Search multiple query embeddings in a batch.

        Args:
            collection: Collection name.
            query_embeddings: List of query embeddings.
            k: Results per query.
            filters: Metadata filters.

        Returns:
            List of result lists, one per query.
        """
        if not query_embeddings:
            return []

        for emb in query_embeddings:
            self._validate_embedding(emb)

        def _search() -> QueryResult:
            col = self._get_or_create_collection(collection)
            return col.query(
                query_embeddings=[e.tolist() for e in query_embeddings],
                n_results=k,
                where=filters,
                include=["embeddings", "metadatas", "distances"],
            )

        try:
            result = self._retry_operation(_search)
        except CollectionNotFoundError:
            return [[] for _ in query_embeddings]

        self._total_searched += len(query_embeddings)

        all_results: List[List[SearchResult]] = []
        if result and result["ids"]:
            for i in range(len(query_embeddings)):
                all_results.append(self._parse_single_result(result, i))
        return all_results

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector store.

        Returns:
            Dict with status, counts, and diagnostics.
        """
        status = "healthy"
        details: Dict[str, Any] = {
            "initialized": self._initialized,
            "collections": [],
            "total_stored": self._total_stored,
            "total_searched": self._total_searched,
            "total_deleted": self._total_deleted,
            "errors": self._errors,
            "embedding_dim": self._embedding_dim,
            "distance_metric": self._distance_metric,
        }

        try:
            if self._initialized and self._client:
                collections = self.list_collections()
                details["collections"] = collections
                details["collection_count"] = len(collections)

                for col_name in collections:
                    try:
                        info = self.get_collection_info(col_name)
                        details[f"collection_{col_name}_count"] = info["count"]
                    except Exception:
                        pass
        except Exception as e:
            status = "degraded"
            details["error"] = str(e)

        return {"status": status, **details}

    def count(self, collection: Optional[str] = None) -> int:
        """Count embeddings in a collection.

        Args:
            collection: Collection name (defaults to primary).

        Returns:
            Number of embeddings.
        """
        name = collection or self._collection_name
        try:
            col = self._get_or_create_collection(name)
            return col.count()
        except Exception:
            return 0

    def get(self, collection: str, id: str) -> Optional[Dict[str, Any]]:
        """Get an embedding by ID.

        Args:
            collection: Collection name.
            id: Document ID.

        Returns:
            Dict with embedding and metadata, or None.
        """
        try:
            col = self._get_or_create_collection(collection)
            result = col.get(ids=[id], include=["embeddings", "metadatas"])
            if result and result["ids"]:
                return {
                    "id": result["ids"][0],
                    "embedding": np.array(result["embeddings"][0]) if result.get("embeddings") else None,
                    "metadata": result["metadatas"][0] if result.get("metadatas") else {},
                }
        except Exception:
            pass
        return None

    def _ensure_collection(self) -> None:
        """Ensure default collection exists."""
        try:
            self._collection = self._client.get_collection(self._collection_name)
        except Exception:
            self._collection = self._client.create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": self._distance_metric},
            )

    def _get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get or create a collection by name.

        Args:
            name: Collection name.

        Returns:
            ChromaDB Collection instance.
        """
        try:
            return self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": self._distance_metric},
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to get/create collection '{name}': {e}") from e

    def _validate_embedding(self, embedding: np.ndarray) -> None:
        """Validate embedding shape and type.

        Args:
            embedding: Embedding vector.

        Raises:
            EmbeddingDimensionMismatchError: If dimension is wrong.
        """
        if embedding is None or embedding.size == 0:
            raise ValueError("Embedding cannot be empty")

        embedding = np.asarray(embedding)
        if embedding.ndim > 1:
            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding.flatten()
            else:
                raise ValueError(f"Embedding must be 1D, got shape {embedding.shape}")

        if embedding.shape[0] != self._embedding_dim:
            raise EmbeddingDimensionMismatchError(
                f"Expected dimension {self._embedding_dim}, got {embedding.shape[0]}"
            )

    def _parse_query_result(self, result: QueryResult) -> List[SearchResult]:
        """Parse ChromaDB query result into SearchResult list.

        Args:
            result: Raw ChromaDB query result.

        Returns:
            List of SearchResult.
        """
        parsed: List[SearchResult] = []
        if not result or not result.get("ids") or not result["ids"]:
            return parsed

        for i in range(len(result["ids"][0])):
            distance = float(result["distances"][0][i]) if result.get("distances") else 0.0
            score = 1.0 - distance  # Convert distance to similarity

            emb = None
            if result.get("embeddings"):
                emb_list = result["embeddings"][0][i]
                if emb_list is not None:
                    emb = np.array(emb_list, dtype=np.float32)

            parsed.append(SearchResult(
                id=result["ids"][0][i],
                embedding=emb,
                metadata=result["metadatas"][0][i] if result.get("metadatas") else {},
                distance=distance,
                score=score,
            ))

        parsed.sort(key=lambda r: r.score, reverse=True)
        return parsed

    def _parse_single_result(self, result: QueryResult, idx: int) -> List[SearchResult]:
        """Parse a single query's results from a batch.

        Args:
            result: Raw ChromaDB query result.
            idx: Index into the batch.

        Returns:
            List of SearchResult for the idx-th query.
        """
        parsed: List[SearchResult] = []
        if not result or not result.get("ids") or idx >= len(result["ids"]):
            return parsed

        for i in range(len(result["ids"][idx])):
            distance = float(result["distances"][idx][i]) if result.get("distances") else 0.0
            score = 1.0 - distance

            emb = None
            if result.get("embeddings"):
                emb_list = result["embeddings"][idx][i]
                if emb_list is not None:
                    emb = np.array(emb_list, dtype=np.float32)

            parsed.append(SearchResult(
                id=result["ids"][idx][i],
                embedding=emb,
                metadata=result["metadatas"][idx][i] if result.get("metadatas") else {},
                distance=distance,
                score=score,
            ))

        parsed.sort(key=lambda r: r.score, reverse=True)
        return parsed

    def _retry_operation(self, operation: callable, retries: Optional[int] = None) -> Any:
        """Execute an operation with retry and exponential backoff.

        Args:
            operation: Callable to execute.
            retries: Override max retries.

        Returns:
            Operation result.

        Raises:
            VectorStoreError: If all retries fail.
        """
        max_retries = retries or self._max_retries
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                return operation()
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    backoff = self.RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "operation_retry",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        backoff=backoff,
                        error=str(e),
                    )
                    time.sleep(backoff)
                else:
                    self._errors += 1
                    logger.error(
                        "operation_failed",
                        attempts=attempt + 1,
                        error=str(e),
                    )

        raise VectorStoreError(f"Operation failed after {max_retries + 1} attempts: {last_exception}") from last_exception
