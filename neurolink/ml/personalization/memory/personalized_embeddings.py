"""
Personalized embeddings module for Neurolink.

Provides the PersonalizedEmbedding class for generating, caching, and
retrieving user-specific embeddings using contrastive learning with
LRU cache eviction and similarity search.
"""

import json
import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for personalized embeddings."""

    embedding_dim: int = 256
    projection_dim: int = 128
    hidden_dim: int = 512

    # Contrastive learning
    temperature: float = 0.07
    contrastive_margin: float = 0.5
    use_supervised_contrastive: bool = True

    # Cache
    cache_capacity: int = 10000
    lru_eviction: bool = True

    # Similarity search
    similarity_metric: str = "cosine"  # "cosine", "euclidean", "dot"
    top_k: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ContrastiveEmbeddingNet(nn.Module):
    """Neural network for learning user-specific embeddings via contrastive learning."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        projection_dim: int = 128,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(
        self, x: torch.Tensor, return_projection: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode input to embedding.

        Args:
            x: (batch, input_dim) input features.
            return_projection: Whether to return projection head output.

        Returns:
            embedding: (batch, embedding_dim)
            (optionally) projection: (batch, projection_dim)
        """
        embedding = self.encoder(x)
        embedding = F.normalize(embedding, dim=-1)

        if return_projection:
            projection = self.projection(embedding)
            projection = F.normalize(projection, dim=-1)
            return embedding, projection

        return embedding


class LRUCache:
    """LRU (Least Recently Used) cache with fixed capacity."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._access_count: int = 0
        logger.info(f"LRU cache initialized (capacity={capacity})")

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache and move to end (most recently used)."""
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        self._access_count += 1
        return self._cache[key]

    def put(self, key: str, value: Any):
        """Put item in cache, evicting LRU if at capacity."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug(f"Evicted LRU entry: {evicted_key}")

    def remove(self, key: str):
        """Remove item from cache."""
        self._cache.pop(key, None)

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache


class PersonalizedEmbedding:
    """Generate, cache, and retrieve personalized embeddings for users.

    Uses contrastive learning to produce user-specific features that
    are stable across sessions. Maintains an LRU cache of embeddings
    for fast retrieval.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.device = torch.device(self.config.device)

        # Embedding network
        self.embedding_net: Optional[ContrastiveEmbeddingNet] = None

        # LRU cache
        self.cache = LRUCache(capacity=self.config.cache_capacity)

        # Embedding storage for similarity search
        self._embeddings: Dict[str, torch.Tensor] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

        # User ID mapping
        self._user_ids: Set[str] = set()

        logger.info(
            f"PersonalizedEmbedding initialized "
            f"(dim={config.embedding_dim if config else 256}, "
            f"cache={config.cache_capacity if config else 10000})"
        )

    def build_network(
        self, input_dim: int
    ) -> ContrastiveEmbeddingNet:
        """Build and return a new embedding network."""
        net = ContrastiveEmbeddingNet(
            input_dim=input_dim,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            projection_dim=self.config.projection_dim,
        )
        return net.to(self.device)

    def generate_user_embedding(
        self,
        user_id: str,
        features: torch.Tensor,
        update_cache: bool = True,
    ) -> torch.Tensor:
        """Generate or retrieve a user embedding.

        First checks the cache. If not found, computes the embedding
        from features and caches it.

        Args:
            user_id: Unique user identifier.
            features: (N, input_dim) feature tensor for the user.
            update_cache: Whether to update the LRU cache.

        Returns:
            embedding: (embedding_dim,) normalized embedding vector.
        """
        # Check cache first
        if user_id in self.cache:
            logger.debug(f"Cache hit for user {user_id}")
            embedding = self.cache.get(user_id)
            if embedding is not None:
                return embedding

        # Compute embedding
        if self.embedding_net is None:
            raise RuntimeError("Embedding network not initialized")

        with torch.no_grad():
            features = features.to(self.device)
            if features.dim() == 1:
                features = features.unsqueeze(0)
            embedding, _ = self.embedding_net(features, return_projection=True)
            # Use mean pooling over the batch
            embedding = embedding.mean(dim=0)  # (embedding_dim,)

        # Update cache and storage
        self._user_ids.add(user_id)
        self._embeddings[user_id] = embedding.cpu()

        if update_cache:
            self.cache.put(user_id, embedding.cpu())

        logger.debug(f"Generated embedding for user {user_id}")
        return embedding.cpu()

    def compute_contrastive_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Compute supervised contrastive loss.

        Args:
            anchor: (batch, projection_dim) anchor embeddings.
            positive: (batch, projection_dim) positive embeddings.
            negatives: (batch, num_negatives, projection_dim) negative embeddings.

        Returns:
            contrastive_loss scalar.
        """
        batch_size = anchor.size(0)
        num_negatives = negatives.size(1)

        # Positive similarity
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1)  # (batch,)
        pos_sim = pos_sim / self.config.temperature

        # Negative similarity
        anchor_expanded = anchor.unsqueeze(1).expand(
            -1, num_negatives, -1
        )  # (batch, num_neg, proj_dim)
        neg_sim = F.cosine_similarity(
            anchor_expanded, negatives, dim=-1
        )  # (batch, num_neg)
        neg_sim = neg_sim / self.config.temperature

        # InfoNCE loss
        pos_exp = torch.exp(pos_sim)  # (batch,)
        neg_exp = torch.exp(neg_sim).sum(dim=1)  # (batch,)
        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8))

        return loss.mean()

    def supervised_contrastive_loss(
        self,
        projections: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Supervised contrastive loss with hard positive/negative mining.

        Args:
            projections: (batch, projection_dim) normalized projections.
            labels: (batch,) class labels.

        Returns:
            Loss scalar.
        """
        batch_size = projections.size(0)
        temperature = self.config.temperature

        # Similarity matrix
        sim = projections @ projections.T / temperature  # (batch, batch)

        # Mask out self-contrast
        labels_expanded = labels.unsqueeze(0).expand(batch_size, -1)
        pos_mask = (labels_expanded == labels_expanded.T).float()
        pos_mask.fill_diagonal_(0)

        neg_mask = 1.0 - pos_mask
        neg_mask.fill_diagonal_(0)

        # Positive and negative pairs
        pos_sim = (sim * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        neg_sim = (sim * neg_mask).sum(dim=1) / (neg_mask.sum(dim=1) + 1e-8)

        loss = -torch.log(
            torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-8)
        )
        return loss.mean()

    def similarity_search(
        self,
        query_embedding: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Search for most similar users in embedding space.

        Args:
            query_embedding: (embedding_dim,) query embedding vector.
            top_k: Number of results to return (default: config value).

        Returns:
            List of (user_id, similarity_score) sorted by descending similarity.
        """
        k = top_k or self.config.top_k
        query = F.normalize(query_embedding, dim=-1).to(self.device)

        scores: List[Tuple[str, float]] = []
        for user_id, emb in self._embeddings.items():
            emb = emb.to(self.device)
            emb = F.normalize(emb, dim=-1)

            if self.config.similarity_metric == "cosine":
                score = F.cosine_similarity(query.unsqueeze(0), emb.unsqueeze(0)).item()
            elif self.config.similarity_metric == "euclidean":
                score = -float(torch.norm(query - emb).item())
            elif self.config.similarity_metric == "dot":
                score = float((query * emb).sum().item())
            else:
                raise ValueError(f"Unknown metric: {self.config.similarity_metric}")

            scores.append((user_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def update_embedding(
        self,
        user_id: str,
        features: torch.Tensor,
        momentum: float = 0.9,
    ):
        """Update a user's embedding with new features (online update).

        Uses exponential moving average to smooth embedding updates.

        Args:
            user_id: Unique user identifier.
            features: (N, input_dim) new feature tensor.
            momentum: EMA decay factor (higher = slower update).
        """
        if self.embedding_net is None:
            raise RuntimeError("Embedding network not initialized")

        with torch.no_grad():
            features = features.to(self.device)
            if features.dim() == 1:
                features = features.unsqueeze(0)
            new_emb, _ = self.embedding_net(features, return_projection=True)
            new_emb = new_emb.mean(dim=0)  # (embedding_dim,)

        if user_id in self._embeddings:
            old_emb = self._embeddings[user_id].to(self.device)
            updated = momentum * old_emb + (1.0 - momentum) * new_emb
            updated = F.normalize(updated, dim=-1)
            self._embeddings[user_id] = updated.cpu()
            self.cache.put(user_id, updated.cpu())
        else:
            self._embeddings[user_id] = new_emb.cpu()
            self.cache.put(user_id, new_emb.cpu())

        logger.debug(f"Updated embedding for user {user_id} (momentum={momentum})")

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all stored embeddings as numpy arrays."""
        return {
            uid: emb.numpy() for uid, emb in self._embeddings.items()
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "cache_capacity": self.cache.capacity,
            "unique_users": len(self._user_ids),
            "embeddings_stored": len(self._embeddings),
        }

    def save(self, path: Union[str, Path]):
        """Save embeddings and metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save embedding network state
        if self.embedding_net is not None:
            torch.save(
                self.embedding_net.state_dict(), path / "embedding_net.pt"
            )

        # Save embeddings
        embeddings_data = {
            uid: emb.numpy().tolist()
            for uid, emb in self._embeddings.items()
        }
        metadata = {
            "config": {
                k: v for k, v in self.config.__dict__.items()
                if not k.startswith("_")
            },
            "embeddings": embeddings_data,
        }
        with open(path / "embeddings.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Saved {len(self._embeddings)} embeddings to {path}"
        )

    def load(self, path: Union[str, Path]):
        """Load embeddings and metadata from disk."""
        path = Path(path)

        config_path = path / "embeddings.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                metadata = json.load(f)
            config_dict = metadata.get("config", {})
            if config_dict:
                self.config = EmbeddingConfig(**config_dict)

            for uid, emb_list in metadata.get("embeddings", {}).items():
                self._embeddings[uid] = torch.tensor(emb_list, dtype=torch.float32)
                self._user_ids.add(uid)

        net_path = path / "embedding_net.pt"
        if net_path.exists():
            input_dim = self.config.hidden_dim
            self.embedding_net = ContrastiveEmbeddingNet(
                input_dim=input_dim,
                embedding_dim=self.config.embedding_dim,
                hidden_dim=self.config.hidden_dim,
                projection_dim=self.config.projection_dim,
            ).to(self.device)
            self.embedding_net.load_state_dict(
                torch.load(net_path, map_location=self.device)
            )

        logger.info(
            f"Loaded {len(self._embeddings)} embeddings from {path}"
        )
