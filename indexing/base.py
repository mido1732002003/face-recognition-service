from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np


class VectorIndex(ABC):
    """Abstract base class for vector similarity index"""

    @abstractmethod
    async def add(self, embeddings: np.ndarray, ids: list[int]) -> None:
        """Add embeddings to index with corresponding IDs"""
        pass

    @abstractmethod
    async def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors
        Returns: (distances, indices)
        """
        pass

    @abstractmethod
    async def remove(self, ids: list[int]) -> None:
        """Remove embeddings by ID"""
        pass

    @abstractmethod
    async def save(self, path: str) -> None:
        """Save index to disk"""
        pass

    @abstractmethod
    async def load(self, path: str) -> None:
        """Load index from disk"""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all embeddings from index"""
        pass

    @abstractmethod
    async def rebuild(self) -> None:
        """Rebuild/optimize index"""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get number of embeddings in index"""
        pass

    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension"""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get index statistics"""
        pass


class IndexConfig:
    """Configuration for vector index"""

    def __init__(
        self,
        dimension: int = 512,
        metric: str = "cosine",
        index_type: str = "flat",
        **kwargs,
    ):
        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type
        self.extra_params = kwargs