from typing import Any, Optional, Tuple

import numpy as np

from indexing.base import IndexConfig, VectorIndex
from utils.logging import get_logger

logger = get_logger(__name__)


class QdrantAdapter(VectorIndex):
    """Qdrant vector database adapter"""

    def __init__(self, config: IndexConfig):
        self.config = config
        self.dimension = config.dimension
        self.collection_name = config.extra_params.get("collection", "face_embeddings")
        logger.warning("Qdrant adapter is a stub. Install qdrant-client to use.")
        
        # To use Qdrant:
        # pip install qdrant-client
        # from qdrant_client import QdrantClient
        # from qdrant_client.models import Distance, VectorParams, PointStruct
        
        # self.client = QdrantClient(
        #     host=config.extra_params.get("host", "localhost"),
        #     port=config.extra_params.get("port", 6333)
        # )
        
        # # Create collection
        # self.client.recreate_collection(
        #     collection_name=self.collection_name,
        #     vectors_config=VectorParams(
        #         size=self.dimension,
        #         distance=Distance.COSINE
        #     )
        # )

    async def add(self, embeddings: np.ndarray, ids: list[int]) -> None:
        """Add embeddings to Qdrant"""
        # points = [
        #     PointStruct(
        #         id=idx,
        #         vector=embedding.tolist(),
        #         payload={"original_id": idx}
        #     )
        #     for idx, embedding in zip(ids, embeddings)
        # ]
        # self.client.upsert(
        #     collection_name=self.collection_name,
        #     points=points
        # )
        logger.info(f"Qdrant add operation - {len(ids)} embeddings")

    async def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search in Qdrant"""
        # results = self.client.search(
        #     collection_name=self.collection_name,
        #     query_vector=query_embedding.tolist(),
        #     limit=k
        # )
        # distances = np.array([r.score for r in results])
        # ids = np.array([r.id for r in results])
        return np.array([]), np.array([])

    async def remove(self, ids: list[int]) -> None:
        """Remove embeddings from Qdrant"""
        # self.client.delete(
        #     collection_name=self.collection_name,
        #     points_selector=ids
        # )
        logger.info(f"Qdrant remove operation - {len(ids)} embeddings")

    async def save(self, path: str) -> None:
        """Qdrant persists automatically"""
        pass

    async def load(self, path: str) -> None:
        """Load collection"""
        pass

    async def clear(self) -> None:
        """Clear collection"""
        # self.client.delete_collection(self.collection_name)
        pass

    async def rebuild(self) -> None:
        """Rebuild index"""
        pass

    def size(self) -> int:
        """Get collection size"""
        # info = self.client.get_collection(self.collection_name)
        # return info.points_count
        return 0

    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension

    def get_stats(self) -> dict[str, Any]:
        """Get Qdrant statistics"""
        return {
            "type": "qdrant",
            "collection": self.collection_name,
            "size": 0,
            "dimension": self.dimension,
            "metric": "cosine",
            "note": "Qdrant adapter not implemented - install qdrant-client",
        }