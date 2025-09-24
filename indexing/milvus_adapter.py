from typing import Any, Optional, Tuple

import numpy as np

from indexing.base import IndexConfig, VectorIndex
from utils.logging import get_logger

logger = get_logger(__name__)


class MilvusAdapter(VectorIndex):
    """Milvus vector database adapter"""

    def __init__(self, config: IndexConfig):
        self.config = config
        self.dimension = config.dimension
        self.collection_name = config.extra_params.get("collection", "face_embeddings")
        logger.warning("Milvus adapter is a stub. Install pymilvus to use.")
        
        # To use Milvus:
        # pip install pymilvus
        # from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
        
        # Example connection:
        # connections.connect(
        #     alias="default",
        #     host=config.extra_params.get("host", "localhost"),
        #     port=config.extra_params.get("port", 19530)
        # )
        
        # Example schema:
        # fields = [
        #     FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        #     FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
        # ]
        # schema = CollectionSchema(fields=fields)
        # self.collection = Collection(name=self.collection_name, schema=schema)
        
        # Create index:
        # index_params = {
        #     "metric_type": "IP",  # Inner product for cosine with normalized vectors
        #     "index_type": "IVF_FLAT",
        #     "params": {"nlist": 1024}
        # }
        # self.collection.create_index(field_name="embedding", index_params=index_params)

    async def add(self, embeddings: np.ndarray, ids: list[int]) -> None:
        """Add embeddings to Milvus"""
        # entities = [ids, embeddings.tolist()]
        # self.collection.insert(entities)
        # self.collection.flush()
        logger.info(f"Milvus add operation - {len(ids)} embeddings")

    async def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search in Milvus"""
        # search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        # results = self.collection.search(
        #     data=[query_embedding.tolist()],
        #     anns_field="embedding",
        #     param=search_params,
        #     limit=k,
        #     output_fields=["id"]
        # )
        # distances = np.array([hit.score for hit in results[0]])
        # ids = np.array([hit.id for hit in results[0]])
        return np.array([]), np.array([])

    async def remove(self, ids: list[int]) -> None:
        """Remove embeddings from Milvus"""
        # expr = f"id in {ids}"
        # self.collection.delete(expr)
        logger.info(f"Milvus remove operation - {len(ids)} embeddings")

    async def save(self, path: str) -> None:
        """Milvus persists automatically"""
        pass

    async def load(self, path: str) -> None:
        """Load collection"""
        # self.collection.load()
        pass

    async def clear(self) -> None:
        """Clear collection"""
        # self.collection.drop()
        pass

    async def rebuild(self) -> None:
        """Rebuild index"""
        # self.collection.compact()
        pass

    def size(self) -> int:
        """Get collection size"""
        # return self.collection.num_entities
        return 0

    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension

    def get_stats(self) -> dict[str, Any]:
        """Get Milvus statistics"""
        return {
            "type": "milvus",
            "collection": self.collection_name,
            "size": 0,
            "dimension": self.dimension,
            "metric": "cosine",
            "note": "Milvus adapter not implemented - install pymilvus",
        }