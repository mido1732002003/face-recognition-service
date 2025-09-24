import asyncio
import os
import pickle
from typing import Any, Optional, Tuple

import faiss
import numpy as np

from api.config import settings
from indexing.base import IndexConfig, VectorIndex
from utils.logging import get_logger
from utils.metrics import INDEX_ADD_DURATION, INDEX_SEARCH_DURATION, INDEX_SIZE

logger = get_logger(__name__)


class FaissIndexFlat(VectorIndex):
    """FAISS flat index implementation for exact cosine similarity search"""

    def __init__(self, config: IndexConfig):
        self.config = config
        self.dimension = config.dimension
        
        # For cosine similarity with L2-normalized vectors, use IndexFlatIP
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Mapping from FAISS internal IDs to our IDs
        self.id_map: dict[int, int] = {}
        self.reverse_id_map: dict[int, int] = {}
        self._next_internal_id = 0

    async def add(self, embeddings: np.ndarray, ids: list[int]) -> None:
        """Add embeddings to index"""
        start_time = asyncio.get_event_loop().time()
        
        # Ensure embeddings are L2-normalized for cosine similarity
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Add to index
        await asyncio.to_thread(self.index.add, embeddings)
        
        # Update ID mappings
        for i, external_id in enumerate(ids):
            internal_id = self._next_internal_id + i
            self.id_map[internal_id] = external_id
            self.reverse_id_map[external_id] = internal_id
        
        self._next_internal_id += len(ids)
        
        # Update metrics
        INDEX_SIZE.set(self.index.ntotal)
        duration = asyncio.get_event_loop().time() - start_time
        INDEX_ADD_DURATION.observe(duration)
        
        logger.info(f"Added {len(ids)} embeddings to index", duration_ms=duration * 1000)

    async def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors"""
        start_time = asyncio.get_event_loop().time()
        
        # Ensure query is L2-normalized
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = await asyncio.to_thread(
            self.index.search, query_embedding, k
        )
        
        # Map internal IDs to external IDs
        external_ids = np.array([
            self.id_map.get(int(idx), -1) for idx in indices[0]
        ])
        
        # Filter out invalid IDs
        valid_mask = external_ids >= 0
        distances = distances[0][valid_mask]
        external_ids = external_ids[valid_mask]
        
        duration = asyncio.get_event_loop().time() - start_time
        INDEX_SEARCH_DURATION.observe(duration)
        
        return distances, external_ids

    async def remove(self, ids: list[int]) -> None:
        """Remove embeddings by ID (not efficiently supported by flat index)"""
        # Flat index doesn't support efficient removal
        # Would need to rebuild entire index
        logger.warning("Remove operation not supported for flat index, consider using IVF")
        
        # Update mappings
        for external_id in ids:
            if external_id in self.reverse_id_map:
                internal_id = self.reverse_id_map[external_id]
                del self.id_map[internal_id]
                del self.reverse_id_map[external_id]

    async def save(self, path: str) -> None:
        """Save index to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        index_path = f"{path}.index"
        await asyncio.to_thread(faiss.write_index, self.index, index_path)
        
        # Save ID mappings
        mapping_path = f"{path}.mapping"
        with open(mapping_path, "wb") as f:
            pickle.dump({
                "id_map": self.id_map,
                "reverse_id_map": self.reverse_id_map,
                "next_internal_id": self._next_internal_id,
            }, f)
        
        logger.info(f"Saved index to {path}")

    async def load(self, path: str) -> None:
        """Load index from disk"""
        # Load FAISS index
        index_path = f"{path}.index"
        if os.path.exists(index_path):
            self.index = await asyncio.to_thread(faiss.read_index, index_path)
            
            # Load ID mappings
            mapping_path = f"{path}.mapping"
            if os.path.exists(mapping_path):
                with open(mapping_path, "rb") as f:
                    data = pickle.load(f)
                    self.id_map = data["id_map"]
                    self.reverse_id_map = data["reverse_id_map"]
                    self._next_internal_id = data["next_internal_id"]
            
            INDEX_SIZE.set(self.index.ntotal)
            logger.info(f"Loaded index from {path}", size=self.index.ntotal)

    async def clear(self) -> None:
        """Clear all embeddings from index"""
        self.index.reset()
        self.id_map.clear()
        self.reverse_id_map.clear()
        self._next_internal_id = 0
        INDEX_SIZE.set(0)

    async def rebuild(self) -> None:
        """Rebuild index (no-op for flat index)"""
        logger.info("Rebuild not needed for flat index")

    def size(self) -> int:
        """Get number of embeddings in index"""
        return self.index.ntotal

    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics"""
        return {
            "type": "faiss_flat",
            "size": self.index.ntotal,
            "dimension": self.dimension,
            "metric": "cosine",
            "supports_removal": False,
        }


def create_faiss_index(config: Optional[IndexConfig] = None) -> FaissIndexFlat:
    """Factory function to create FAISS index"""
    if config is None:
        config = IndexConfig(
            dimension=settings.embedding_size,
            metric="cosine",
            index_type="flat",
        )
    return FaissIndexFlat(config)