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


class FaissIndexIVFPQ(VectorIndex):
    """FAISS IVF-PQ index for large-scale approximate search"""

    def __init__(self, config: IndexConfig):
        self.config = config
        self.dimension = config.dimension
        self.nlist = config.extra_params.get("nlist", settings.ivf_nlist)
        self.m = config.extra_params.get("m", settings.pq_m)
        self.nbits = config.extra_params.get("nbits", settings.pq_nbits)
        
        # Create IVF-PQ index
        quantizer = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFPQ(
            quantizer, self.dimension, self.nlist, self.m, self.nbits
        )
        
        # ID mapping
        self.id_map: dict[int, int] = {}
        self.reverse_id_map: dict[int, int] = {}
        self._next_internal_id = 0
        self._is_trained = False
        
        # Training data buffer
        self._training_data: list[np.ndarray] = []
        self._min_training_samples = max(self.nlist * 40, 10000)

    async def add(self, embeddings: np.ndarray, ids: list[int]) -> None:
        """Add embeddings to index"""
        start_time = asyncio.get_event_loop().time()
        
        # Ensure embeddings are L2-normalized
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Train index if not trained
        if not self._is_trained:
            self._training_data.append(embeddings)
            total_samples = sum(d.shape[0] for d in self._training_data)
            
            if total_samples >= self._min_training_samples:
                await self._train_index()
        
        # Add to index only if trained
        if self._is_trained:
            await asyncio.to_thread(self.index.add, embeddings)
            
            # Update ID mappings
            for i, external_id in enumerate(ids):
                internal_id = self._next_internal_id + i
                self.id_map[internal_id] = external_id
                self.reverse_id_map[external_id] = internal_id
            
            self._next_internal_id += len(ids)
            
            INDEX_SIZE.set(self.index.ntotal)
        
        duration = asyncio.get_event_loop().time() - start_time
        INDEX_ADD_DURATION.observe(duration)

    async def _train_index(self):
        """Train the IVF-PQ index"""
        logger.info("Training IVF-PQ index")
        
        # Concatenate training data
        training_data = np.vstack(self._training_data)
        
        # Train index
        await asyncio.to_thread(self.index.train, training_data)
        self._is_trained = True
        
        # Add training data to index
        await asyncio.to_thread(self.index.add, training_data)
        
        # Update ID mappings for training data
        for i in range(training_data.shape[0]):
            self.id_map[i] = i
            self.reverse_id_map[i] = i
        
        self._next_internal_id = training_data.shape[0]
        
        # Clear training buffer
        self._training_data.clear()
        
        logger.info("IVF-PQ index trained", samples=training_data.shape[0])

    async def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors"""
        if not self._is_trained:
            return np.array([]), np.array([])
        
        start_time = asyncio.get_event_loop().time()
        
        # Ensure query is L2-normalized
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Set search parameters
        self.index.nprobe = min(self.nlist // 4, 16)  # Search 25% of cells
        
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

    async def save(self, path: str) -> None:
        """Save index to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        index_path = f"{path}.ivfpq"
        await asyncio.to_thread(faiss.write_index, self.index, index_path)
        
        # Save metadata
        metadata_path = f"{path}.metadata"
        with open(metadata_path, "wb") as f:
            pickle.dump({
                "id_map": self.id_map,
                "reverse_id_map": self.reverse_id_map,
                "next_internal_id": self._next_internal_id,
                "is_trained": self._is_trained,
                "config": {
                    "dimension": self.dimension,
                    "nlist": self.nlist,
                    "m": self.m,
                    "nbits": self.nbits,
                },
            }, f)

    async def load(self, path: str) -> None:
        """Load index from disk"""
        index_path = f"{path}.ivfpq"
        if os.path.exists(index_path):
            self.index = await asyncio.to_thread(faiss.read_index, index_path)
            
            metadata_path = f"{path}.metadata"
            if os.path.exists(metadata_path):
                with open(metadata_path, "rb") as f:
                    data = pickle.load(f)
                    self.id_map = data["id_map"]
                    self.reverse_id_map = data["reverse_id_map"]
                    self._next_internal_id = data["next_internal_id"]
                    self._is_trained = data["is_trained"]
            
            INDEX_SIZE.set(self.index.ntotal)

    async def remove(self, ids: list[int]) -> None:
        """Remove embeddings by ID"""
        # IVF-PQ doesn't support efficient removal
        for external_id in ids:
            if external_id in self.reverse_id_map:
                internal_id = self.reverse_id_map[external_id]
                del self.id_map[internal_id]
                del self.reverse_id_map[external_id]

    async def clear(self) -> None:
        """Clear all embeddings from index"""
        self.index.reset()
        self.id_map.clear()
        self.reverse_id_map.clear()
        self._next_internal_id = 0
        self._is_trained = False
        self._training_data.clear()
        INDEX_SIZE.set(0)

    async def rebuild(self) -> None:
        """Rebuild index with current data"""
        if self._is_trained and self.index.ntotal > 0:
            logger.info("Rebuilding IVF-PQ index")
            # In practice, you'd reload all embeddings and retrain
            # This is a placeholder
            pass

    def size(self) -> int:
        """Get number of embeddings in index"""
        return self.index.ntotal if self._is_trained else 0

    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics"""
        return {
            "type": "faiss_ivfpq",
            "size": self.size(),
            "dimension": self.dimension,
            "metric": "cosine",
            "trained": self._is_trained,
            "nlist": self.nlist,
            "m": self.m,
            "nbits": self.nbits,
            "supports_removal": False,
        }