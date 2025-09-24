from typing import Optional

from api.config import settings
from indexing.base import IndexConfig, VectorIndex
from indexing.faiss_index import FaissIndexFlat
from indexing.ivfpq_index import FaissIndexIVFPQ
from indexing.milvus_adapter import MilvusAdapter
from indexing.scann_adapter import ScannAdapter
from utils.logging import get_logger

logger = get_logger(__name__)

# Global index instance
_vector_index: Optional[VectorIndex] = None


def create_index(config: Optional[IndexConfig] = None) -> VectorIndex:
    """Factory to create appropriate vector index"""
    if config is None:
        config = IndexConfig(
            dimension=settings.embedding_size,
            metric="cosine",
            index_type=settings.index_type,
        )
    
    index_type = config.index_type.lower()
    
    if index_type == "flat":
        logger.info("Creating FAISS flat index")
        return FaissIndexFlat(config)
    elif index_type == "ivfpq":
        logger.info("Creating FAISS IVF-PQ index")
        return FaissIndexIVFPQ(config)
    elif index_type == "scann":
        logger.info("Creating ScaNN adapter")
        return ScannAdapter(config)
    elif index_type == "milvus":
        logger.info("Creating Milvus adapter")
        return MilvusAdapter(config)
    elif index_type == "qdrant":
        logger.warning("Qdrant adapter not implemented, using FAISS flat")
        return FaissIndexFlat(config)
    else:
        logger.warning(f"Unknown index type {index_type}, using FAISS flat")
        return FaissIndexFlat(config)


async def get_index() -> VectorIndex:
    """Get or create global index instance"""
    global _vector_index
    
    if _vector_index is None:
        _vector_index = create_index()
        
        # Try to load existing index
        try:
            await _vector_index.load(settings.index_path)
            logger.info("Loaded existing index", size=_vector_index.size())
        except Exception as e:
            logger.info("No existing index found, starting fresh", error=str(e))
    
    return _vector_index


async def save_index():
    """Save current index to disk"""
    if _vector_index is not None:
        await _vector_index.save(settings.index_path)


__all__ = ["VectorIndex", "IndexConfig", "create_index", "get_index", "save_index"]