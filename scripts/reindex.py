#!/usr/bin/env python3
"""Rebuild vector index from database"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select

from core.database import get_db_context
from core.models import Face
from indexing import get_index, save_index
from services.face_engine import get_face_engine
from utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


async def reindex():
    """Rebuild index from database"""
    logger.info("Starting reindex process")
    
    # Get index
    index = await get_index()
    
    # Clear existing index
    await index.clear()
    
    # Load all faces from database
    async with get_db_context() as session:
        result = await session.execute(select(Face))
        faces = result.scalars().all()
        
        logger.info(f"Found {len(faces)} faces to index")
        
        # In production, you'd load actual embeddings
        # For now, this is a placeholder
        embeddings = []
        ids = []
        
        for face in faces:
            # Load embedding from storage or regenerate
            # embedding = load_embedding(face)
            # embeddings.append(embedding)
            ids.append(face.embedding_id)
        
        if embeddings:
            # Add to index in batches
            batch_size = 1000
            for i in range(0, len(embeddings), batch_size):
                batch_emb = embeddings[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                await index.add(batch_emb, batch_ids)
                logger.info(f"Indexed batch {i//batch_size + 1}")
    
    # Save index
    await save_index()
    logger.info("Reindex completed")


if __name__ == "__main__":
    asyncio.run(reindex())