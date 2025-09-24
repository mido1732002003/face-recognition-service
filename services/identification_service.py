import json
import time
from typing import Any, Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from core.database import get_db_context
from core.models import Face, Person
from indexing import get_index
from services.face_engine import get_face_engine
from services.face_quality import get_quality_analyzer
from services.liveness import get_liveness_service
from utils.image_utils import pil_to_cv2, validate_image
from utils.logging import get_logger
from utils.metrics import track_identification, track_similarity_score

logger = get_logger(__name__)


class IdentificationService:
    """Service for face identification"""

    def __init__(self):
        self.face_engine = get_face_engine()
        self.quality_analyzer = get_quality_analyzer()
        self.liveness_service = get_liveness_service()

    async def identify_face(
        self,
        image_bytes: bytes,
        similarity_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        return_face_data: bool = False,
    ) -> dict[str, Any]:
        """Identify a face in the image"""
        start_time = time.time()
        
        # Use configured defaults if not specified
        threshold = similarity_threshold or settings.similarity_threshold
        k = top_k or settings.top_k_results
        
        # Validate image
        pil_image = validate_image(image_bytes)
        cv2_image = pil_to_cv2(pil_image)
        
        # Detect and extract face
        face_data, embedding = self.face_engine.process_single_face(cv2_image)
        
        # Check face quality
        quality_result = self.quality_analyzer.analyze_face(cv2_image, face_data)
        
        # Check liveness
        liveness_result = self.liveness_service.check_liveness(cv2_image, face_data)
        
        # Search in index
        index = await get_index()
        
        if index.size() == 0:
            logger.warning("Index is empty, no faces enrolled")
            track_identification("unknown")
            return {
                "matches": [],
                "face_quality": quality_result["overall_score"],
                "processing_time_ms": (time.time() - start_time) * 1000,
            }
        
        distances, embedding_ids = await index.search(embedding, k)
        
        # Convert distances to similarities (cosine similarity)
        similarities = distances  # Already cosine similarity from IndexFlatIP
        
        # Get person data for matches
        matches = []
        async with get_db_context() as session:
            for i, (similarity, embedding_id) in enumerate(zip(similarities, embedding_ids)):
                if similarity >= threshold:
                    match_data = await self._get_match_data(
                        session, int(embedding_id), float(similarity), return_face_data
                    )
                    if match_data:
                        matches.append(match_data)
                        track_similarity_score(float(similarity))
        
        # Track identification result
        if matches:
            track_identification("matched")
        else:
            track_identification("unknown")
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "matches": matches,
            "face_quality": quality_result["overall_score"],
            "liveness": liveness_result,
            "processing_time_ms": processing_time,
        }

    async def _get_match_data(
        self,
        session: AsyncSession,
        embedding_id: int,
        similarity: float,
        include_face_data: bool,
    ) -> Optional[dict[str, Any]]:
        """Get match data for a specific embedding ID"""
        # Find face by embedding ID
        result = await session.execute(
            select(Face).where(Face.embedding_id == embedding_id)
        )
        face = result.scalar_one_or_none()
        
        if not face:
            logger.warning(f"Face not found for embedding_id {embedding_id}")
            return None
        
        # Get person data
        result = await session.execute(
            select(Person).where(Person.id == face.person_id)
        )
        person = result.scalar_one_or_none()
        
        if not person:
            logger.warning(f"Person not found for person_id {face.person_id}")
            return None
        
        match_data = {
            "person_id": person.id,
            "similarity": similarity,
            "name": person.name,
        }
        
        if person.metadata:
            try:
                match_data["metadata"] = json.loads(person.metadata)
            except:
                match_data["metadata"] = person.metadata
        
        if include_face_data:
            match_data["face_id"] = str(face.id)
            match_data["quality_score"] = face.quality_score
        
        return match_data

    async def verify_face(
        self,
        person_id: str,
        image_bytes: bytes,
        similarity_threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        """Verify if face belongs to specific person (1:1 matching)"""
        start_time = time.time()
        threshold = similarity_threshold or settings.similarity_threshold
        
        # Get person's faces
        async with get_db_context() as session:
            result = await session.execute(
                select(Face).where(Face.person_id == person_id)
            )
            faces = result.scalars().all()
            
            if not faces:
                return {
                    "verified": False,
                    "reason": "No enrolled faces for person",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                }
        
        # Process query image
        pil_image = validate_image(image_bytes)
        cv2_image = pil_to_cv2(pil_image)
        
        face_data, embedding = self.face_engine.process_single_face(cv2_image)
        
        # Get embeddings for person's faces
        index = await get_index()
        embedding_ids = [face.embedding_id for face in faces]
        
        # Compare with each enrolled face
        max_similarity = 0.0
        for embedding_id in embedding_ids:
            # This is simplified - in production, you'd retrieve stored embeddings
            distances, ids = await index.search(embedding, k=len(embedding_ids))
            
            for dist, idx in zip(distances, ids):
                if idx == embedding_id:
                    max_similarity = max(max_similarity, dist)
        
        verified = max_similarity >= threshold
        
        return {
            "verified": verified,
            "similarity": float(max_similarity),
            "threshold": threshold,
            "processing_time_ms": (time.time() - start_time) * 1000,
        }


# Global instance
_identification_service: Optional[IdentificationService] = None


def get_identification_service() -> IdentificationService:
    """Get or create identification service instance"""
    global _identification_service
    if _identification_service is None:
        _identification_service = IdentificationService()
    return _identification_service