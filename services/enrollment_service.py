import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Any, Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db_context
from core.exceptions import (
    LowQualityFaceException,
    PersonNotFoundException,
)
from core.models import Enrollment, Face, Person
from indexing import get_index, save_index
from services.face_engine import get_face_engine
from services.face_quality import get_quality_analyzer
from services.liveness import get_liveness_service
from utils.image_utils import pil_to_cv2, save_image_to_disk, validate_image
from utils.logging import get_logger
from utils.metrics import track_enrollment

logger = get_logger(__name__)


class EnrollmentService:
    """Service for enrolling faces"""

    def __init__(self):
        self.face_engine = get_face_engine()
        self.quality_analyzer = get_quality_analyzer()
        self.liveness_service = get_liveness_service()

    async def enroll_faces(
        self,
        person_id: str,
        images: list[bytes],
        quality_threshold: Optional[float] = None,
        update_if_exists: bool = True,
    ) -> dict[str, Any]:
        """Enroll multiple face images for a person"""
        start_time = time.time()
        enrollment_id = uuid.uuid4()
        
        async with get_db_context() as session:
            # Create enrollment record
            enrollment = Enrollment(
                id=enrollment_id,
                person_id=person_id,
                status="processing",
            )
            session.add(enrollment)
            await session.flush()
            
            try:
                # Check if person exists
                person = await self._get_or_create_person(session, person_id)
                
                # Process each image
                successful_faces = []
                failed_faces = []
                
                for img_bytes in images:
                    try:
                        face_data = await self._process_single_image(
                            session,
                            person_id,
                            img_bytes,
                            quality_threshold,
                        )
                        successful_faces.append(face_data)
                    except Exception as e:
                        logger.error(f"Failed to process image", error=str(e))
                        failed_faces.append(str(e))
                
                # Update enrollment
                enrollment.face_count = len(successful_faces)
                enrollment.status = "completed" if successful_faces else "failed"
                enrollment.completed_at = datetime.utcnow()
                
                if failed_faces and not successful_faces:
                    enrollment.error_message = "; ".join(failed_faces)
                
                await session.commit()
                
                # Save index to disk
                await save_index()
                
                # Track metrics
                track_enrollment("completed" if successful_faces else "failed")
                
                processing_time = (time.time() - start_time) * 1000
                
                return {
                    "enrollment_id": enrollment_id,
                    "person_id": person_id,
                    "faces_enrolled": len(successful_faces),
                    "faces_failed": len(failed_faces),
                    "status": enrollment.status,
                    "processing_time_ms": processing_time,
                    "errors": failed_faces if failed_faces else None,
                }
                
            except Exception as e:
                enrollment.status = "failed"
                enrollment.error_message = str(e)
                await session.commit()
                track_enrollment("failed")
                raise

    async def _process_single_image(
        self,
        session: AsyncSession,
        person_id: str,
        image_bytes: bytes,
        quality_threshold: Optional[float],
    ) -> dict[str, Any]:
        """Process single image for enrollment"""
        # Validate image
        pil_image = validate_image(image_bytes)
        cv2_image = pil_to_cv2(pil_image)
        
        # Detect and extract face
        face_data, embedding = self.face_engine.process_single_face(cv2_image)
        
        # Check face quality
        quality_result = self.quality_analyzer.analyze_face(cv2_image, face_data)
        
        threshold = quality_threshold or self.quality_analyzer.quality_threshold
        if quality_result["overall_score"] < threshold:
            raise LowQualityFaceException(quality_result["overall_score"], threshold)
        
        # Check liveness
        liveness_result = self.liveness_service.check_liveness(cv2_image, face_data)
        
        # Get vector index
        index = await get_index()
        
        # Generate embedding ID
        embedding_id = index.size()  # Use current size as ID
        
        # Add to index
        await index.add(embedding.reshape(1, -1), [embedding_id])
        
        # Save face record
        face = Face(
            person_id=person_id,
            embedding_id=embedding_id,
            quality_score=quality_result["overall_score"],
            face_bbox=json.dumps(face_data["bbox"]),
            landmarks=json.dumps(face_data.get("landmarks")),
        )
        
        # Optionally save image
        if settings.enable_image_storage:
            image_path = save_image_to_disk(
                image_bytes,
                settings.image_storage_path,
            )
            face.image_path = image_path
        
        session.add(face)
        await session.flush()
        
        return {
            "face_id": face.id,
            "quality_score": quality_result["overall_score"],
            "liveness": liveness_result,
        }

    async def _get_or_create_person(
        self, session: AsyncSession, person_id: str
    ) -> Person:
        """Get existing person or create new one"""
        result = await session.execute(
            select(Person).where(Person.id == person_id)
        )
        person = result.scalar_one_or_none()
        
        if not person:
            person = Person(id=person_id)
            session.add(person)
            await session.flush()
        
        return person

    async def update_person_metadata(
        self,
        person_id: str,
        name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Update person metadata"""
        async with get_db_context() as session:
            result = await session.execute(
                select(Person).where(Person.id == person_id)
            )
            person = result.scalar_one_or_none()
            
            if not person:
                raise PersonNotFoundException(person_id)
            
            if name is not None:
                person.name = name
            
            if metadata is not None:
                person.metadata = json.dumps(metadata)
            
            await session.commit()


# Global instance
_enrollment_service: Optional[EnrollmentService] = None


def get_enrollment_service() -> EnrollmentService:
    """Get or create enrollment service instance"""
    global _enrollment_service
    if _enrollment_service is None:
        _enrollment_service = EnrollmentService()
    return _enrollment_service