"""
Background job processing with Celery
"""

import asyncio
import json
import time
from typing import Any, Dict, List

from celery import Celery, Task
from celery.utils.log import get_task_logger
from sqlalchemy import select

from core.database import get_db_context
from core.models import Enrollment, Face, Person
from indexing import get_index, save_index
from services.face_engine import get_face_engine
from utils.image_utils import validate_image, pil_to_cv2

# Configure Celery
celery_app = Celery(
    'face_recognition',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_routes={
        'face_recognition.batch_enrollment': {'queue': 'enrollment'},
        'face_recognition.reindex': {'queue': 'maintenance'},
        'face_recognition.cleanup_old_data': {'queue': 'maintenance'},
    },
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=240,  # 4 minutes
)

logger = get_task_logger(__name__)


class AsyncTask(Task):
    """Base task with async support"""
    
    def run(self, *args, **kwargs):
        """Run task in async context"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.async_run(*args, **kwargs))
        finally:
            loop.close()
    
    async def async_run(self, *args, **kwargs):
        """Override this in subclasses"""
        raise NotImplementedError


@celery_app.task(base=AsyncTask, bind=True, name='face_recognition.batch_enrollment')
class BatchEnrollmentTask(AsyncTask):
    """Process batch enrollment in background"""
    
    async def async_run(self, enrollment_id: str, person_id: str, 
                       image_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process batch enrollment"""
        logger.info(f"Starting batch enrollment for {person_id}")
        
        async with get_db_context() as session:
            # Get enrollment record
            result = await session.execute(
                select(Enrollment).where(Enrollment.id == enrollment_id)
            )
            enrollment = result.scalar_one_or_none()
            
            if not enrollment:
                logger.error(f"Enrollment {enrollment_id} not found")
                return {"status": "error", "message": "Enrollment not found"}
            
            # Update status
            enrollment.status = "processing"
            await session.commit()
            
            face_engine = get_face_engine()
            successful = 0
            failed = 0
            
            for img_data in image_data_list:
                try:
                    # Process image
                    image_bytes = img_data['data']
                    pil_image = validate_image(image_bytes)
                    cv2_image = pil_to_cv2(pil_image)
                    
                    # Extract face
                    face_data, embedding = face_engine.process_single_face(cv2_image)
                    
                    # Add to index
                    index = await get_index()
                    embedding_id = index.size()
                    await index.add(embedding.reshape(1, -1), [embedding_id])
                    
                    # Save face record
                    face = Face(
                        person_id=person_id,
                        embedding_id=embedding_id,
                        quality_score=img_data.get('quality_score'),
                        face_bbox=json.dumps(face_data['bbox']),
                    )
                    session.add(face)
                    
                    successful += 1
                    
                    # Report progress
                    self.update_state(
                        state='PROGRESS',
                        meta={'current': successful + failed, 
                              'total': len(image_data_list)}
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to process image: {e}")
                    failed += 1
            
            # Update enrollment
            enrollment.status = "completed"
            enrollment.face_count = successful
            enrollment.completed_at = time.time()
            
            await session.commit()
            await save_index()
            
            logger.info(f"Batch enrollment completed: {successful} successful, {failed} failed")
            
            return {
                "status": "completed",
                "successful": successful,
                "failed": failed,
                "enrollment_id": enrollment_id
            }


@celery_app.task(name='face_recognition.reindex')
def reindex_task():
    """Rebuild index from database"""
    logger.info("Starting reindex task")
    
    async def _reindex():
        index = await get_index()
        await index.clear()
        
        async with get_db_context() as session:
            result = await session.execute(select(Face))
            faces = result.scalars().all()
            
            # In production, load actual embeddings from storage
            # This is simplified for demonstration
            
            batch_size = 1000
            for i in range(0, len(faces), batch_size):
                batch = faces[i:i+batch_size]
                # Process batch
                logger.info(f"Reindexed batch {i//batch_size + 1}")
            
            await save_index()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_reindex())
        logger.info("Reindex completed")
        return {"status": "completed"}
    finally:
        loop.close()


@celery_app.task(name='face_recognition.cleanup_old_data')
def cleanup_old_data_task(days: int = 90):
    """Clean up old enrollment data"""
    logger.info(f"Cleaning up data older than {days} days")
    
    from datetime import datetime, timedelta
    
    async def _cleanup():
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        async with get_db_context() as session:
            # Delete old enrollments
            result = await session.execute(
                select(Enrollment).where(Enrollment.created_at < cutoff_date)
            )
            old_enrollments = result.scalars().all()
            
            for enrollment in old_enrollments:
                await session.delete(enrollment)
            
            await session.commit()
            
            return len(old_enrollments)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        deleted_count = loop.run_until_complete(_cleanup())
        logger.info(f"Deleted {deleted_count} old enrollments")
        return {"deleted": deleted_count}
    finally:
        loop.close()


# Celery Beat Schedule for periodic tasks
celery_app.conf.beat_schedule = {
    'reindex-weekly': {
        'task': 'face_recognition.reindex',
        'schedule': 604800.0,  # Weekly
    },
    'cleanup-daily': {
        'task': 'face_recognition.cleanup_old_data',
        'schedule': 86400.0,  # Daily
        'args': (90,)  # Keep 90 days
    },
}