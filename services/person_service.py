import json
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db_context
from core.exceptions import PersonNotFoundException
from core.models import Enrollment, Face, Person
from core.schemas import PersonResponse
from utils.logging import get_logger

logger = get_logger(__name__)


class PersonService:
    """Service for managing persons"""

    async def create_person(self, person_id: str, name: str, metadata: Optional[dict] = None) -> PersonResponse:
        """Create a new person"""
        async with get_db_context() as session:
            # Check if already exists
            result = await session.execute(
                select(Person).where(Person.id == person_id)
            )
            existing = result.scalar_one_or_none()
            if existing:
                raise Exception(f"Person with ID '{person_id}' already exists")

            # Create new person
            new_person = Person(
                id=person_id,
                name=name,
                metadata=json.dumps(metadata) if metadata else None,
            )
            session.add(new_person)
            await session.commit()
            await session.refresh(new_person)

            return PersonResponse(
                id=new_person.id,
                name=new_person.name,
                metadata=json.loads(new_person.metadata) if new_person.metadata else None,
                face_count=0,
                created_at=new_person.created_at,
                updated_at=new_person.updated_at,
            )

    async def get_person(self, person_id: str) -> PersonResponse:
        """Get person by ID"""
        async with get_db_context() as session:
            result = await session.execute(
                select(Person).where(Person.id == person_id)
            )
            person = result.scalar_one_or_none()

            if not person:
                raise PersonNotFoundException(person_id)

            # Get face count
            face_count_result = await session.execute(
                select(func.count(Face.id)).where(Face.person_id == person_id)
            )
            face_count = face_count_result.scalar() or 0

            return PersonResponse(
                id=person.id,
                name=person.name,
                metadata=json.loads(person.metadata) if person.metadata else None,
                face_count=face_count,
                created_at=person.created_at,
                updated_at=person.updated_at,
            )

    async def list_persons(
        self,
        offset: int = 0,
        limit: int = 100,
        search: Optional[str] = None,
    ) -> dict[str, Any]:
        """List all persons with pagination"""
        async with get_db_context() as session:
            query = select(Person)

            if search:
                query = query.where(
                    Person.id.ilike(f"%{search}%")
                    | Person.name.ilike(f"%{search}%")
                )

            # Get total count
            count_result = await session.execute(
                select(func.count()).select_from(query.subquery())
            )
            total = count_result.scalar() or 0

            # Get paginated results
            query = query.offset(offset).limit(limit).order_by(Person.created_at.desc())
            result = await session.execute(query)
            persons = result.scalars().all()

            # Get face counts
            person_ids = [p.id for p in persons]
            face_counts = {}

            if person_ids:
                face_count_result = await session.execute(
                    select(Face.person_id, func.count(Face.id))
                    .where(Face.person_id.in_(person_ids))
                    .group_by(Face.person_id)
                )
                face_counts = dict(face_count_result.all())

            items = []
            for person in persons:
                items.append(
                    PersonResponse(
                        id=person.id,
                        name=person.name,
                        metadata=json.loads(person.metadata) if person.metadata else None,
                        face_count=face_counts.get(person.id, 0),
                        created_at=person.created_at,
                        updated_at=person.updated_at,
                    )
                )

            return {
                "items": items,
                "total": total,
                "offset": offset,
                "limit": limit,
            }

    async def delete_person(self, person_id: str) -> None:
        """Delete person and all associated data"""
        async with get_db_context() as session:
            result = await session.execute(
                select(Person).where(Person.id == person_id)
            )
            person = result.scalar_one_or_none()

            if not person:
                raise PersonNotFoundException(person_id)

            # Faces will be cascade deleted
            await session.delete(person)
            await session.commit()

            # TODO: Remove embeddings from index
            logger.warning(f"Person {person_id} deleted but embeddings remain in index")

    async def get_stats(self) -> dict[str, Any]:
        """Get overall statistics"""
        async with get_db_context() as session:
            # Get counts
            person_count = await session.execute(select(func.count(Person.id)))
            face_count = await session.execute(select(func.count(Face.id)))
            enrollment_count = await session.execute(select(func.count(Enrollment.id)))

            # Get index stats
            from indexing import get_index
            index = await get_index()

            from api.config import settings
            return {
                "total_persons": person_count.scalar() or 0,
                "total_faces": face_count.scalar() or 0,
                "total_enrollments": enrollment_count.scalar() or 0,
                "index_size": index.size(),
                "index_dimension": index.dimension(),
                "similarity_threshold": settings.similarity_threshold,
                "index_type": settings.index_type,
                "face_model": settings.face_model,
                "detector_backend": settings.detector_backend,
            }


# Global instance
_person_service: Optional[PersonService] = None


def get_person_service() -> PersonService:
    """Get or create person service instance"""
    global _person_service
    if _person_service is None:
        _person_service = PersonService()
    return _person_service
