from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from core.schemas import PersonCreate, PersonResponse, StatsResponse
from services.person_service import get_person_service
from utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/persons/{person_id}", response_model=PersonResponse)
async def get_person(person_id: str):
    """Get person by ID"""
    person_service = get_person_service()
    
    try:
        return await person_service.get_person(person_id)
    except Exception as e:
        logger.error(f"Failed to get person", error=str(e), person_id=person_id)
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/persons")
async def list_persons(
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None),
):
    """List all persons with pagination"""
    person_service = get_person_service()
    
    try:
        return await person_service.list_persons(offset, limit, search)
    except Exception as e:
        logger.error(f"Failed to list persons", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/persons", response_model=PersonResponse)
async def create_person(person: PersonCreate):
    """Create a new person"""
    # Person will be created automatically during enrollment
    # This endpoint is for pre-creating persons with metadata
    from services.enrollment_service import get_enrollment_service
    
    enrollment_service = get_enrollment_service()
    
    try:
        await enrollment_service.update_person_metadata(
            person_id=person.id,
            name=person.name,
            metadata=person.metadata,
        )
        
        person_service = get_person_service()
        return await person_service.get_person(person.id)
        
    except Exception as e:
        logger.error(f"Failed to create person", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/persons/{person_id}")
async def delete_person(person_id: str):
    """Delete a person and all associated data"""
    person_service = get_person_service()
    
    try:
        await person_service.delete_person(person_id)
        return {"message": f"Person {person_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete person", error=str(e), person_id=person_id)
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    person_service = get_person_service()
    
    try:
        return await person_service.get_stats()
    except Exception as e:
        logger.error(f"Failed to get stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))