from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from core.schemas import IdentificationRequest, IdentificationResponse
from services.identification_service import get_identification_service
from utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/identify", response_model=IdentificationResponse)
async def identify_face(
    image: UploadFile = File(...),
    similarity_threshold: Optional[float] = Form(None),
    top_k: Optional[int] = Form(None),
    return_face_data: bool = Form(False),
):
    """
    Identify a person from a face image.
    
    - **image**: Face image to identify
    - **similarity_threshold**: Minimum similarity score (0-1)
    - **top_k**: Number of top matches to return
    - **return_face_data**: Include face metadata in response
    """
    # Validate image
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image type: {image.content_type}",
        )
    
    # Read image data
    image_data = await image.read()
    
    # Identify face
    identification_service = get_identification_service()
    
    try:
        result = await identification_service.identify_face(
            image_bytes=image_data,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            return_face_data=return_face_data,
        )
        
        return IdentificationResponse(
            matches=result["matches"],
            face_quality=result["face_quality"],
            processing_time_ms=result["processing_time_ms"],
        )
        
    except Exception as e:
        logger.error(f"Identification failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify/{person_id}")
async def verify_face(
    person_id: str,
    image: UploadFile = File(...),
    similarity_threshold: Optional[float] = Form(None),
):
    """
    Verify if a face belongs to a specific person (1:1 matching).
    
    - **person_id**: ID of the person to verify against
    - **image**: Face image to verify
    - **similarity_threshold**: Minimum similarity score (0-1)
    """
    # Validate image
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image type: {image.content_type}",
        )
    
    # Read image data
    image_data = await image.read()
    
    # Verify face
    identification_service = get_identification_service()
    
    try:
        result = await identification_service.verify_face(
            person_id=person_id,
            image_bytes=image_data,
            similarity_threshold=similarity_threshold,
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Verification failed", error=str(e), person_id=person_id)
        raise HTTPException(status_code=500, detail=str(e))