from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from core.schemas import EnrollmentRequest, EnrollmentResponse
from services.enrollment_service import get_enrollment_service
from utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/enroll/{person_id}", response_model=EnrollmentResponse)
async def enroll_faces(
    person_id: str,
    images: List[UploadFile] = File(...),
    quality_threshold: Optional[float] = Form(None),
    update_if_exists: bool = Form(True),
):
    """
    Enroll face images for a person.
    
    - **person_id**: Unique identifier for the person
    - **images**: One or more face images
    - **quality_threshold**: Minimum face quality score (0-1)
    - **update_if_exists**: Whether to update existing enrollment
    """
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")
    
    # Read image data
    image_data = []
    for image in images:
        if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image type: {image.content_type}",
            )
        
        contents = await image.read()
        image_data.append(contents)
    
    # Enroll faces
    enrollment_service = get_enrollment_service()
    
    try:
        result = await enrollment_service.enroll_faces(
            person_id=person_id,
            images=image_data,
            quality_threshold=quality_threshold,
            update_if_exists=update_if_exists,
        )
        
        return EnrollmentResponse(
            enrollment_id=result["enrollment_id"],
            person_id=result["person_id"],
            faces_enrolled=result["faces_enrolled"],
            status=result["status"],
            message=result.get("errors"),
        )
        
    except Exception as e:
        logger.error(f"Enrollment failed", error=str(e), person_id=person_id)
        raise HTTPException(status_code=500, detail=str(e))