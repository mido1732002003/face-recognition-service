import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class PersonCreate(BaseModel):
    id: str = Field(..., min_length=1, max_length=100)
    name: Optional[str] = Field(None, max_length=255)
    metadata: Optional[dict[str, Any]] = None


class PersonResponse(BaseModel):
    id: str
    name: Optional[str]
    metadata: Optional[dict[str, Any]]
    face_count: int = 0
    created_at: datetime
    updated_at: datetime


class EnrollmentRequest(BaseModel):
    quality_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    update_if_exists: bool = Field(default=True)


class EnrollmentResponse(BaseModel):
    enrollment_id: uuid.UUID
    person_id: str
    faces_enrolled: int
    status: str
    message: Optional[str] = None


class IdentificationRequest(BaseModel):
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1, le=100)
    return_face_data: bool = Field(default=False)


class IdentificationMatch(BaseModel):
    person_id: str
    similarity: float
    face_id: Optional[uuid.UUID] = None
    name: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class IdentificationResponse(BaseModel):
    matches: list[IdentificationMatch]
    face_quality: Optional[float] = None
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    database: str
    index_status: str
    face_engine: str


class StatsResponse(BaseModel):
    total_persons: int
    total_faces: int
    total_enrollments: int
    index_size: int
    index_dimension: int
    similarity_threshold: float