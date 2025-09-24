from typing import Any, Optional


class FaceRecognitionException(Exception):
    """Base exception for face recognition service"""

    def __init__(self, message: str, code: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class PersonNotFoundException(FaceRecognitionException):
    def __init__(self, person_id: str):
        super().__init__(
            message=f"Person with ID '{person_id}' not found",
            code="PERSON_NOT_FOUND",
            details={"person_id": person_id},
        )


class NoFaceDetectedException(FaceRecognitionException):
    def __init__(self, details: Optional[dict[str, Any]] = None):
        super().__init__(
            message="No face detected in the provided image",
            code="NO_FACE_DETECTED",
            details=details or {},
        )


class MultipleFacesDetectedException(FaceRecognitionException):
    def __init__(self, face_count: int):
        super().__init__(
            message=f"Multiple faces detected ({face_count}). Please provide an image with a single face",
            code="MULTIPLE_FACES_DETECTED",
            details={"face_count": face_count},
        )


class LowQualityFaceException(FaceRecognitionException):
    def __init__(self, quality_score: float, threshold: float):
        super().__init__(
            message=f"Face quality score ({quality_score:.2f}) is below threshold ({threshold:.2f})",
            code="LOW_QUALITY_FACE",
            details={"quality_score": quality_score, "threshold": threshold},
        )


class IndexException(FaceRecognitionException):
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(
            message=f"Index operation failed: {message}",
            code="INDEX_ERROR",
            details=details or {},
        )


class InvalidImageException(FaceRecognitionException):
    def __init__(self, reason: str):
        super().__init__(
            message=f"Invalid image: {reason}",
            code="INVALID_IMAGE",
            details={"reason": reason},
        )


class LivenessCheckFailedException(FaceRecognitionException):
    def __init__(self, confidence: float):
        super().__init__(
            message="Liveness check failed - possible spoofing detected",
            code="LIVENESS_CHECK_FAILED",
            details={"confidence": confidence},
        )