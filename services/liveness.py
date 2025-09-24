from typing import Any, Optional, Protocol

import numpy as np

from api.config import settings
from core.exceptions import LivenessCheckFailedException
from utils.logging import get_logger

logger = get_logger(__name__)


class LivenessDetector(Protocol):
    """Protocol for liveness detection implementations"""

    def check_liveness(self, image: np.ndarray, face_data: dict[str, Any]) -> dict[str, Any]:
        """Check if face is live (not spoofed)"""
        ...


class NoOpLivenessDetector:
    """Placeholder liveness detector that always passes"""

    def check_liveness(self, image: np.ndarray, face_data: dict[str, Any]) -> dict[str, Any]:
        """Always return live (no-op implementation)"""
        return {
            "is_live": True,
            "confidence": 1.0,
            "method": "none",
            "details": {"message": "Liveness check disabled"},
        }


class SimpleLivenessDetector:
    """Simple liveness detector using basic heuristics"""

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold

    def check_liveness(self, image: np.ndarray, face_data: dict[str, Any]) -> dict[str, Any]:
        """Basic liveness check using texture analysis"""
        # This is a placeholder for demonstration
        # Real implementation would use:
        # - Silent-Face-Anti-Spoofing
        # - FAS-PyTorch
        # - Or commercial solutions

        # Simulate texture-based anti-spoofing
        confidence = 0.85  # Would be computed from actual model

        is_live = confidence >= self.confidence_threshold

        if not is_live:
            raise LivenessCheckFailedException(confidence)

        return {
            "is_live": is_live,
            "confidence": confidence,
            "method": "texture_analysis",
            "details": {
                "threshold": self.confidence_threshold,
                "message": "Basic texture analysis",
            },
        }


class LivenessService:
    """Service for managing liveness detection"""

    def __init__(self):
        self.enabled = settings.liveness_check_enabled
        self.detector = self._create_detector()

    def _create_detector(self) -> LivenessDetector:
        """Create appropriate liveness detector"""
        if not self.enabled:
            return NoOpLivenessDetector()

        # Add more detector options here
        # Example: if settings.liveness_backend == "silent_face":
        #     return SilentFaceDetector()

        return SimpleLivenessDetector(settings.liveness_confidence_threshold)

    def check_liveness(self, image: np.ndarray, face_data: dict[str, Any]) -> dict[str, Any]:
        """Check liveness of detected face"""
        if not self.enabled:
            return self.detector.check_liveness(image, face_data)

        try:
            result = self.detector.check_liveness(image, face_data)
            logger.info("Liveness check completed", **result)
            return result
        except LivenessCheckFailedException:
            raise
        except Exception as e:
            logger.error("Liveness check failed", error=str(e))
            # Fail open or closed based on configuration
            if settings.liveness_fail_open:
                return {
                    "is_live": True,
                    "confidence": 0.0,
                    "method": "error",
                    "details": {"error": str(e)},
                }
            raise


# Global instance
_liveness_service: Optional[LivenessService] = None


def get_liveness_service() -> LivenessService:
    """Get or create liveness service instance"""
    global _liveness_service
    if _liveness_service is None:
        _liveness_service = LivenessService()
    return _liveness_service


# Integration points for open-source solutions:
# 1. Silent-Face-Anti-Spoofing: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
# 2. FAS-PyTorch: https://github.com/vokhidovhusan/FAS-PyTorch
# 3. DeepPixBiS: https://github.com/vokhidovhusan/DeepPixBiS
# 4. Commercial: FaceTec, BioID, Jumio