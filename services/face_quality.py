from typing import Any, Optional

import cv2
import numpy as np

from core.constants import DEFAULT_QUALITY_THRESHOLD, FaceQuality
from utils.logging import get_logger
from utils.metrics import track_face_quality

logger = get_logger(__name__)


class FaceQualityAnalyzer:
    """Analyze face quality for enrollment and identification"""

    def __init__(self):
        self.quality_threshold = DEFAULT_QUALITY_THRESHOLD

    def analyze_face(self, image: np.ndarray, face_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze face quality and return quality metrics"""
        bbox = face_data["bbox"]
        landmarks = face_data.get("landmarks")

        # Calculate quality metrics
        size_score = self._calculate_size_score(bbox, image.shape)
        pose_score = self._calculate_pose_score(landmarks) if landmarks else 0.5
        sharpness_score = self._calculate_sharpness_score(image, bbox)
        brightness_score = self._calculate_brightness_score(image, bbox)

        # Combined quality score (weighted average)
        overall_score = (
            size_score * 0.3
            + pose_score * 0.3
            + sharpness_score * 0.2
            + brightness_score * 0.2
        )

        # Track quality score
        track_face_quality(overall_score)

        # Determine quality level
        if overall_score >= 0.8:
            quality_level = FaceQuality.EXCELLENT
        elif overall_score >= 0.6:
            quality_level = FaceQuality.GOOD
        elif overall_score >= 0.4:
            quality_level = FaceQuality.FAIR
        else:
            quality_level = FaceQuality.POOR

        return {
            "overall_score": overall_score,
            "quality_level": quality_level,
            "metrics": {
                "size_score": size_score,
                "pose_score": pose_score,
                "sharpness_score": sharpness_score,
                "brightness_score": brightness_score,
            },
            "is_acceptable": overall_score >= self.quality_threshold,
        }

    def _calculate_size_score(self, bbox: list[float], image_shape: tuple) -> float:
        """Calculate face size score (larger is better)"""
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        image_area = image_shape[0] * image_shape[1]
        face_area = face_width * face_height

        # Face should occupy reasonable portion of image
        area_ratio = face_area / image_area
        
        # Ideal ratio between 0.1 and 0.5
        if area_ratio < 0.05:
            return 0.0
        elif area_ratio > 0.5:
            return 0.8
        else:
            return min(1.0, area_ratio * 4)

    def _calculate_pose_score(self, landmarks: Optional[list]) -> float:
        """Calculate face pose score based on landmarks symmetry"""
        if not landmarks or len(landmarks) < 5:
            return 0.5

        landmarks = np.array(landmarks)
        
        # Check eye alignment (should be roughly horizontal)
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        eye_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        
        # Penalize large angles
        angle_penalty = min(abs(eye_angle) / (np.pi / 6), 1.0)
        
        # Check face symmetry
        nose = landmarks[2]
        eye_center = (left_eye + right_eye) / 2
        symmetry = 1.0 - min(abs(nose[0] - eye_center[0]) / (right_eye[0] - left_eye[0]), 1.0)
        
        return (1.0 - angle_penalty) * 0.5 + symmetry * 0.5

    def _calculate_sharpness_score(self, image: np.ndarray, bbox: list[float]) -> float:
        """Calculate image sharpness using Laplacian variance"""
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        face_roi = image[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 range (empirically determined thresholds)
        score = min(variance / 500.0, 1.0)
        return score

    def _calculate_brightness_score(self, image: np.ndarray, bbox: list[float]) -> float:
        """Calculate brightness/contrast score"""
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        face_roi = image[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi
        
        # Calculate mean and std
        mean = gray.mean()
        std = gray.std()
        
        # Ideal brightness around 127 with good contrast
        brightness_score = 1.0 - abs(mean - 127) / 127
        contrast_score = min(std / 50.0, 1.0)
        
        return brightness_score * 0.5 + contrast_score * 0.5

    def set_threshold(self, threshold: float):
        """Update quality threshold"""
        self.quality_threshold = max(0.0, min(1.0, threshold))


# Global instance
_quality_analyzer: Optional[FaceQualityAnalyzer] = None


def get_quality_analyzer() -> FaceQualityAnalyzer:
    """Get or create quality analyzer instance"""
    global _quality_analyzer
    if _quality_analyzer is None:
        _quality_analyzer = FaceQualityAnalyzer()
    return _quality_analyzer