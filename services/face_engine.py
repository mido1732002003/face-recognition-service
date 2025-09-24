import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from api.config import settings
from core.constants import DETECTOR_BACKENDS, INSIGHTFACE_MODELS
from core.exceptions import (
    InvalidImageException,
    MultipleFacesDetectedException,
    NoFaceDetectedException,
)
from utils.logging import get_logger
from utils.metrics import (
    FACE_DETECTION_COUNT,
    FACE_DETECTION_DURATION,
    FACE_EMBEDDING_DURATION,
)

logger = get_logger(__name__)


class FaceEngine:
    """Face detection and embedding extraction engine using InsightFace"""

    def __init__(self):
        self.app = None
        self.detector_backend = settings.detector_backend
        self.device = settings.device
        self.face_model = settings.face_model
        self._initialize()

    def _initialize(self):
        """Initialize InsightFace models"""
        logger.info(
            "Initializing face engine",
            model=self.face_model,
            detector=self.detector_backend,
            device=self.device,
        )

        # Initialize FaceAnalysis with specified model
        self.app = FaceAnalysis(
            name=INSIGHTFACE_MODELS.get(self.face_model, "buffalo_l"),
            providers=["CUDAExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )

        # Configure detector backend
        detector_config = DETECTOR_BACKENDS[self.detector_backend]
        self.app.prepare(ctx_id=0, det_thresh=detector_config["conf_threshold"])

        # Override detector if using RetinaFace
        if self.detector_backend == "retinaface":
            self._setup_retinaface()

        logger.info("Face engine initialized successfully")

    def _setup_retinaface(self):
        """Setup RetinaFace detector"""
        # RetinaFace is already included in buffalo models
        # This is where you would override with a specific RetinaFace model if needed
        pass

    def detect_faces(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect faces in image"""
        start_time = time.time()

        try:
            faces = self.app.get(image)
            duration = time.time() - start_time
            FACE_DETECTION_DURATION.observe(duration)

            if not faces:
                FACE_DETECTION_COUNT.labels(status="no_face", reason="none_detected").inc()
                raise NoFaceDetectedException()

            FACE_DETECTION_COUNT.labels(status="success", reason="detected").inc()

            # Convert InsightFace format to our format
            face_data = []
            for face in faces:
                face_data.append({
                    "bbox": face.bbox.tolist(),
                    "landmarks": face.kps.tolist() if face.kps is not None else None,
                    "det_score": float(face.det_score),
                    "embedding": face.normed_embedding,  # L2-normalized by default
                    "face": face,
                })

            logger.info(f"Detected {len(face_data)} face(s)", duration_ms=duration * 1000)
            return face_data

        except Exception as e:
            if isinstance(e, NoFaceDetectedException):
                raise
            logger.error("Face detection failed", error=str(e))
            FACE_DETECTION_COUNT.labels(status="error", reason="detection_failed").inc()
            raise InvalidImageException(f"Face detection failed: {str(e)}")

    def extract_embedding(self, face_data: dict[str, Any]) -> np.ndarray:
        """Extract face embedding (already extracted during detection)"""
        start_time = time.time()

        # InsightFace already computes embeddings during detection
        embedding = face_data["embedding"]

        # Ensure L2 normalization (InsightFace already does this, but double-check)
        embedding = embedding / np.linalg.norm(embedding)

        duration = time.time() - start_time
        FACE_EMBEDDING_DURATION.observe(duration)

        return embedding

    def process_single_face(
        self, image: np.ndarray, allow_multiple: bool = False
    ) -> Tuple[dict[str, Any], np.ndarray]:
        """Process image and extract embedding for single face"""
        faces = self.detect_faces(image)

        if len(faces) > 1 and not allow_multiple:
            raise MultipleFacesDetectedException(len(faces))

        # Select largest face if multiple detected
        if len(faces) > 1:
            faces.sort(key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]), reverse=True)
            logger.warning(f"Multiple faces detected, selecting largest face")

        face_data = faces[0]
        embedding = face_data["embedding"]

        return face_data, embedding

    def process_multiple_faces(self, image: np.ndarray) -> list[Tuple[dict[str, Any], np.ndarray]]:
        """Process image and extract embeddings for all faces"""
        faces = self.detect_faces(image)

        results = []
        for face_data in faces:
            embedding = face_data["embedding"]
            results.append((face_data, embedding))

        return results

    def align_face(self, image: np.ndarray, face_data: dict[str, Any]) -> np.ndarray:
        """Align face using landmarks (if needed)"""
        # InsightFace already provides aligned faces
        # This method is here for compatibility/future use
        face = face_data.get("face")
        if face and hasattr(face, "normed_img"):
            return face.normed_img
        return image

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        # Since embeddings are L2-normalized, dot product = cosine similarity
        return float(np.dot(embedding1, embedding2))


# Global instance
_face_engine: Optional[FaceEngine] = None


def get_face_engine() -> FaceEngine:
    """Get or create face engine instance"""
    global _face_engine
    if _face_engine is None:
        _face_engine = FaceEngine()
    return _face_engine