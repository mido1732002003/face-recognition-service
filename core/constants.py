from enum import Enum


class EnrollmentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FaceQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


# Face detection and quality thresholds
MIN_FACE_SIZE = 40  # minimum face size in pixels
MAX_FACES_PER_IMAGE = 10  # maximum faces to process per image
DEFAULT_QUALITY_THRESHOLD = 0.5

# Image processing
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_FORMATS = {"JPEG", "JPG", "PNG", "BMP", "WEBP"}
IMAGE_PROCESSING_TIMEOUT = 30  # seconds

# Index constants
FAISS_INDEX_FILENAME = "face_index.faiss"
FAISS_METADATA_FILENAME = "face_metadata.pkl"
INDEX_REBUILD_THRESHOLD = 10000  # rebuild index after this many deletions

# Model names
INSIGHTFACE_MODELS = {
    "buffalo_l": "buffalo_l",
    "buffalo_m": "buffalo_m",
    "buffalo_s": "buffalo_s",
    "buffalo_sc": "buffalo_sc",
}

DETECTOR_BACKENDS = {
    "scrfd": {"model": "scrfd_10g_bnkps", "conf_threshold": 0.5},
    "retinaface": {"model": "retinaface_r50_v1", "conf_threshold": 0.8},
}

# API versioning
API_VERSION = "v1"
SERVICE_NAME = "face-recognition-service"