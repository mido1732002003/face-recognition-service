from typing import Optional
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    generate_latest,
)

# ✅ Custom registry عشان نتفادى duplication
registry = CollectorRegistry()

# Request metrics
REQUEST_COUNT = Counter(
    "face_recognition_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
    registry=registry,
)

REQUEST_DURATION = Histogram(
    "face_recognition_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
    registry=registry,
)

# Face processing metrics
FACE_DETECTION_COUNT = Counter(
    "face_detections_total",
    "Total number of face detections",
    ["status", "reason"],
    registry=registry,
)

FACE_DETECTION_DURATION = Histogram(
    "face_detection_duration_seconds",
    "Face detection duration in seconds",
    registry=registry,
)

FACE_EMBEDDING_DURATION = Histogram(
    "face_embedding_duration_seconds",
    "Face embedding extraction duration in seconds",
    registry=registry,
)

FACE_QUALITY_SCORE = Histogram(
    "face_quality_scores",
    "Distribution of face quality scores",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=registry,
)

# Index metrics
INDEX_SIZE = Gauge(
    "face_index_size",
    "Number of faces in the index",
    registry=registry,
)

INDEX_SEARCH_DURATION = Histogram(
    "face_index_search_duration_seconds",
    "Index search duration in seconds",
    registry=registry,
)

INDEX_ADD_DURATION = Histogram(
    "face_index_add_duration_seconds",
    "Index add operation duration in seconds",
    registry=registry,
)

# Database metrics
DB_QUERY_DURATION = Histogram(
    "database_query_duration_seconds",
    "Database query duration in seconds",
    ["operation", "table"],
    registry=registry,
)

DB_CONNECTION_POOL_SIZE = Gauge(
    "database_connection_pool_size",
    "Current size of the database connection pool",
    registry=registry,
)

# Business metrics
ENROLLMENT_COUNT = Counter(
    "face_enrollments_total",
    "Total number of face enrollments",
    ["status"],
    registry=registry,
)

IDENTIFICATION_COUNT = Counter(
    "face_identifications_total",
    "Total number of face identifications",
    ["result"],  # matched, unknown, error
    registry=registry,
)

SIMILARITY_SCORES = Histogram(
    "face_similarity_scores",
    "Distribution of face similarity scores",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=registry,
)


def get_metrics() -> bytes:
    """Generate Prometheus metrics"""
    return generate_latest(registry)


def track_face_quality(score: float) -> None:
    """Track face quality score"""
    FACE_QUALITY_SCORE.observe(score)


def track_similarity_score(score: float) -> None:
    """Track similarity score"""
    SIMILARITY_SCORES.observe(score)


def track_enrollment(status: str) -> None:
    """Track enrollment"""
    ENROLLMENT_COUNT.labels(status=status).inc()


def track_identification(result: str) -> None:
    """Track identification"""
    IDENTIFICATION_COUNT.labels(result=result).inc()
