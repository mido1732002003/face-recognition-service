from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/facedb"
    )
    db_pool_size: int = Field(default=20)
    db_max_overflow: int = Field(default=0)

    # Face Recognition
    face_model: str = Field(default="buffalo_l")
    detector_backend: Literal["scrfd", "retinaface"] = Field(default="scrfd")
    device: Literal["cpu", "cuda"] = Field(default="cpu")
    embedding_size: int = Field(default=512)
    similarity_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    top_k_results: int = Field(default=5, ge=1)

    # Index Configuration
    index_type: Literal["flat", "ivfpq", "scann", "milvus", "qdrant"] = Field(default="flat")
    index_path: str = Field(default="/app/data/faiss_index")
    ivf_nlist: int = Field(default=100)
    pq_m: int = Field(default=64)
    pq_nbits: int = Field(default=8)

    # Security
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests: int = Field(default=100)
    rate_limit_window: int = Field(default=60)
    rbac_enabled: bool = Field(default=False)

    # Storage
    image_storage_path: str = Field(default="/app/data/images")
    enable_image_storage: bool = Field(default=False)

    # Monitoring
    metrics_enabled: bool = Field(default=True)
    trace_enabled: bool = Field(default=False)

    # Anti-spoofing
    liveness_check_enabled: bool = Field(default=False)
    liveness_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("similarity_threshold", "liveness_confidence_threshold")
    def validate_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        return v


settings = Settings()