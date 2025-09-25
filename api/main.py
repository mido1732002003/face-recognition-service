from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings
from api.middleware import (
    add_correlation_id,
    add_request_logging,
    add_response_time,
    rate_limit_middleware,
)
from api.routers import enrollment, health, identification, metrics, persons
from core.database import engine
from core.exceptions import FaceRecognitionException
from core.models import Base
from indexing import get_index
from utils.logging import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Face Recognition Service")

    # Create database tables
    try:
        logger.info("Creating DB tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("DB tables created successfully")
    except Exception as e:
        logger.error(f"DB init failed: {e}")

    # Load index
    try:
        logger.info("Loading FAISS index...")
        index = await get_index()
        logger.info(f"Index loaded with {index.size()} embeddings")
    except Exception as e:
        logger.error(f"Index load failed: {e}")

    yield

    # Cleanup
    logger.info("Shutting down Face Recognition Service")
    await engine.dispose()


# Create FastAPI app
app = FastAPI(
    title="Face Recognition Service",
    description="Enterprise-grade 1:N face recognition API",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.middleware("http")(add_correlation_id)
app.middleware("http")(add_response_time)
app.middleware("http")(add_request_logging)

if settings.rate_limit_enabled:
    app.middleware("http")(rate_limit_middleware)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(metrics.router, tags=["Metrics"])
app.include_router(enrollment.router, prefix="/api/v1", tags=["Enrollment"])
app.include_router(identification.router, prefix="/api/v1", tags=["Identification"])
app.include_router(persons.router, prefix="/api/v1", tags=["Persons"])


# Global exception handler
@app.exception_handler(FaceRecognitionException)
async def face_recognition_exception_handler(request, exc: FaceRecognitionException):
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception", error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
        },
    )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": "Face Recognition Service",
        "version": "1.0.0",
        "status": "operational",
    }
