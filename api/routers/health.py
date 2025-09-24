from datetime import datetime

from fastapi import APIRouter
from sqlalchemy import text

from api.config import settings
from core.database import engine
from core.schemas import HealthResponse
from indexing import get_index
from utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow(),
        "database": "unknown",
        "index_status": "unknown",
        "face_engine": settings.face_model,
    }
    
    # Check database
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        health_status["database"] = "connected"
    except Exception as e:
        logger.error(f"Database health check failed", error=str(e))
        health_status["database"] = "disconnected"
        health_status["status"] = "unhealthy"
    
    # Check index
    try:
        index = await get_index()
        health_status["index_status"] = f"loaded ({index.size()} embeddings)"
    except Exception as e:
        logger.error(f"Index health check failed", error=str(e))
        health_status["index_status"] = "error"
        health_status["status"] = "degraded"
    
    return health_status


@router.get("/readiness")
async def readiness_check():
    """Readiness check for Kubernetes"""
    try:
        # Check if all services are ready
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        
        index = await get_index()
        
        return {"ready": True}
    except Exception as e:
        logger.error(f"Readiness check failed", error=str(e))
        return {"ready": False, "error": str(e)}


@router.get("/liveness")
async def liveness_check():
    """Liveness check for Kubernetes"""
    return {"alive": True}