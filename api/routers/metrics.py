from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from utils.metrics import get_metrics

router = APIRouter()


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint"""
    return get_metrics()