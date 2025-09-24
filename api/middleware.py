import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from api.config import settings
from utils.logging import get_logger
from utils.metrics import REQUEST_COUNT, REQUEST_DURATION

logger = get_logger(__name__)

# Rate limiting storage (in production, use Redis)
rate_limit_storage = defaultdict(list)


async def add_correlation_id(request: Request, call_next):
    """Add correlation ID to requests"""
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    
    return response


async def add_response_time(request: Request, call_next):
    """Add response time header"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


async def add_request_logging(request: Request, call_next):
    """Log requests and responses"""
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        method=request.method,
        path=request.url.path,
        correlation_id=getattr(request.state, "correlation_id", None),
    )
    
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start_time
    logger.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration * 1000,
        correlation_id=getattr(request.state, "correlation_id", None),
    )
    
    # Update metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)
    
    return response


async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware"""
    if not settings.rate_limit_enabled:
        return await call_next(request)
    
    # Get client identifier (IP address or API key)
    client_id = request.client.host if request.client else "unknown"
    
    # Check rate limit
    now = datetime.now()
    window_start = now - timedelta(seconds=settings.rate_limit_window)
    
    # Clean old entries
    rate_limit_storage[client_id] = [
        timestamp for timestamp in rate_limit_storage[client_id]
        if timestamp > window_start
    ]
    
    # Check if limit exceeded
    if len(rate_limit_storage[client_id]) >= settings.rate_limit_requests:
        return Response(
            content="Rate limit exceeded",
            status_code=429,
            headers={"Retry-After": str(settings.rate_limit_window)},
        )
    
    # Add current request
    rate_limit_storage[client_id].append(now)
    
    return await call_next(request)


class RBACMiddleware(BaseHTTPMiddleware):
    """Role-based access control middleware (placeholder)"""
    
    async def dispatch(self, request: Request, call_next):
        # In production, implement proper RBAC with JWT/OAuth
        # Check roles and permissions here
        
        # For now, just pass through
        response = await call_next(request)
        return response