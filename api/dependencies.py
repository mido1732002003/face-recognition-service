from typing import Optional

from fastapi import Header, HTTPException

from api.config import settings


async def get_api_key(api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Validate API key (placeholder)"""
    # In production, implement proper API key validation
    # For now, this is a placeholder
    if settings.rbac_enabled and not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    return api_key


async def check_admin_role(api_key: str = Header(None, alias="X-API-Key")):
    """Check if user has admin role (placeholder)"""
    # In production, decode JWT and check roles
    # For now, this is a placeholder
    pass