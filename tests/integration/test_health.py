import pytest


@pytest.mark.asyncio
async def test_health_check(client):
    """Test health check endpoint"""
    response = await client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "version" in data
    assert "timestamp" in data
    assert "database" in data
    assert "index_status" in data
    assert "face_engine" in data


@pytest.mark.asyncio
async def test_readiness_check(client):
    """Test readiness check endpoint"""
    response = await client.get("/readiness")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "ready" in data
    assert isinstance(data["ready"], bool)


@pytest.mark.asyncio
async def test_liveness_check(client):
    """Test liveness check endpoint"""
    response = await client.get("/liveness")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "alive" in data
    assert data["alive"] is True


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    """Test metrics endpoint"""
    response = await client.get("/metrics")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    
    # Check for Prometheus metrics format
    content = response.text
    assert "# HELP" in content
    assert "# TYPE" in content