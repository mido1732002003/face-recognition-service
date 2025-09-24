import asyncio
import uuid
from typing import AsyncGenerator

import numpy as np
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from api.main import app
from core.database import get_db
from core.models import Base


# Test database
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def test_db():
    """Create test database"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    TestSessionLocal = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with TestSessionLocal() as session:
            yield session
    
    app.dependency_overrides[get_db] = override_get_db
    
    yield
    
    await engine.dispose()


@pytest.fixture
async def client(test_db):
    """Create test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_embedding():
    """Generate mock face embedding"""
    embedding = np.random.randn(512).astype(np.float32)
    # L2 normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


@pytest.fixture
def mock_face_data():
    """Generate mock face detection data"""
    return {
        "bbox": [100.0, 100.0, 200.0, 200.0],
        "landmarks": [[120.0, 130.0], [180.0, 130.0], [150.0, 160.0], 
                      [130.0, 190.0], [170.0, 190.0]],
        "det_score": 0.99,
        "embedding": np.random.randn(512).astype(np.float32),
    }


@pytest.fixture
def sample_person_id():
    """Generate sample person ID"""
    return f"person_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def mock_image_bytes():
    """Generate mock image bytes"""
    # Create a simple 1x1 pixel image
    from PIL import Image
    import io
    
    img = Image.new('RGB', (224, 224), color='white')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()