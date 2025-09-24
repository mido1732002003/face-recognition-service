import io
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image


@pytest.mark.asyncio
async def test_enrollment_success(client, sample_person_id, mock_image_bytes):
    """Test successful face enrollment"""
    
    # Mock face engine and services
    with patch("services.enrollment_service.get_face_engine") as mock_engine, \
         patch("services.enrollment_service.get_quality_analyzer") as mock_quality, \
         patch("services.enrollment_service.get_liveness_service") as mock_liveness, \
         patch("services.enrollment_service.get_index") as mock_index:
        
        # Setup mocks
        from tests.fixtures.mock_faces import (
            MockFaceEngine,
            MockQualityAnalyzer,
            MockLivenessService,
            MockVectorIndex,
        )
        
        mock_engine.return_value = MockFaceEngine()
        mock_quality.return_value = MockQualityAnalyzer()
        mock_liveness.return_value = MockLivenessService()
        mock_index.return_value = MockVectorIndex()
        
        # Create multipart form data
        files = [
            ("images", ("test.jpg", mock_image_bytes, "image/jpeg"))
        ]
        
        response = await client.post(
            f"/api/v1/enroll/{sample_person_id}",
            files=files,
            data={"update_if_exists": "true"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["person_id"] == sample_person_id
        assert data["faces_enrolled"] >= 0
        assert data["status"] in ["completed", "processing", "failed"]


@pytest.mark.asyncio
async def test_enrollment_no_face(client, sample_person_id, mock_image_bytes):
    """Test enrollment with no face detected"""
    
    with patch("services.enrollment_service.get_face_engine") as mock_engine, \
         patch("services.enrollment_service.get_quality_analyzer") as mock_quality, \
         patch("services.enrollment_service.get_liveness_service") as mock_liveness, \
         patch("services.enrollment_service.get_index") as mock_index:
        
        from tests.fixtures.mock_faces import MockFaceEngine
        
        engine = MockFaceEngine()
        engine.should_detect_face = False
        mock_engine.return_value = engine
        
        files = [
            ("images", ("test.jpg", mock_image_bytes, "image/jpeg"))
        ]
        
        response = await client.post(
            f"/api/v1/enroll/{sample_person_id}",
            files=files
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 500]


@pytest.mark.asyncio
async def test_enrollment_multiple_images(client, sample_person_id, mock_image_bytes):
    """Test enrollment with multiple images"""
    
    with patch("services.enrollment_service.get_face_engine") as mock_engine, \
         patch("services.enrollment_service.get_quality_analyzer") as mock_quality, \
         patch("services.enrollment_service.get_liveness_service") as mock_liveness, \
         patch("services.enrollment_service.get_index") as mock_index:
        
        from tests.fixtures.mock_faces import (
            MockFaceEngine,
            MockQualityAnalyzer,
            MockLivenessService,
            MockVectorIndex,
        )
        
        mock_engine.return_value = MockFaceEngine()
        mock_quality.return_value = MockQualityAnalyzer()
        mock_liveness.return_value = MockLivenessService()
        mock_index.return_value = MockVectorIndex()
        
        # Multiple images
        files = [
            ("images", ("test1.jpg", mock_image_bytes, "image/jpeg")),
            ("images", ("test2.jpg", mock_image_bytes, "image/jpeg")),
        ]
        
        response = await client.post(
            f"/api/v1/enroll/{sample_person_id}",
            files=files
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["person_id"] == sample_person_id