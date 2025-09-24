from unittest.mock import patch

import numpy as np
import pytest


@pytest.mark.asyncio
async def test_identification_success(client, mock_image_bytes):
    """Test successful face identification"""
    
    with patch("services.identification_service.get_face_engine") as mock_engine, \
         patch("services.identification_service.get_quality_analyzer") as mock_quality, \
         patch("services.identification_service.get_liveness_service") as mock_liveness, \
         patch("services.identification_service.get_index") as mock_index:
        
        from tests.fixtures.mock_faces import (
            MockFaceEngine,
            MockQualityAnalyzer,
            MockLivenessService,
            MockVectorIndex,
        )
        
        # Setup mocks
        mock_engine.return_value = MockFaceEngine()
        mock_quality.return_value = MockQualityAnalyzer()
        mock_liveness.return_value = MockLivenessService()
        
        # Setup index with some embeddings
        index = MockVectorIndex()
        # Add a known embedding
        known_embedding = np.random.randn(512).astype(np.float32)
        known_embedding = known_embedding / np.linalg.norm(known_embedding)
        await index.add(known_embedding.reshape(1, -1), [0])
        mock_index.return_value = index
        
        files = [
            ("image", ("test.jpg", mock_image_bytes, "image/jpeg"))
        ]
        
        response = await client.post(
            "/api/v1/identify",
            files=files,
            data={"top_k": "5"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "matches" in data
        assert "face_quality" in data
        assert "processing_time_ms" in data
        assert isinstance(data["matches"], list)


@pytest.mark.asyncio
async def test_identification_no_match(client, mock_image_bytes):
    """Test identification with no matches"""
    
    with patch("services.identification_service.get_face_engine") as mock_engine, \
         patch("services.identification_service.get_quality_analyzer") as mock_quality, \
         patch("services.identification_service.get_liveness_service") as mock_liveness, \
         patch("services.identification_service.get_index") as mock_index:
        
        from tests.fixtures.mock_faces import (
            MockFaceEngine,
            MockQualityAnalyzer,
            MockLivenessService,
            MockVectorIndex,
        )
        
        mock_engine.return_value = MockFaceEngine()
        mock_quality.return_value = MockQualityAnalyzer()
        mock_liveness.return_value = MockLivenessService()
        
        # Empty index
        mock_index.return_value = MockVectorIndex()
        
        files = [
            ("image", ("test.jpg", mock_image_bytes, "image/jpeg"))
        ]
        
        response = await client.post(
            "/api/v1/identify",
            files=files
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "matches" in data
        assert len(data["matches"]) == 0


@pytest.mark.asyncio
async def test_verify_face(client, sample_person_id, mock_image_bytes):
    """Test face verification (1:1 matching)"""
    
    with patch("services.identification_service.get_face_engine") as mock_engine, \
         patch("services.identification_service.get_quality_analyzer") as mock_quality, \
         patch("services.identification_service.get_liveness_service") as mock_liveness, \
         patch("services.identification_service.get_index") as mock_index:
        
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
        
        files = [
            ("image", ("test.jpg", mock_image_bytes, "image/jpeg"))
        ]
        
        response = await client.post(
            f"/api/v1/verify/{sample_person_id}",
            files=files
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "verified" in data
        assert isinstance(data["verified"], bool)