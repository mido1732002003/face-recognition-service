import numpy as np
import pytest

from tests.fixtures.mock_faces import MockFaceEngine, MockQualityAnalyzer


def test_mock_face_detection():
    """Test mock face detection"""
    engine = MockFaceEngine()
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Test single face detection
    faces = engine.detect_faces(image)
    assert len(faces) == 1
    assert "bbox" in faces[0]
    assert "embedding" in faces[0]
    assert faces[0]["embedding"].shape == (512,)
    
    # Test no face detection
    engine.should_detect_face = False
    faces = engine.detect_faces(image)
    assert len(faces) == 0
    
    # Test multiple faces
    engine.should_detect_face = True
    engine.should_detect_multiple = True
    faces = engine.detect_faces(image)
    assert len(faces) == 2


def test_mock_face_processing():
    """Test mock face processing"""
    engine = MockFaceEngine()
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Test successful processing
    face_data, embedding = engine.process_single_face(image)
    assert face_data is not None
    assert embedding.shape == (512,)
    assert np.allclose(np.linalg.norm(embedding), 1.0)  # L2 normalized
    
    # Test no face error
    engine.should_detect_face = False
    from core.exceptions import NoFaceDetectedException
    
    with pytest.raises(NoFaceDetectedException):
        engine.process_single_face(image)
    
    # Test multiple faces error
    engine.should_detect_face = True
    engine.should_detect_multiple = True
    from core.exceptions import MultipleFacesDetectedException
    
    with pytest.raises(MultipleFacesDetectedException):
        engine.process_single_face(image, allow_multiple=False)


def test_mock_quality_analyzer():
    """Test mock quality analyzer"""
    analyzer = MockQualityAnalyzer()
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    face_data = {"bbox": [100, 100, 200, 200]}
    
    result = analyzer.analyze_face(image, face_data)
    
    assert "overall_score" in result
    assert "quality_level" in result
    assert "metrics" in result
    assert "is_acceptable" in result
    assert 0 <= result["overall_score"] <= 1
    assert result["is_acceptable"] == (result["overall_score"] >= analyzer.quality_threshold)


def test_similarity_computation():
    """Test similarity computation"""
    engine = MockFaceEngine()
    
    # Create two random embeddings
    emb1 = np.random.randn(512).astype(np.float32)
    emb1 = emb1 / np.linalg.norm(emb1)
    
    emb2 = np.random.randn(512).astype(np.float32)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    # Test similarity
    similarity = engine.compute_similarity(emb1, emb2)
    assert -1 <= similarity <= 1  # Cosine similarity range
    
    # Test self-similarity
    self_similarity = engine.compute_similarity(emb1, emb1)
    assert np.isclose(self_similarity, 1.0)