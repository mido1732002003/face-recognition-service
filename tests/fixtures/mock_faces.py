from typing import Any, Tuple

import numpy as np


class MockFaceEngine:
    """Mock face engine for testing"""
    
    def __init__(self):
        self.should_detect_face = True
        self.should_detect_multiple = False
        self.face_quality = 0.8
    
    def detect_faces(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Mock face detection"""
        if not self.should_detect_face:
            return []
        
        if self.should_detect_multiple:
            return [
                self._create_mock_face(i) for i in range(2)
            ]
        
        return [self._create_mock_face(0)]
    
    def _create_mock_face(self, index: int) -> dict[str, Any]:
        """Create mock face data"""
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        return {
            "bbox": [100.0 + index * 50, 100.0, 200.0 + index * 50, 200.0],
            "landmarks": [[120.0, 130.0], [180.0, 130.0], [150.0, 160.0], 
                         [130.0, 190.0], [170.0, 190.0]],
            "det_score": 0.99 - index * 0.1,
            "embedding": embedding,
            "face": None,
        }
    
    def process_single_face(
        self, image: np.ndarray, allow_multiple: bool = False
    ) -> Tuple[dict[str, Any], np.ndarray]:
        """Mock single face processing"""
        faces = self.detect_faces(image)
        
        if not faces:
            from core.exceptions import NoFaceDetectedException
            raise NoFaceDetectedException()
        
        if len(faces) > 1 and not allow_multiple:
            from core.exceptions import MultipleFacesDetectedException
            raise MultipleFacesDetectedException(len(faces))
        
        face = faces[0]
        return face, face["embedding"]
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Mock similarity computation"""
        return float(np.dot(emb1, emb2))


class MockQualityAnalyzer:
    """Mock face quality analyzer"""
    
    def __init__(self):
        self.quality_threshold = 0.5
        self.quality_score = 0.8
    
    def analyze_face(self, image: np.ndarray, face_data: dict[str, Any]) -> dict[str, Any]:
        """Mock quality analysis"""
        return {
            "overall_score": self.quality_score,
            "quality_level": "good",
            "metrics": {
                "size_score": 0.7,
                "pose_score": 0.8,
                "sharpness_score": 0.9,
                "brightness_score": 0.8,
            },
            "is_acceptable": self.quality_score >= self.quality_threshold,
        }


class MockLivenessService:
    """Mock liveness service"""
    
    def __init__(self):
        self.is_live = True
        self.confidence = 0.95
    
    def check_liveness(self, image: np.ndarray, face_data: dict[str, Any]) -> dict[str, Any]:
        """Mock liveness check"""
        return {
            "is_live": self.is_live,
            "confidence": self.confidence,
            "method": "mock",
            "details": {"message": "Mock liveness check"},
        }


class MockVectorIndex:
    """Mock vector index for testing"""
    
    def __init__(self):
        self.embeddings = {}
        self._next_id = 0
        self.dimension = 512
    
    async def add(self, embeddings: np.ndarray, ids: list[int]) -> None:
        """Add embeddings to mock index"""
        for emb, idx in zip(embeddings, ids):
            self.embeddings[idx] = emb
    
    async def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search in mock index"""
        if not self.embeddings:
            return np.array([]), np.array([])
        
        # Compute similarities
        similarities = []
        indices = []
        
        for idx, emb in self.embeddings.items():
            sim = float(np.dot(query_embedding.flatten(), emb.flatten()))
            similarities.append(sim)
            indices.append(idx)
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1][:k]
        
        return (
            np.array([similarities[i] for i in sorted_indices]),
            np.array([indices[i] for i in sorted_indices]),
        )
    
    async def remove(self, ids: list[int]) -> None:
        """Remove from mock index"""
        for idx in ids:
            self.embeddings.pop(idx, None)
    
    async def save(self, path: str) -> None:
        """Mock save"""
        pass
    
    async def load(self, path: str) -> None:
        """Mock load"""
        pass
    
    async def clear(self) -> None:
        """Clear mock index"""
        self.embeddings.clear()
    
    async def rebuild(self) -> None:
        """Mock rebuild"""
        pass
    
    def size(self) -> int:
        """Get mock index size"""
        return len(self.embeddings)
    
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension
    
    def get_stats(self) -> dict[str, Any]:
        """Get mock stats"""
        return {
            "type": "mock",
            "size": self.size(),
            "dimension": self.dimension,
        }