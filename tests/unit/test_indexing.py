import numpy as np
import pytest

from tests.fixtures.mock_faces import MockVectorIndex


@pytest.mark.asyncio
async def test_mock_index_add_search():
    """Test mock index add and search operations"""
    index = MockVectorIndex()
    
    # Add embeddings
    embeddings = np.random.randn(3, 512).astype(np.float32)
    for i in range(3):
        embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])
    
    ids = [0, 1, 2]
    await index.add(embeddings, ids)
    
    assert index.size() == 3
    
    # Search
    query = embeddings[0]
    distances, indices = await index.search(query, k=2)
    
    assert len(distances) == 2
    assert len(indices) == 2
    assert indices[0] == 0  # First result should be exact match
    assert np.isclose(distances[0], 1.0)  # Perfect similarity


@pytest.mark.asyncio
async def test_mock_index_remove():
    """Test mock index remove operation"""
    index = MockVectorIndex()
    
    # Add embeddings
    embeddings = np.random.randn(3, 512).astype(np.float32)
    ids = [0, 1, 2]
    await index.add(embeddings, ids)
    
    assert index.size() == 3
    
    # Remove
    await index.remove([1])
    assert index.size() == 2
    
    # Search should not find removed embedding
    distances, indices = await index.search(embeddings[1], k=3)
    assert 1 not in indices


@pytest.mark.asyncio
async def test_mock_index_clear():
    """Test mock index clear operation"""
    index = MockVectorIndex()
    
    # Add embeddings
    embeddings = np.random.randn(3, 512).astype(np.float32)
    ids = [0, 1, 2]
    await index.add(embeddings, ids)
    
    assert index.size() == 3
    
    # Clear
    await index.clear()
    assert index.size() == 0
    
    # Search should return empty
    distances, indices = await index.search(embeddings[0], k=1)
    assert len(distances) == 0
    assert len(indices) == 0


@pytest.mark.asyncio
async def test_mock_index_empty_search():
    """Test searching in empty index"""
    index = MockVectorIndex()
    
    query = np.random.randn(512).astype(np.float32)
    distances, indices = await index.search(query, k=5)
    
    assert len(distances) == 0
    assert len(indices) == 0