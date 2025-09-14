"""
Contract tests for GET /vector-store/embeddings endpoint

These tests validate the API contract for embedding listing operations.
They test pagination, filtering, and response schemas.

IMPORTANT: These tests MUST FAIL initially (TDD Red phase)
Implementation will be done in T017-T024.
"""

import pytest
from fastapi.testclient import TestClient


class TestVectorStoreGetEmbeddings:
    """Contract tests for GET /vector-store/embeddings endpoint."""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client - will fail until implementation exists."""
        # This import will fail until the API is implemented
        from src.api.main import app  # Will fail - no implementation yet
        return TestClient(app)

    def test_get_embeddings_default_pagination_200(self, client):
        """Test GET embeddings with default pagination returns 200."""
        response = client.get("/vector-store/embeddings")
        
        # Should return 200 OK
        assert response.status_code == 200
        
        # Should return paginated list
        response_data = response.json()
        assert "embeddings" in response_data
        assert "total_count" in response_data
        assert "limit" in response_data
        assert "offset" in response_data
        assert isinstance(response_data["embeddings"], list)

    def test_get_embeddings_custom_pagination_200(self, client):
        """Test GET embeddings with custom pagination parameters."""
        response = client.get("/vector-store/embeddings?limit=5&offset=10")
        
        # Should return 200 OK
        assert response.status_code == 200
        
        # Should respect pagination parameters
        response_data = response.json()
        assert response_data["limit"] == 5
        assert response_data["offset"] == 10

    def test_get_embeddings_filter_by_artifact_type_200(self, client):
        """Test GET embeddings filtered by artifact_type."""
        response = client.get("/vector-store/embeddings?artifact_type=code")
        
        # Should return 200 OK
        assert response.status_code == 200
        
        # Should return filtered results
        response_data = response.json()
        assert "embeddings" in response_data
        
        # All returned embeddings should have artifact_type = "code"
        for embedding in response_data["embeddings"]:
            assert embedding["artifact_type"] == "code"

    def test_get_embeddings_filter_by_created_since_200(self, client):
        """Test GET embeddings filtered by created_since timestamp."""
        created_since = "2025-01-01T00:00:00Z"
        response = client.get(f"/vector-store/embeddings?created_since={created_since}")
        
        # Should return 200 OK
        assert response.status_code == 200
        
        # Should return filtered results
        response_data = response.json()
        assert "embeddings" in response_data

    def test_get_embeddings_invalid_limit_400(self, client):
        """Test GET embeddings with invalid limit parameter returns 400."""
        # Test limit too high
        response = client.get("/vector-store/embeddings?limit=200")
        assert response.status_code == 400
        
        # Test limit too low
        response = client.get("/vector-store/embeddings?limit=0")
        assert response.status_code == 400
        
        # Test invalid limit type
        response = client.get("/vector-store/embeddings?limit=invalid")
        assert response.status_code == 400

    def test_get_embeddings_invalid_offset_400(self, client):
        """Test GET embeddings with invalid offset parameter returns 400."""
        # Test negative offset
        response = client.get("/vector-store/embeddings?offset=-1")
        assert response.status_code == 400
        
        # Test invalid offset type
        response = client.get("/vector-store/embeddings?offset=invalid")
        assert response.status_code == 400

    def test_get_embeddings_invalid_artifact_type_400(self, client):
        """Test GET embeddings with invalid artifact_type returns 400."""
        response = client.get("/vector-store/embeddings?artifact_type=invalid_type")
        
        # Should return 400 Bad Request
        assert response.status_code == 400
        
        # Should return validation error
        response_data = response.json()
        assert "error" in response_data

    def test_get_embeddings_invalid_date_format_400(self, client):
        """Test GET embeddings with invalid created_since format returns 400."""
        response = client.get("/vector-store/embeddings?created_since=invalid-date")
        
        # Should return 400 Bad Request
        assert response.status_code == 400

    def test_get_embeddings_response_schema(self, client):
        """Test GET embeddings response follows expected schema."""
        response = client.get("/vector-store/embeddings?limit=2")
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Validate response schema
        required_fields = ["embeddings", "total_count", "limit", "offset"]
        for field in required_fields:
            assert field in response_data
        
        # Validate field types
        assert isinstance(response_data["embeddings"], list)
        assert isinstance(response_data["total_count"], int)
        assert isinstance(response_data["limit"], int)
        assert isinstance(response_data["offset"], int)
        
        # Validate embedding item schema if any embeddings exist
        if response_data["embeddings"]:
            embedding = response_data["embeddings"][0]
            expected_fields = [
                "chunk_id", "text_content", "source", "artifact_type", 
                "kind", "api_path", "record_name", "embedding", "created_at"
            ]
            for field in expected_fields:
                if field in embedding:  # Some fields are optional
                    if field == "embedding":
                        assert isinstance(embedding[field], list)
                        assert len(embedding[field]) == 768
                    elif field == "created_at":
                        assert isinstance(embedding[field], str)
                    else:
                        assert isinstance(embedding[field], str)

    def test_get_embeddings_next_offset_calculation(self, client):
        """Test next_offset is calculated correctly in pagination."""
        response = client.get("/vector-store/embeddings?limit=10&offset=0")
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Should include next_offset if more results available
        if response_data["total_count"] > response_data["limit"]:
            assert "next_offset" in response_data
            expected_next = response_data["offset"] + response_data["limit"]
            assert response_data["next_offset"] == expected_next
        else:
            # Should be null if no more results
            assert response_data.get("next_offset") is None

    def test_get_embeddings_empty_results(self, client):
        """Test GET embeddings returns valid schema even with empty results."""
        # Request from high offset to likely get empty results
        response = client.get("/vector-store/embeddings?offset=999999")
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Should still return valid schema
        assert response_data["embeddings"] == []
        assert "total_count" in response_data
        assert "limit" in response_data
        assert "offset" in response_data

    def test_get_embeddings_multiple_filters(self, client):
        """Test GET embeddings with multiple filter parameters."""
        response = client.get(
            "/vector-store/embeddings"
            "?artifact_type=code"
            "&limit=5"
            "&offset=0"
            "&created_since=2025-01-01T00:00:00Z"
        )
        
        # Should return 200 OK and handle multiple filters
        assert response.status_code == 200
        
        response_data = response.json()
        assert "embeddings" in response_data
        assert response_data["limit"] == 5
        assert response_data["offset"] == 0