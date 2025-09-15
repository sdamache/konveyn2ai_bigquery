"""
Contract tests for POST /vector-store/search endpoint

These tests validate the API contract for vector similarity search operations.
They test both text and vector query types with various parameters.

IMPORTANT: These tests MUST FAIL initially (TDD Red phase)
Implementation will be done in T017-T024.
"""

import pytest
from fastapi.testclient import TestClient


class TestVectorStoreSearch:
    """Contract tests for POST /vector-store/search endpoint."""

    @pytest.fixture
    def client(self):
        """FastAPI test client - will fail until implementation exists."""
        # This import will fail until the API is implemented
        from src.api.main import app  # Will fail - no implementation yet

        return TestClient(app)

    @pytest.fixture
    def text_search_request(self):
        """Valid text-based search request."""
        return {
            "query_text": "function that calculates cosine similarity between vectors",
            "limit": 5,
            "similarity_threshold": 0.7,
            "artifact_types": ["code"],
        }

    @pytest.fixture
    def vector_search_request(self):
        """Valid vector-based search request."""
        return {
            "query_embedding": [0.1] * 768,  # 768-dimensional query vector
            "limit": 10,
            "similarity_threshold": 0.6,
        }

    def test_post_search_text_query_200(self, client, text_search_request):
        """Test text-based similarity search returns 200."""
        response = client.post("/vector-store/search", json=text_search_request)

        # Should return 200 OK
        assert response.status_code == 200

        # Should return search results
        response_data = response.json()
        assert "results" in response_data
        assert "query_embedding" in response_data
        assert "total_results" in response_data
        assert "search_time_ms" in response_data
        assert isinstance(response_data["results"], list)

    def test_post_search_vector_query_200(self, client, vector_search_request):
        """Test vector-based similarity search returns 200."""
        response = client.post("/vector-store/search", json=vector_search_request)

        # Should return 200 OK
        assert response.status_code == 200

        # Should return search results
        response_data = response.json()
        assert "results" in response_data
        assert "query_embedding" in response_data
        assert "total_results" in response_data
        assert "search_time_ms" in response_data

    def test_post_search_missing_query_400(self, client):
        """Test search without query_text or query_embedding returns 400."""
        invalid_request = {
            "limit": 5,
            "similarity_threshold": 0.7,
            # Missing both query_text and query_embedding
        }

        response = client.post("/vector-store/search", json=invalid_request)

        # Should return 400 Bad Request
        assert response.status_code == 400

        # Should return validation error
        response_data = response.json()
        assert "error" in response_data

    def test_post_search_both_queries_400(self, client):
        """Test search with both query_text and query_embedding returns 400."""
        invalid_request = {
            "query_text": "test query",
            "query_embedding": [0.1] * 768,
            "limit": 5,
        }

        response = client.post("/vector-store/search", json=invalid_request)

        # Should return 400 Bad Request (only one query type allowed)
        assert response.status_code == 400

        response_data = response.json()
        assert "error" in response_data

    def test_post_search_invalid_vector_dimensions_400(self, client):
        """Test search with wrong vector dimensions returns 400."""
        invalid_request = {
            "query_embedding": [0.1] * 512,  # Wrong dimension - should be 768
            "limit": 5,
        }

        response = client.post("/vector-store/search", json=invalid_request)

        # Should return 400 Bad Request
        assert response.status_code == 400

        response_data = response.json()
        assert "error" in response_data
        assert "768" in response_data["message"]

    def test_post_search_invalid_limit_400(self, client, text_search_request):
        """Test search with invalid limit parameter returns 400."""
        # Test limit too high
        invalid_request = text_search_request.copy()
        invalid_request["limit"] = 200

        response = client.post("/vector-store/search", json=invalid_request)
        assert response.status_code == 400

        # Test limit too low
        invalid_request["limit"] = 0
        response = client.post("/vector-store/search", json=invalid_request)
        assert response.status_code == 400

    def test_post_search_invalid_similarity_threshold_400(
        self, client, text_search_request
    ):
        """Test search with invalid similarity_threshold returns 400."""
        # Test threshold too high
        invalid_request = text_search_request.copy()
        invalid_request["similarity_threshold"] = 1.5

        response = client.post("/vector-store/search", json=invalid_request)
        assert response.status_code == 400

        # Test threshold too low
        invalid_request["similarity_threshold"] = -0.1
        response = client.post("/vector-store/search", json=invalid_request)
        assert response.status_code == 400

    def test_post_search_invalid_artifact_types_400(self, client, text_search_request):
        """Test search with invalid artifact_types returns 400."""
        invalid_request = text_search_request.copy()
        invalid_request["artifact_types"] = ["invalid_type"]

        response = client.post("/vector-store/search", json=invalid_request)

        # Should return 400 Bad Request
        assert response.status_code == 400

        response_data = response.json()
        assert "error" in response_data

    def test_post_search_response_schema(self, client, text_search_request):
        """Test search response follows expected schema."""
        response = client.post("/vector-store/search", json=text_search_request)

        assert response.status_code == 200
        response_data = response.json()

        # Validate response schema
        required_fields = [
            "results",
            "query_embedding",
            "total_results",
            "search_time_ms",
        ]
        for field in required_fields:
            assert field in response_data

        # Validate field types
        assert isinstance(response_data["results"], list)
        assert isinstance(response_data["query_embedding"], list)
        assert isinstance(response_data["total_results"], int)
        assert isinstance(response_data["search_time_ms"], int)

        # Validate query_embedding is 768-dimensional
        assert len(response_data["query_embedding"]) == 768

    def test_post_search_result_item_schema(self, client, text_search_request):
        """Test individual search result items follow expected schema."""
        response = client.post("/vector-store/search", json=text_search_request)

        assert response.status_code == 200
        response_data = response.json()

        # Validate result item schema if results exist
        if response_data["results"]:
            result = response_data["results"][0]
            required_fields = [
                "chunk_id",
                "similarity_score",
                "text_content",
                "source",
                "artifact_type",
            ]
            for field in required_fields:
                assert field in result

            # Validate field types and constraints
            assert isinstance(result["chunk_id"], str)
            assert isinstance(result["similarity_score"], float)
            assert 0.0 <= result["similarity_score"] <= 1.0
            assert isinstance(result["text_content"], str)
            assert isinstance(result["source"], str)
            assert isinstance(result["artifact_type"], str)

    def test_post_search_results_sorted_by_similarity(
        self, client, text_search_request
    ):
        """Test search results are sorted by similarity score descending."""
        # Request multiple results
        search_request = text_search_request.copy()
        search_request["limit"] = 10
        search_request["similarity_threshold"] = (
            0.1  # Lower threshold to get more results
        )

        response = client.post("/vector-store/search", json=search_request)

        assert response.status_code == 200
        response_data = response.json()

        # Results should be sorted by similarity score descending
        results = response_data["results"]
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert (
                    results[i]["similarity_score"] >= results[i + 1]["similarity_score"]
                )

    def test_post_search_similarity_threshold_filtering(
        self, client, text_search_request
    ):
        """Test results respect similarity_threshold parameter."""
        response = client.post("/vector-store/search", json=text_search_request)

        assert response.status_code == 200
        response_data = response.json()

        # All results should meet similarity threshold
        threshold = text_search_request["similarity_threshold"]
        for result in response_data["results"]:
            assert result["similarity_score"] >= threshold

    def test_post_search_artifact_type_filtering(self, client, text_search_request):
        """Test results respect artifact_types filter."""
        response = client.post("/vector-store/search", json=text_search_request)

        assert response.status_code == 200
        response_data = response.json()

        # All results should match requested artifact types
        allowed_types = text_search_request["artifact_types"]
        for result in response_data["results"]:
            assert result["artifact_type"] in allowed_types

    def test_post_search_sources_filtering(self, client, text_search_request):
        """Test search with sources filter."""
        # Add sources filter to request
        search_request = text_search_request.copy()
        search_request["sources"] = ["src/utils/similarity.py", "src/models/vector.py"]

        response = client.post("/vector-store/search", json=search_request)

        assert response.status_code == 200
        response_data = response.json()

        # All results should match requested sources
        allowed_sources = search_request["sources"]
        for result in response_data["results"]:
            assert result["source"] in allowed_sources

    def test_post_search_empty_results(self, client, text_search_request):
        """Test search returns valid schema even with no matching results."""
        # Use very high similarity threshold to get no results
        search_request = text_search_request.copy()
        search_request["similarity_threshold"] = 0.99

        response = client.post("/vector-store/search", json=search_request)

        assert response.status_code == 200
        response_data = response.json()

        # Should still return valid schema
        assert response_data["results"] == []
        assert response_data["total_results"] == 0
        assert "query_embedding" in response_data
        assert "search_time_ms" in response_data
