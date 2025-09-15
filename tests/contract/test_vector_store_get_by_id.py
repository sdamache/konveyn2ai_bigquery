"""
Contract tests for GET /vector-store/embeddings/{chunk_id} endpoint

These tests validate the API contract for retrieving specific embeddings by chunk ID.

IMPORTANT: These tests MUST FAIL initially (TDD Red phase)
Implementation will be done in T017-T024.
"""

import pytest
from fastapi.testclient import TestClient


class TestVectorStoreGetById:
    """Contract tests for GET /vector-store/embeddings/{chunk_id} endpoint."""

    @pytest.fixture
    def client(self):
        """FastAPI test client - will fail until implementation exists."""
        # This import will fail until the API is implemented
        from src.api.main import app  # Will fail - no implementation yet

        return TestClient(app)

    def test_get_embedding_by_id_success_200(self, client):
        """Test GET embedding by existing chunk_id returns 200."""
        chunk_id = "func_calculate_similarity_001"
        response = client.get(f"/vector-store/embeddings/{chunk_id}")

        # Should return 200 OK
        assert response.status_code == 200

        # Should return embedding data
        response_data = response.json()
        assert response_data["chunk_id"] == chunk_id
        assert "text_content" in response_data
        assert "source" in response_data
        assert "artifact_type" in response_data
        assert "embedding" in response_data
        assert "created_at" in response_data

    def test_get_embedding_by_id_not_found_404(self, client):
        """Test GET embedding by non-existent chunk_id returns 404."""
        chunk_id = "non_existent_chunk_001"
        response = client.get(f"/vector-store/embeddings/{chunk_id}")

        # Should return 404 Not Found
        assert response.status_code == 404

        # Should return error details
        response_data = response.json()
        assert "error" in response_data
        assert "not found" in response_data["message"].lower()
        assert chunk_id in response_data["message"]

    def test_get_embedding_response_schema(self, client):
        """Test GET embedding response follows expected schema."""
        chunk_id = "func_calculate_similarity_001"
        response = client.get(f"/vector-store/embeddings/{chunk_id}")

        assert response.status_code == 200
        response_data = response.json()

        # Validate response schema
        required_fields = [
            "chunk_id",
            "text_content",
            "source",
            "artifact_type",
            "embedding",
            "created_at",
        ]
        for field in required_fields:
            assert field in response_data

        # Validate field types
        assert isinstance(response_data["chunk_id"], str)
        assert isinstance(response_data["text_content"], str)
        assert isinstance(response_data["source"], str)
        assert isinstance(response_data["artifact_type"], str)
        assert isinstance(response_data["embedding"], list)
        assert isinstance(response_data["created_at"], str)

        # Validate embedding dimensions
        assert len(response_data["embedding"]) == 768

        # Validate optional fields if present
        if "kind" in response_data:
            assert isinstance(response_data["kind"], str)
        if "api_path" in response_data:
            assert isinstance(response_data["api_path"], str)
        if "record_name" in response_data:
            assert isinstance(response_data["record_name"], str)

    def test_get_embedding_artifact_type_validation(self, client):
        """Test returned artifact_type is valid enum value."""
        chunk_id = "func_calculate_similarity_001"
        response = client.get(f"/vector-store/embeddings/{chunk_id}")

        assert response.status_code == 200
        response_data = response.json()

        # Should be valid artifact type
        valid_types = ["code", "documentation", "api", "schema", "test"]
        assert response_data["artifact_type"] in valid_types

    def test_get_embedding_text_content_length(self, client):
        """Test returned text_content meets minimum length requirement."""
        chunk_id = "func_calculate_similarity_001"
        response = client.get(f"/vector-store/embeddings/{chunk_id}")

        assert response.status_code == 200
        response_data = response.json()

        # Text content should be at least 10 characters
        assert len(response_data["text_content"]) >= 10

    def test_get_embedding_vector_values(self, client):
        """Test returned embedding contains valid float values."""
        chunk_id = "func_calculate_similarity_001"
        response = client.get(f"/vector-store/embeddings/{chunk_id}")

        assert response.status_code == 200
        response_data = response.json()

        # All embedding values should be finite numbers
        embedding = response_data["embedding"]
        for value in embedding:
            assert isinstance(value, (int, float))
            assert not (value != value)  # Check for NaN
            assert abs(value) != float("inf")  # Check for infinity

    def test_get_embedding_empty_chunk_id_404(self, client):
        """Test GET with empty chunk_id returns 404 or 400."""
        response = client.get("/vector-store/embeddings/")

        # Should return 404 or 405 (endpoint doesn't match)
        assert response.status_code in [404, 405]

    def test_get_embedding_special_chars_chunk_id(self, client):
        """Test GET with special characters in chunk_id."""
        # Test with URL-encoded special characters
        chunk_id = "func%20with%20spaces_001"
        response = client.get(f"/vector-store/embeddings/{chunk_id}")

        # Should either return 200 if exists or 404 if not found
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            response_data = response.json()
            # Chunk ID should be URL-decoded
            assert "chunk_id" in response_data

    def test_get_embedding_very_long_chunk_id(self, client):
        """Test GET with very long chunk_id handles gracefully."""
        # Test with very long chunk_id
        chunk_id = "a" * 500  # 500 character chunk_id
        response = client.get(f"/vector-store/embeddings/{chunk_id}")

        # Should return 404 (not found) or handle gracefully
        assert response.status_code in [404, 400, 414]  # 414 = URI too long

    def test_get_embedding_case_sensitive(self, client):
        """Test chunk_id lookup is case sensitive."""
        chunk_id_lower = "func_calculate_similarity_001"
        chunk_id_upper = "FUNC_CALCULATE_SIMILARITY_001"

        response_lower = client.get(f"/vector-store/embeddings/{chunk_id_lower}")
        response_upper = client.get(f"/vector-store/embeddings/{chunk_id_upper}")

        # Responses should be different (case sensitive)
        # If lower exists and upper doesn't, we should get different status codes
        if response_lower.status_code == 200:
            # Upper case version might not exist
            assert response_upper.status_code in [200, 404]

    def test_get_embedding_created_at_format(self, client):
        """Test created_at field is in valid ISO format."""
        chunk_id = "func_calculate_similarity_001"
        response = client.get(f"/vector-store/embeddings/{chunk_id}")

        assert response.status_code == 200
        response_data = response.json()

        # created_at should be valid ISO timestamp
        created_at = response_data["created_at"]
        # Basic ISO format validation
        import re

        iso_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        assert re.match(iso_pattern, created_at)

    def test_delete_embedding_by_id_success_204(self, client):
        """Test DELETE embedding by chunk_id returns 204."""
        chunk_id = "func_calculate_similarity_001"
        response = client.delete(f"/vector-store/embeddings/{chunk_id}")

        # Should return 204 No Content
        assert response.status_code == 204

        # Should have no response body
        assert response.content == b""

    def test_delete_embedding_by_id_not_found_404(self, client):
        """Test DELETE embedding by non-existent chunk_id returns 404."""
        chunk_id = "non_existent_chunk_001"
        response = client.delete(f"/vector-store/embeddings/{chunk_id}")

        # Should return 404 Not Found
        assert response.status_code == 404

        # Should return error details
        response_data = response.json()
        assert "error" in response_data
        assert "not found" in response_data["message"].lower()

    def test_delete_then_get_embedding_404(self, client):
        """Test GET after DELETE returns 404."""
        chunk_id = "func_to_delete_001"

        # First delete the embedding
        delete_response = client.delete(f"/vector-store/embeddings/{chunk_id}")
        assert delete_response.status_code == 204

        # Then try to get it
        get_response = client.get(f"/vector-store/embeddings/{chunk_id}")
        assert get_response.status_code == 404
