"""
Contract tests for POST /vector-store/embeddings endpoint

These tests validate the API contract for embedding insertion operations.
They test request/response schemas and status codes without implementation.

IMPORTANT: These tests MUST FAIL initially (TDD Red phase)
Implementation will be done in T017-T024.
"""

import pytest
from fastapi.testclient import TestClient
import json


class TestVectorStorePostEmbeddings:
    """Contract tests for POST /vector-store/embeddings endpoint."""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client - will fail until implementation exists."""
        # This import will fail until the API is implemented
        from src.api.main import app  # Will fail - no implementation yet
        return TestClient(app)
    
    @pytest.fixture
    def valid_embedding_request(self):
        """Valid embedding insertion request payload."""
        return {
            "chunk_id": "func_calculate_similarity_001",
            "text_content": "def calculate_similarity(vector1, vector2): return cosine_similarity(vector1, vector2)",
            "source": "src/utils/similarity.py",
            "artifact_type": "code",
            "kind": "function",
            "record_name": "calculate_similarity",
            "embedding": [0.1] * 768  # 768-dimensional vector
        }
    
    @pytest.fixture
    def invalid_embedding_request_short_vector(self):
        """Invalid request with wrong embedding dimensions."""
        return {
            "chunk_id": "func_test_001",
            "text_content": "def test(): pass",
            "source": "test.py",
            "artifact_type": "code",
            "embedding": [0.1] * 512  # Wrong dimension - should be 768
        }
    
    @pytest.fixture
    def invalid_embedding_request_missing_fields(self):
        """Invalid request with missing required fields."""
        return {
            "chunk_id": "func_test_001",
            # Missing text_content, source, artifact_type, embedding
        }

    def test_post_embeddings_success_201(self, client, valid_embedding_request):
        """Test successful embedding insertion returns 201."""
        response = client.post(
            "/vector-store/embeddings",
            json=valid_embedding_request
        )
        
        # Should return 201 Created
        assert response.status_code == 201
        
        # Should return insertion confirmation
        response_data = response.json()
        assert response_data["chunk_id"] == "func_calculate_similarity_001"
        assert response_data["status"] == "inserted"
        assert "created_at" in response_data

    def test_post_embeddings_invalid_dimensions_400(self, client, invalid_embedding_request_short_vector):
        """Test embedding with wrong dimensions returns 400."""
        response = client.post(
            "/vector-store/embeddings",
            json=invalid_embedding_request_short_vector
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400
        
        # Should return error details
        response_data = response.json()
        assert "error" in response_data
        assert "embedding must contain exactly 768 dimensions" in response_data["message"]

    def test_post_embeddings_missing_fields_400(self, client, invalid_embedding_request_missing_fields):
        """Test request with missing required fields returns 400."""
        response = client.post(
            "/vector-store/embeddings",
            json=invalid_embedding_request_missing_fields
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400
        
        # Should return validation error details
        response_data = response.json()
        assert "error" in response_data

    def test_post_embeddings_duplicate_chunk_id_409(self, client, valid_embedding_request):
        """Test duplicate chunk_id returns 409 Conflict."""
        # First insertion should succeed
        response1 = client.post(
            "/vector-store/embeddings",
            json=valid_embedding_request
        )
        assert response1.status_code == 201
        
        # Second insertion with same chunk_id should fail
        response2 = client.post(
            "/vector-store/embeddings",
            json=valid_embedding_request
        )
        
        # Should return 409 Conflict
        assert response2.status_code == 409
        
        # Should return conflict error details
        response_data = response2.json()
        assert "error" in response_data
        assert "already exists" in response_data["message"]

    def test_post_embeddings_invalid_artifact_type_400(self, client, valid_embedding_request):
        """Test invalid artifact_type returns 400."""
        invalid_request = valid_embedding_request.copy()
        invalid_request["artifact_type"] = "invalid_type"
        
        response = client.post(
            "/vector-store/embeddings",
            json=invalid_request
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400
        
        # Should return validation error
        response_data = response.json()
        assert "error" in response_data
        assert "artifact_type must be one of" in response_data["message"]

    def test_post_embeddings_text_content_too_short_400(self, client, valid_embedding_request):
        """Test text_content shorter than 10 characters returns 400."""
        invalid_request = valid_embedding_request.copy()
        invalid_request["text_content"] = "short"  # Less than 10 characters
        
        response = client.post(
            "/vector-store/embeddings",
            json=invalid_request
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400
        
        # Should return validation error
        response_data = response.json()
        assert "error" in response_data
        assert "text_content must be at least 10 characters" in response_data["message"]

    def test_post_embeddings_invalid_embedding_values_400(self, client, valid_embedding_request):
        """Test embedding with invalid values (NaN, Inf) returns 400."""
        invalid_request = valid_embedding_request.copy()
        invalid_request["embedding"] = [float('nan')] * 768  # NaN values
        
        response = client.post(
            "/vector-store/embeddings",
            json=invalid_request
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400
        
        # Should return validation error
        response_data = response.json()
        assert "error" in response_data
        assert "finite numeric values" in response_data["message"]

    def test_post_embeddings_content_type_validation(self, client, valid_embedding_request):
        """Test endpoint requires application/json content type."""
        response = client.post(
            "/vector-store/embeddings",
            data=json.dumps(valid_embedding_request),  # Send as form data instead of JSON
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        # Should return 400 or 415 for wrong content type
        assert response.status_code in [400, 415]

    def test_post_embeddings_response_schema(self, client, valid_embedding_request):
        """Test successful response follows expected schema."""
        response = client.post(
            "/vector-store/embeddings",
            json=valid_embedding_request
        )
        
        assert response.status_code == 201
        response_data = response.json()
        
        # Validate response schema
        required_fields = ["chunk_id", "status", "created_at"]
        for field in required_fields:
            assert field in response_data
        
        # Validate field types
        assert isinstance(response_data["chunk_id"], str)
        assert isinstance(response_data["status"], str)
        assert isinstance(response_data["created_at"], str)