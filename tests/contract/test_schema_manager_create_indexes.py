"""
Contract tests for POST /schema/indexes endpoint

These tests validate the API contract for BigQuery vector index creation operations.

IMPORTANT: These tests MUST FAIL initially (TDD Red phase)
Implementation will be done in T015-T016.
"""

import pytest
from fastapi.testclient import TestClient


class TestSchemaManagerCreateIndexes:
    """Contract tests for POST /schema/indexes endpoint."""

    @pytest.fixture
    def client(self):
        """FastAPI test client - will fail until implementation exists."""
        # This import will fail until the API is implemented
        from src.api.main import app  # Will fail - no implementation yet

        return TestClient(app)

    @pytest.fixture
    def valid_create_indexes_request(self):
        """Valid index creation request."""
        return {
            "indexes": [
                {
                    "name": "embedding_similarity_index",
                    "table": "source_embeddings",
                    "column": "embedding",
                    "index_type": "IVF",
                    "distance_type": "COSINE",
                    "options": {"num_lists": 1000, "fraction_lists_to_search": 0.01},
                }
            ]
        }

    @pytest.fixture
    def multiple_indexes_request(self):
        """Request to create multiple indexes."""
        return {
            "indexes": [
                {
                    "name": "embedding_similarity_index",
                    "table": "source_embeddings",
                    "column": "embedding",
                    "index_type": "IVF",
                    "distance_type": "COSINE",
                    "options": {"num_lists": 1000},
                },
                {
                    "name": "metadata_lookup_index",
                    "table": "source_metadata",
                    "column": "artifact_type",
                    "index_type": "BTREE",
                },
            ]
        }

    def test_post_create_indexes_success_201(
        self, client, valid_create_indexes_request
    ):
        """Test successful index creation returns 201."""
        response = client.post("/schema/indexes", json=valid_create_indexes_request)

        # Should return 201 Created
        assert response.status_code == 201

        # Should return creation details
        response_data = response.json()
        assert "indexes_created" in response_data
        assert "creation_time_ms" in response_data
        assert isinstance(response_data["indexes_created"], list)
        assert isinstance(response_data["creation_time_ms"], int)

    def test_post_create_indexes_invalid_table_400(self, client):
        """Test creating index on non-existent table returns 400."""
        invalid_request = {
            "indexes": [
                {
                    "name": "invalid_index",
                    "table": "non_existent_table",
                    "column": "some_column",
                    "index_type": "IVF",
                    "distance_type": "COSINE",
                }
            ]
        }

        response = client.post("/schema/indexes", json=invalid_request)

        # Should return 400 Bad Request
        assert response.status_code == 400

        response_data = response.json()
        assert "error" in response_data

    def test_post_create_indexes_invalid_column_400(self, client):
        """Test creating index on non-existent column returns 400."""
        invalid_request = {
            "indexes": [
                {
                    "name": "invalid_index",
                    "table": "source_embeddings",
                    "column": "non_existent_column",
                    "index_type": "IVF",
                    "distance_type": "COSINE",
                }
            ]
        }

        response = client.post("/schema/indexes", json=invalid_request)

        # Should return 400 Bad Request
        assert response.status_code == 400

        response_data = response.json()
        assert "error" in response_data

    def test_post_create_indexes_invalid_index_type_400(self, client):
        """Test creating index with invalid type returns 400."""
        invalid_request = {
            "indexes": [
                {
                    "name": "invalid_index",
                    "table": "source_embeddings",
                    "column": "embedding",
                    "index_type": "INVALID_TYPE",
                    "distance_type": "COSINE",
                }
            ]
        }

        response = client.post("/schema/indexes", json=invalid_request)

        # Should return 400 Bad Request
        assert response.status_code == 400

        response_data = response.json()
        assert "error" in response_data

    def test_post_create_indexes_invalid_distance_type_400(self, client):
        """Test creating vector index with invalid distance type returns 400."""
        invalid_request = {
            "indexes": [
                {
                    "name": "invalid_index",
                    "table": "source_embeddings",
                    "column": "embedding",
                    "index_type": "IVF",
                    "distance_type": "INVALID_DISTANCE",
                }
            ]
        }

        response = client.post("/schema/indexes", json=invalid_request)

        # Should return 400 Bad Request
        assert response.status_code == 400

        response_data = response.json()
        assert "error" in response_data

    def test_post_create_indexes_invalid_num_lists_400(self, client):
        """Test creating IVF index with invalid num_lists returns 400."""
        invalid_request = {
            "indexes": [
                {
                    "name": "invalid_index",
                    "table": "source_embeddings",
                    "column": "embedding",
                    "index_type": "IVF",
                    "distance_type": "COSINE",
                    "options": {"num_lists": 6000},  # Too high - max is 5000
                }
            ]
        }

        response = client.post("/schema/indexes", json=invalid_request)

        # Should return 400 Bad Request
        assert response.status_code == 400

        response_data = response.json()
        assert "error" in response_data

    def test_post_create_indexes_response_schema(
        self, client, valid_create_indexes_request
    ):
        """Test create indexes response follows expected schema."""
        response = client.post("/schema/indexes", json=valid_create_indexes_request)

        assert response.status_code == 201
        response_data = response.json()

        # Validate response schema
        required_fields = ["indexes_created", "creation_time_ms"]
        for field in required_fields:
            assert field in response_data

        # Validate field types
        assert isinstance(response_data["indexes_created"], list)
        assert isinstance(response_data["creation_time_ms"], int)

        # Validate index info schema
        for index in response_data["indexes_created"]:
            required_index_fields = ["name", "table", "status", "coverage_percentage"]
            for field in required_index_fields:
                assert field in index

            assert isinstance(index["name"], str)
            assert isinstance(index["table"], str)
            assert isinstance(index["status"], str)
            assert index["status"] in ["CREATING", "ACTIVE", "ERROR"]

    def test_post_create_multiple_indexes_201(self, client, multiple_indexes_request):
        """Test creating multiple indexes returns 201."""
        response = client.post("/schema/indexes", json=multiple_indexes_request)

        # Should return 201 Created
        assert response.status_code == 201

        response_data = response.json()

        # Should create all requested indexes
        created_indexes = response_data["indexes_created"]
        requested_count = len(multiple_indexes_request["indexes"])
        assert len(created_indexes) == requested_count

    def test_get_existing_indexes_200(self, client):
        """Test GET /schema/indexes returns existing indexes list."""
        response = client.get("/schema/indexes")

        # Should return 200 OK
        assert response.status_code == 200

        # Should return indexes list
        response_data = response.json()
        assert "indexes" in response_data
        assert isinstance(response_data["indexes"], list)

    def test_get_indexes_response_schema(self, client):
        """Test GET indexes response follows expected schema."""
        response = client.get("/schema/indexes")

        assert response.status_code == 200
        response_data = response.json()

        # Validate response schema
        assert "indexes" in response_data
        assert isinstance(response_data["indexes"], list)

        # Validate index status schema if indexes exist
        for index in response_data["indexes"]:
            required_fields = [
                "name",
                "table",
                "column",
                "status",
                "index_type",
                "distance_type",
                "coverage_percentage",
            ]
            for field in required_fields:
                if field in index:  # Some fields are optional
                    assert isinstance(index[field], (str, float, int))

            # Validate enum values
            if "status" in index:
                valid_statuses = ["CREATING", "ACTIVE", "ERROR", "DISABLED"]
                assert index["status"] in valid_statuses

    def test_post_create_indexes_default_options(self, client):
        """Test creating index with default options."""
        default_request = {
            "indexes": [
                {
                    "name": "default_index",
                    "table": "source_embeddings",
                    "column": "embedding",
                    # No index_type, distance_type, or options specified
                }
            ]
        }

        response = client.post("/schema/indexes", json=default_request)

        # Should return 201 Created with defaults applied
        assert response.status_code == 201

        response_data = response.json()
        created_index = response_data["indexes_created"][0]

        # Should have used default values (if specified in contract)
        assert created_index["name"] == "default_index"

    def test_post_create_indexes_tree_ah_type(self, client):
        """Test creating TREE_AH index type."""
        tree_ah_request = {
            "indexes": [
                {
                    "name": "tree_ah_index",
                    "table": "source_embeddings",
                    "column": "embedding",
                    "index_type": "TREE_AH",
                    "distance_type": "COSINE",
                }
            ]
        }

        response = client.post("/schema/indexes", json=tree_ah_request)

        # Should return 201 Created
        assert response.status_code == 201

        response_data = response.json()
        created_index = response_data["indexes_created"][0]
        assert "tree_ah" in created_index["name"].lower()

    def test_post_create_indexes_fraction_lists_validation(self, client):
        """Test fraction_lists_to_search parameter validation."""
        # Test invalid fraction (> 1.0)
        invalid_request = {
            "indexes": [
                {
                    "name": "invalid_fraction_index",
                    "table": "source_embeddings",
                    "column": "embedding",
                    "index_type": "IVF",
                    "distance_type": "COSINE",
                    "options": {"fraction_lists_to_search": 1.5},  # Invalid: > 1.0
                }
            ]
        }

        response = client.post("/schema/indexes", json=invalid_request)

        # Should return 400 Bad Request
        assert response.status_code == 400

    def test_post_create_indexes_empty_request_default(self, client):
        """Test POST with empty indexes array returns success."""
        empty_request = {"indexes": []}

        response = client.post("/schema/indexes", json=empty_request)

        # Should return 201 Created with no indexes
        assert response.status_code == 201

        response_data = response.json()
        assert response_data["indexes_created"] == []
