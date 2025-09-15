"""
Contract tests for POST /schema/tables endpoint

These tests validate the API contract for BigQuery table creation operations.

IMPORTANT: These tests MUST FAIL initially (TDD Red phase)
Implementation will be done in T015-T016.
"""

import pytest
from fastapi.testclient import TestClient


class TestSchemaManagerCreateTables:
    """Contract tests for POST /schema/tables endpoint."""

    @pytest.fixture
    def client(self):
        """FastAPI test client - will fail until implementation exists."""
        # This import will fail until the API is implemented
        from src.api.main import app  # Will fail - no implementation yet

        return TestClient(app)

    @pytest.fixture
    def valid_create_tables_request(self):
        """Valid table creation request."""
        return {
            "force_recreate": False,
            "tables": ["all"],
            "partition_expiration_days": 365,
        }

    @pytest.fixture
    def specific_tables_request(self):
        """Request to create specific tables only."""
        return {
            "force_recreate": False,
            "tables": ["source_metadata", "source_embeddings"],
            "partition_expiration_days": 180,
        }

    def test_post_create_tables_success_201(self, client, valid_create_tables_request):
        """Test successful table creation returns 201."""
        response = client.post("/schema/tables", json=valid_create_tables_request)

        # Should return 201 Created
        assert response.status_code == 201

        # Should return creation details
        response_data = response.json()
        assert "tables_created" in response_data
        assert "creation_time_ms" in response_data
        assert isinstance(response_data["tables_created"], list)
        assert isinstance(response_data["creation_time_ms"], int)

    def test_post_create_tables_already_exist_409(
        self, client, valid_create_tables_request
    ):
        """Test creating tables that already exist returns 409."""
        # First creation should succeed
        response1 = client.post("/schema/tables", json=valid_create_tables_request)
        assert response1.status_code == 201

        # Second creation without force_recreate should fail
        response2 = client.post("/schema/tables", json=valid_create_tables_request)

        # Should return 409 Conflict
        assert response2.status_code == 409

        # Should return conflict error details
        response_data = response2.json()
        assert "error" in response_data
        assert "already exist" in response_data["message"].lower()

    def test_post_create_tables_force_recreate_success_201(
        self, client, valid_create_tables_request
    ):
        """Test force recreate of existing tables returns 201."""
        # First creation
        response1 = client.post("/schema/tables", json=valid_create_tables_request)
        assert response1.status_code == 201

        # Force recreate should succeed
        force_request = valid_create_tables_request.copy()
        force_request["force_recreate"] = True

        response2 = client.post("/schema/tables", json=force_request)

        # Should return 201 Created
        assert response2.status_code == 201

        response_data = response2.json()
        assert "tables_created" in response_data
        # Should include warnings about recreation
        if "warnings" in response_data:
            assert isinstance(response_data["warnings"], list)

    def test_post_create_specific_tables_201(self, client, specific_tables_request):
        """Test creating specific tables only returns 201."""
        response = client.post("/schema/tables", json=specific_tables_request)

        # Should return 201 Created
        assert response.status_code == 201

        response_data = response.json()

        # Should only create requested tables
        created_tables = response_data["tables_created"]
        requested_tables = specific_tables_request["tables"]

        # All requested tables should be created
        created_names = [table["name"] for table in created_tables]
        for table_name in requested_tables:
            assert table_name in created_names

    def test_post_create_tables_invalid_table_name_400(self, client):
        """Test creating table with invalid name returns 400."""
        invalid_request = {"tables": ["invalid_table_name"]}

        response = client.post("/schema/tables", json=invalid_request)

        # Should return 400 Bad Request
        assert response.status_code == 400

        response_data = response.json()
        assert "error" in response_data

    def test_post_create_tables_invalid_expiration_400(self, client):
        """Test invalid partition_expiration_days returns 400."""
        invalid_request = {
            "tables": ["all"],
            "partition_expiration_days": 0,  # Invalid: must be > 0
        }

        response = client.post("/schema/tables", json=invalid_request)

        # Should return 400 Bad Request
        assert response.status_code == 400

        response_data = response.json()
        assert "error" in response_data

    def test_post_create_tables_response_schema(
        self, client, valid_create_tables_request
    ):
        """Test create tables response follows expected schema."""
        response = client.post("/schema/tables", json=valid_create_tables_request)

        assert response.status_code == 201
        response_data = response.json()

        # Validate response schema
        required_fields = ["tables_created", "creation_time_ms"]
        for field in required_fields:
            assert field in response_data

        # Validate field types
        assert isinstance(response_data["tables_created"], list)
        assert isinstance(response_data["creation_time_ms"], int)

        # Validate table info schema
        for table in response_data["tables_created"]:
            required_table_fields = [
                "name",
                "full_name",
                "schema",
                "partitioning",
                "clustering",
            ]
            for field in required_table_fields:
                assert field in table

            assert isinstance(table["name"], str)
            assert isinstance(table["full_name"], str)
            assert isinstance(table["schema"], list)
            assert isinstance(table["clustering"], list)

    def test_post_create_tables_schema_field_validation(
        self, client, valid_create_tables_request
    ):
        """Test created table schema fields are properly defined."""
        response = client.post("/schema/tables", json=valid_create_tables_request)

        assert response.status_code == 201
        response_data = response.json()

        # Check source_metadata table schema
        metadata_table = next(
            (
                t
                for t in response_data["tables_created"]
                if t["name"] == "source_metadata"
            ),
            None,
        )
        if metadata_table:
            schema_fields = metadata_table["schema"]
            field_names = [field["name"] for field in schema_fields]

            # Required fields should be present
            required_fields = [
                "chunk_id",
                "source",
                "artifact_type",
                "text_content",
                "created_at",
                "partition_date",
            ]
            for field in required_fields:
                assert field in field_names

    def test_post_create_tables_partitioning_config(
        self, client, valid_create_tables_request
    ):
        """Test created tables have correct partitioning configuration."""
        response = client.post("/schema/tables", json=valid_create_tables_request)

        assert response.status_code == 201
        response_data = response.json()

        # All tables should be partitioned by date
        for table in response_data["tables_created"]:
            partitioning = table["partitioning"]
            assert partitioning["type"] == "TIME"
            assert partitioning["field"] == "partition_date"
            assert (
                partitioning["expiration_days"]
                == valid_create_tables_request["partition_expiration_days"]
            )

    def test_get_existing_tables_200(self, client):
        """Test GET /schema/tables returns existing tables list."""
        response = client.get("/schema/tables")

        # Should return 200 OK
        assert response.status_code == 200

        # Should return tables list
        response_data = response.json()
        assert "tables" in response_data
        assert "dataset_info" in response_data
        assert isinstance(response_data["tables"], list)

    def test_get_tables_response_schema(self, client):
        """Test GET tables response follows expected schema."""
        response = client.get("/schema/tables")

        assert response.status_code == 200
        response_data = response.json()

        # Validate response schema
        required_fields = ["tables", "dataset_info"]
        for field in required_fields:
            assert field in response_data

        # Validate dataset info
        dataset_info = response_data["dataset_info"]
        dataset_fields = ["project_id", "dataset_id", "location", "created"]
        for field in dataset_fields:
            assert field in dataset_info

    def test_delete_tables_confirm_required_400(self, client):
        """Test DELETE tables without confirmation returns 400."""
        response = client.delete("/schema/tables")

        # Should return 400 Bad Request (missing confirmation)
        assert response.status_code == 400

        response_data = response.json()
        assert "error" in response_data
        assert "confirm" in response_data["message"].lower()

    def test_delete_tables_with_confirm_204(self, client):
        """Test DELETE tables with confirmation returns 204."""
        response = client.delete("/schema/tables?confirm=true")

        # Should return 204 No Content
        assert response.status_code == 204

        # Should have no response body
        assert response.content == b""

    def test_post_create_tables_empty_request_default_behavior(self, client):
        """Test POST with empty request uses default values."""
        response = client.post("/schema/tables", json={})

        # Should return 201 Created with defaults
        assert response.status_code == 201

        response_data = response.json()
        # Should create all tables by default
        assert len(response_data["tables_created"]) == 3  # All 3 tables

