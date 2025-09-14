"""
Contract tests for POST /schema/validate endpoint

These tests validate the API contract for BigQuery schema validation operations.

IMPORTANT: These tests MUST FAIL initially (TDD Red phase)
Implementation will be done in T015-T016.
"""

import pytest
from fastapi.testclient import TestClient


class TestSchemaManagerValidate:
    """Contract tests for POST /schema/validate endpoint."""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client - will fail until implementation exists."""
        # This import will fail until the API is implemented
        from src.api.main import app  # Will fail - no implementation yet
        return TestClient(app)
    
    @pytest.fixture
    def basic_validation_request(self):
        """Basic validation request with default options."""
        return {
            "validate_data": False,
            "check_indexes": True,
            "sample_size": 1000
        }
    
    @pytest.fixture
    def full_validation_request(self):
        """Full validation request including data validation."""
        return {
            "validate_data": True,
            "check_indexes": True,
            "sample_size": 5000
        }

    def test_post_validate_schema_success_200(self, client, basic_validation_request):
        """Test successful schema validation returns 200."""
        response = client.post(
            "/schema/validate",
            json=basic_validation_request
        )
        
        # Should return 200 OK
        assert response.status_code == 200
        
        # Should return validation results
        response_data = response.json()
        assert "overall_status" in response_data
        assert "tables" in response_data
        assert "indexes" in response_data
        assert response_data["overall_status"] in ["VALID", "INVALID", "WARNING"]

    def test_post_validate_schema_empty_request_default_200(self, client):
        """Test validation with empty request uses defaults."""
        response = client.post("/schema/validate", json={})
        
        # Should return 200 OK with default validation
        assert response.status_code == 200
        
        response_data = response.json()
        assert "overall_status" in response_data
        assert "tables" in response_data
        assert "indexes" in response_data

    def test_post_validate_schema_with_data_validation_200(self, client, full_validation_request):
        """Test schema validation with data validation enabled."""
        response = client.post(
            "/schema/validate",
            json=full_validation_request
        )
        
        # Should return 200 OK (may take longer due to data validation)
        assert response.status_code == 200
        
        response_data = response.json()
        assert "data_quality" in response_data
        assert response_data["overall_status"] in ["VALID", "INVALID", "WARNING"]

    def test_post_validate_schema_invalid_sample_size_400(self, client):
        """Test validation with invalid sample_size returns 400."""
        invalid_request = {
            "validate_data": True,
            "sample_size": 0  # Invalid: must be > 0
        }
        
        response = client.post(
            "/schema/validate",
            json=invalid_request
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400
        
        response_data = response.json()
        assert "error" in response_data

    def test_post_validate_schema_sample_size_too_large_400(self, client):
        """Test validation with sample_size too large returns 400."""
        invalid_request = {
            "validate_data": True,
            "sample_size": 50000  # Invalid: > 10000
        }
        
        response = client.post(
            "/schema/validate",
            json=invalid_request
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400

    def test_post_validate_schema_response_schema(self, client, basic_validation_request):
        """Test validation response follows expected schema."""
        response = client.post(
            "/schema/validate",
            json=basic_validation_request
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Validate response schema
        required_fields = ["overall_status", "tables", "indexes", "recommendations"]
        for field in required_fields:
            assert field in response_data
        
        # Validate field types
        assert isinstance(response_data["overall_status"], str)
        assert isinstance(response_data["tables"], dict)
        assert isinstance(response_data["indexes"], dict)
        assert isinstance(response_data["recommendations"], list)

    def test_post_validate_schema_table_validation_results(self, client, basic_validation_request):
        """Test table validation results schema."""
        response = client.post(
            "/schema/validate",
            json=basic_validation_request
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Validate table validation results
        tables = response_data["tables"]
        expected_tables = ["source_metadata", "source_embeddings", "gap_metrics"]
        
        for table_name in expected_tables:
            if table_name in tables:
                table_result = tables[table_name]
                required_fields = ["exists", "schema_valid", "row_count", "data_quality_score"]
                
                for field in required_fields:
                    if field in table_result:
                        if field == "exists" or field == "schema_valid":
                            assert isinstance(table_result[field], bool)
                        elif field == "row_count":
                            assert isinstance(table_result[field], int)
                        elif field == "data_quality_score":
                            assert isinstance(table_result[field], float)
                            assert 0.0 <= table_result[field] <= 1.0

    def test_post_validate_schema_index_validation_results(self, client, basic_validation_request):
        """Test index validation results schema."""
        response = client.post(
            "/schema/validate",
            json=basic_validation_request
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Validate index validation results
        indexes = response_data["indexes"]
        
        # Check embedding index if it exists
        if "embedding_similarity_index" in indexes:
            index_result = indexes["embedding_similarity_index"]
            required_fields = ["exists", "status", "performance_score"]
            
            for field in required_fields:
                if field in index_result:
                    if field == "exists":
                        assert isinstance(index_result[field], bool)
                    elif field == "status":
                        assert isinstance(index_result[field], str)
                        valid_statuses = ["CREATING", "ACTIVE", "ERROR", "DISABLED"]
                        assert index_result[field] in valid_statuses
                    elif field == "performance_score":
                        assert isinstance(index_result[field], float)
                        assert 0.0 <= index_result[field] <= 1.0

    def test_post_validate_schema_with_data_quality_report(self, client, full_validation_request):
        """Test validation with data quality report included."""
        response = client.post(
            "/schema/validate",
            json=full_validation_request
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Should include data quality report
        assert "data_quality" in response_data
        data_quality = response_data["data_quality"]
        
        # Validate data quality report schema
        expected_sections = ["embedding_quality", "metadata_quality", "referential_integrity"]
        for section in expected_sections:
            if section in data_quality:
                assert isinstance(data_quality[section], dict)

    def test_post_validate_schema_embedding_quality_checks(self, client, full_validation_request):
        """Test embedding quality validation checks."""
        response = client.post(
            "/schema/validate",
            json=full_validation_request
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Check embedding quality section
        if "data_quality" in response_data:
            embedding_quality = response_data["data_quality"].get("embedding_quality", {})
            
            quality_checks = [
                "dimension_consistency", "nan_count", 
                "infinity_count", "zero_vector_count"
            ]
            
            for check in quality_checks:
                if check in embedding_quality:
                    if check == "dimension_consistency":
                        assert isinstance(embedding_quality[check], bool)
                    else:
                        assert isinstance(embedding_quality[check], int)
                        assert embedding_quality[check] >= 0

    def test_post_validate_schema_metadata_quality_checks(self, client, full_validation_request):
        """Test metadata quality validation checks."""
        response = client.post(
            "/schema/validate",
            json=full_validation_request
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Check metadata quality section
        if "data_quality" in response_data:
            metadata_quality = response_data["data_quality"].get("metadata_quality", {})
            
            quality_checks = [
                "missing_fields", "duplicate_chunk_ids", "invalid_artifact_types"
            ]
            
            for check in quality_checks:
                if check in metadata_quality:
                    assert isinstance(metadata_quality[check], int)
                    assert metadata_quality[check] >= 0

    def test_post_validate_schema_referential_integrity_checks(self, client, full_validation_request):
        """Test referential integrity validation checks."""
        response = client.post(
            "/schema/validate",
            json=full_validation_request
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Check referential integrity section
        if "data_quality" in response_data:
            ref_integrity = response_data["data_quality"].get("referential_integrity", {})
            
            integrity_checks = ["orphaned_embeddings", "missing_embeddings"]
            
            for check in integrity_checks:
                if check in ref_integrity:
                    assert isinstance(ref_integrity[check], int)
                    assert ref_integrity[check] >= 0

    def test_post_validate_schema_recommendations(self, client, basic_validation_request):
        """Test validation recommendations are provided."""
        response = client.post(
            "/schema/validate",
            json=basic_validation_request
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Should include recommendations
        recommendations = response_data["recommendations"]
        assert isinstance(recommendations, list)
        
        # Each recommendation should be a string
        for recommendation in recommendations:
            assert isinstance(recommendation, str)

    def test_post_validate_schema_invalid_overall_status(self, client, basic_validation_request):
        """Test validation when schema is invalid."""
        # This test assumes there might be schema issues
        response = client.post(
            "/schema/validate",
            json=basic_validation_request
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        # If overall status is INVALID, should have specific issues
        if response_data["overall_status"] == "INVALID":
            # Should have issues in tables or indexes
            tables_valid = all(
                table.get("schema_valid", True) and table.get("exists", False)
                for table in response_data["tables"].values()
            )
            indexes_valid = all(
                index.get("exists", False) and index.get("status") == "ACTIVE"
                for index in response_data["indexes"].values()
            )
            
            # At least one should be invalid
            assert not (tables_valid and indexes_valid)

    def test_post_validate_schema_check_indexes_false(self, client):
        """Test validation with index checking disabled."""
        no_index_request = {
            "validate_data": False,
            "check_indexes": False
        }
        
        response = client.post(
            "/schema/validate",
            json=no_index_request
        )
        
        # Should return 200 OK
        assert response.status_code == 200
        
        response_data = response.json()
        # Indexes section might be empty or contain placeholder data
        assert "indexes" in response_data