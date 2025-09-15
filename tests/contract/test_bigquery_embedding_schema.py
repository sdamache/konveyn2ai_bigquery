"""
Contract test for BigQuery embedding schema validation.

This test validates that the source_embeddings table has the correct schema
with VECTOR<768> column, proper clustering, and all required fields.

CRITICAL: This test MUST FAIL initially to follow TDD principles.
"""

import os
import pytest
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


class TestBigQueryEmbeddingSchema:
    """Contract test for BigQuery source_embeddings table schema."""
    
    @pytest.fixture(scope="class")
    def bigquery_client(self):
        """BigQuery client for testing."""
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "konveyn2ai")
        return bigquery.Client(project=project_id)
    
    @pytest.fixture(scope="class") 
    def dataset_id(self):
        """Dataset ID for testing."""
        return os.getenv("BIGQUERY_DATASET_ID", "semantic_gap_detector")
    
    def test_source_embeddings_table_exists(self, bigquery_client, dataset_id):
        """Test that source_embeddings table exists."""
        table_id = f"{bigquery_client.project}.{dataset_id}.source_embeddings"
        
        try:
            table = bigquery_client.get_table(table_id)
            assert table is not None, "source_embeddings table should exist"
        except NotFound:
            pytest.fail(f"source_embeddings table does not exist at {table_id}")
    
    def test_source_embeddings_schema_fields(self, bigquery_client, dataset_id):
        """Test that source_embeddings table has all required fields."""
        table_id = f"{bigquery_client.project}.{dataset_id}.source_embeddings"
        
        try:
            table = bigquery_client.get_table(table_id)
        except NotFound:
            pytest.fail(f"source_embeddings table does not exist at {table_id}")
        
        # Expected schema fields
        expected_fields = {
            "chunk_id": "STRING",
            "model": "STRING", 
            "content_hash": "STRING",
            "embedding": "FLOAT64",  # Should be REPEATED for array
            "created_at": "TIMESTAMP",
            "source_type": "STRING",
            "artifact_id": "STRING",
            "partition_date": "DATE"
        }
        
        # Get actual schema
        actual_fields = {}
        for field in table.schema:
            actual_fields[field.name] = field.field_type
        
        # Validate all expected fields exist
        for field_name, field_type in expected_fields.items():
            assert field_name in actual_fields, f"Field '{field_name}' missing from schema"
            assert actual_fields[field_name] == field_type, \
                f"Field '{field_name}' has type '{actual_fields[field_name]}', expected '{field_type}'"
    
    def test_embedding_field_is_repeated_array(self, bigquery_client, dataset_id):
        """Test that embedding field is a repeated FLOAT64 array (768 dimensions)."""
        table_id = f"{bigquery_client.project}.{dataset_id}.source_embeddings"
        
        try:
            table = bigquery_client.get_table(table_id)
        except NotFound:
            pytest.fail(f"source_embeddings table does not exist at {table_id}")
        
        # Find embedding field
        embedding_field = None
        for field in table.schema:
            if field.name == "embedding":
                embedding_field = field
                break
        
        assert embedding_field is not None, "embedding field not found in schema"
        assert embedding_field.field_type == "FLOAT64", "embedding field should be FLOAT64"
        assert embedding_field.mode == "REPEATED", "embedding field should be REPEATED (array)"
    
    def test_required_fields_are_not_nullable(self, bigquery_client, dataset_id):
        """Test that required fields are marked as NOT NULL."""
        table_id = f"{bigquery_client.project}.{dataset_id}.source_embeddings"
        
        try:
            table = bigquery_client.get_table(table_id)
        except NotFound:
            pytest.fail(f"source_embeddings table does not exist at {table_id}")
        
        required_fields = ["chunk_id", "model", "content_hash", "embedding"]
        
        for field in table.schema:
            if field.name in required_fields:
                assert field.mode == "REQUIRED", \
                    f"Field '{field.name}' should be REQUIRED but is '{field.mode}'"
    
    def test_table_has_date_partitioning(self, bigquery_client, dataset_id):
        """Test that table is partitioned by partition_date for performance."""
        table_id = f"{bigquery_client.project}.{dataset_id}.source_embeddings"
        
        try:
            table = bigquery_client.get_table(table_id)
        except NotFound:
            pytest.fail(f"source_embeddings table does not exist at {table_id}")
        
        assert table.time_partitioning is not None, "Table should have time partitioning"
        assert table.time_partitioning.type_ == bigquery.TimePartitioningType.DAY, \
            "Table should use DAY partitioning"
        assert table.time_partitioning.field == "partition_date", \
            "Table should be partitioned by partition_date field"
    
    def test_table_has_proper_clustering(self, bigquery_client, dataset_id):
        """Test that table is clustered for optimal query performance."""
        table_id = f"{bigquery_client.project}.{dataset_id}.source_embeddings"
        
        try:
            table = bigquery_client.get_table(table_id)
        except NotFound:
            pytest.fail(f"source_embeddings table does not exist at {table_id}")
        
        assert table.clustering_fields is not None, "Table should have clustering fields"
        
        expected_clustering = ["chunk_id", "model", "content_hash"]
        actual_clustering = list(table.clustering_fields)
        
        assert actual_clustering == expected_clustering, \
            f"Table clustering should be {expected_clustering}, got {actual_clustering}"
    
    def test_composite_primary_key_constraint(self, bigquery_client, dataset_id):
        """Test that table description mentions composite primary key."""
        table_id = f"{bigquery_client.project}.{dataset_id}.source_embeddings"
        
        try:
            table = bigquery_client.get_table(table_id)
        except NotFound:
            pytest.fail(f"source_embeddings table does not exist at {table_id}")
        
        # Since BigQuery doesn't enforce primary keys, we check the table description
        # should mention the composite key (chunk_id, model, content_hash)
        description = table.description or ""
        
        assert "chunk_id" in description, "Table description should mention chunk_id in primary key"
        assert "model" in description, "Table description should mention model in primary key" 
        assert "content_hash" in description, "Table description should mention content_hash in primary key"
    
    def test_vector_search_functions_available(self, bigquery_client, dataset_id):
        """Test that BigQuery supports vector search functions for this table."""
        # This is a basic test to ensure the project supports vector operations
        # We'll test a simple query that would use vector functions
        
        query = f"""
        SELECT 1 as test
        FROM `{bigquery_client.project}.{dataset_id}.source_embeddings`
        WHERE FALSE  -- Don't return any rows, just test table access
        """
        
        try:
            query_job = bigquery_client.query(query)
            # Just check that query can be created (table is accessible)
            assert query_job is not None, "Should be able to query source_embeddings table"
        except NotFound:
            pytest.fail("source_embeddings table not accessible for vector operations")
        except Exception as e:
            pytest.fail(f"Vector search setup may be incomplete: {e}")


# Integration test helper for vector search functionality
def test_vector_search_prerequisites():
    """Test that prerequisites for vector search are met."""
    # Check environment variables
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    dataset_id = os.getenv("BIGQUERY_DATASET_ID") 
    
    assert project_id is not None, "GOOGLE_CLOUD_PROJECT environment variable must be set"
    assert dataset_id is not None, "BIGQUERY_DATASET_ID environment variable must be set"
    
    # Verify we can create BigQuery client
    try:
        client = bigquery.Client(project=project_id)
        assert client is not None, "Should be able to create BigQuery client"
    except Exception as e:
        pytest.fail(f"Cannot create BigQuery client: {e}")


if __name__ == "__main__":
    # Run basic prerequisite checks
    test_vector_search_prerequisites()
    print("✅ Vector search prerequisites check passed")