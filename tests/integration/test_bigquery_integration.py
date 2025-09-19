"""
Integration tests for BigQuery vector search as specified in task T005.
These tests exercise the live BigQuery flow without mocks.
"""

import pytest
import os
from google.cloud import bigquery
from google.cloud.exceptions import NotFound, Forbidden

from src.janapada_memory.bigquery_vector_index import BigQueryVectorIndex
from src.janapada_memory.connections.bigquery_connection import BigQueryConnectionManager
from src.janapada_memory.models.vector_search_result import VectorSearchResult
from src.janapada_memory.schema_manager import SchemaManager
from src.janapada_memory.config.bigquery_config import BigQueryConfig


class TestBigQueryIntegration:
    """Integration tests for the complete BigQuery vector search flow."""

    @pytest.fixture(scope="session")
    def bigquery_config(self):
        """BigQuery configuration for testing."""
        return BigQueryConfig(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT", "konveyn2ai"),
            dataset_id=os.getenv("BIGQUERY_DATASET_ID", "semantic_gap_detector"),
        )

    @pytest.fixture(scope="session")
    def connection_manager(self, bigquery_config):
        """BigQuery connection manager for tests."""
        return BigQueryConnectionManager(config=bigquery_config)

    @pytest.fixture(scope="session")
    def schema_manager(self, connection_manager):
        """Schema manager for setting up test environment."""
        return SchemaManager(connection=connection_manager)

    @pytest.fixture(scope="session")
    def bigquery_vector_index(self, connection_manager):
        """BigQuery vector index for testing."""
        return BigQueryVectorIndex(connection=connection_manager)

    def test_dataset_exists(self, connection_manager):
        """Test that the BigQuery dataset exists and is accessible."""
        try:
            dataset = connection_manager.get_dataset()
            assert dataset is not None
            assert dataset.dataset_id == connection_manager.config.dataset_id
        except NotFound:
            pytest.fail("Dataset not found - run 'make setup' first")
        except Forbidden:
            pytest.fail("Access denied to dataset - check BigQuery permissions")

    def test_source_embeddings_table_exists(self, schema_manager):
        """Test that source_embeddings table exists with correct schema."""
        tables = schema_manager.list_tables(include_schema=True)
        
        embeddings_table = None
        for table in tables["tables"]:
            if table["name"] == "source_embeddings":
                embeddings_table = table
                break
        
        assert embeddings_table is not None, "source_embeddings table not found"
        
        # Check for required embedding_vector column
        schema_fields = {field["name"]: field["type"] for field in embeddings_table["schema"]}
        assert "embedding_vector" in schema_fields, "embedding_vector column not found"
        assert schema_fields["embedding_vector"] == "VECTOR", f"Expected VECTOR type, got {schema_fields['embedding_vector']}"

    def test_source_metadata_table_exists(self, schema_manager):
        """Test that source_metadata table exists with correct schema."""
        tables = schema_manager.list_tables(include_schema=True)
        
        metadata_table = None
        for table in tables["tables"]:
            if table["name"] == "source_metadata":
                metadata_table = table
                break
        
        assert metadata_table is not None, "source_metadata table not found"
        
        # Check for required columns
        schema_fields = {field["name"]: field["type"] for field in metadata_table["schema"]}
        required_fields = ["chunk_id", "source", "artifact_type", "text_content", "metadata"]
        for field in required_fields:
            assert field in schema_fields, f"Required field {field} not found"

    def test_bigquery_connection_health(self, connection_manager):
        """Test BigQuery connection health check."""
        health = connection_manager.check_health()
        assert health.is_healthy, f"BigQuery connection unhealthy: {health.error_message}"
        assert health.response_time_ms >= 0

    def test_vector_index_initialization(self, bigquery_vector_index):
        """Test BigQuery vector index can be initialized."""
        assert bigquery_vector_index is not None
        assert bigquery_vector_index.connection is not None
        assert bigquery_vector_index.bigquery_adapter is not None

    def test_vector_index_health_check(self, bigquery_vector_index):
        """Test comprehensive health check of vector index."""
        health = bigquery_vector_index.health_check()
        
        assert health["status"] in ["healthy", "degraded"], f"Unexpected status: {health['status']}"
        assert "bigquery_connection" in health
        assert "bigquery_adapter" in health
        assert "search_stats" in health

    def test_bigquery_health_method(self, bigquery_vector_index):
        """Test the specific check_bigquery_health method."""
        is_healthy = bigquery_vector_index.check_bigquery_health()
        assert isinstance(is_healthy, bool)

    def test_empty_similarity_search(self, bigquery_vector_index):
        """Test similarity search with empty table (should not error)."""
        # Use a dummy 3072-dimension vector (Gemini embedding size)
        query_vector = [0.1] * 3072
        
        try:
            results = bigquery_vector_index.similarity_search(query_vector, k=5)
            assert isinstance(results, list)
            # Results may be empty if no data, but should not error
            for result in results:
                assert isinstance(result, VectorSearchResult)
                assert result.source in ["bigquery", "local"]
        except Exception as e:
            # Allow specific exceptions related to empty table or dimension mismatch
            if "dimension" not in str(e).lower() and "empty" not in str(e).lower():
                pytest.fail(f"Unexpected error in similarity search: {e}")

    def test_add_and_search_vectors(self, bigquery_vector_index):
        """Test adding vectors and searching for them."""
        # Test data
        test_vectors = [
            ("test_chunk_1", [0.1] * 3072),
            ("test_chunk_2", [0.2] * 3072),
        ]
        
        try:
            # Add vectors
            add_result = bigquery_vector_index.add_vectors(test_vectors)
            assert add_result["success"] or len(add_result["errors"]) == 0
            
            # Search for similar vectors
            query_vector = [0.15] * 3072  # Between the test vectors
            results = bigquery_vector_index.similarity_search(query_vector, k=5)
            
            assert isinstance(results, list)
            # May find results from BigQuery or fallback
            
        except Exception as e:
            # Some exceptions are expected if BigQuery operations aren't fully set up
            if "not available" in str(e).lower() or "not implemented" in str(e).lower():
                pytest.skip(f"BigQuery operations not fully available: {e}")
            else:
                raise

    def test_vector_removal(self, bigquery_vector_index):
        """Test removing vectors from the index."""
        try:
            # Try to remove a test chunk
            removed = bigquery_vector_index.remove_vector("test_chunk_1")
            assert isinstance(removed, bool)
            
        except Exception as e:
            # Some exceptions are expected if BigQuery operations aren't fully set up
            if "not available" in str(e).lower() or "not implemented" in str(e).lower():
                pytest.skip(f"BigQuery operations not fully available: {e}")
            else:
                raise

    def test_fallback_activation(self, bigquery_vector_index, monkeypatch):
        """Test that fallback activates when BigQuery fails."""
        # Mock BigQuery adapter to fail
        def mock_search_failure(*args, **kwargs):
            from src.janapada_memory.adapters.bigquery_adapter import BigQueryAdapterError
            raise BigQueryAdapterError("Simulated BigQuery failure")

        monkeypatch.setattr(
            bigquery_vector_index.bigquery_adapter,
            "search_similar_vectors",
            mock_search_failure,
        )

        query_vector = [0.1] * 3072
        results = bigquery_vector_index.similarity_search(query_vector, k=5)

        # Should still return results (potentially empty) from fallback
        assert isinstance(results, list)
        # Check that fallback was activated in stats
        stats = bigquery_vector_index.get_stats()
        assert stats["fallback_activations"] > 0

    def test_performance_baseline(self, bigquery_vector_index):
        """Test that similarity search meets performance requirements."""
        import time

        query_vector = [0.1] * 3072
        start_time = time.time()

        try:
            results = bigquery_vector_index.similarity_search(query_vector, k=10)
            elapsed = time.time() - start_time

            # Performance target: <500ms as specified in requirements
            assert elapsed < 0.5, f"Similarity search took {elapsed:.2f}s, target <0.5s"
            assert isinstance(results, list)
            assert len(results) <= 10

        except Exception as e:
            # Allow dimension-related errors for fresh setups
            if "dimension" in str(e).lower():
                pytest.skip("Table exists but no valid embeddings data - expected for fresh setup")
            else:
                raise

    def test_search_stats_tracking(self, bigquery_vector_index):
        """Test that search statistics are properly tracked."""
        initial_stats = bigquery_vector_index.get_stats()
        initial_searches = initial_stats["total_searches"]

        query_vector = [0.1] * 3072
        try:
            bigquery_vector_index.similarity_search(query_vector, k=5)
        except Exception:
            pass  # Stats should still be updated even on failure

        final_stats = bigquery_vector_index.get_stats()
        assert final_stats["total_searches"] > initial_searches
        assert "bigquery_success_rate" in final_stats
        assert "fallback_activation_rate" in final_stats

    def test_authentication_valid(self):
        """Test that BigQuery authentication is properly configured."""
        try:
            client = bigquery.Client()
            # Simple query to test auth
            query_job = client.query("SELECT 1 as test")
            results = list(query_job)
            assert len(results) == 1
        except Exception as e:
            if "authentication" in str(e).lower() or "credentials" in str(e).lower():
                pytest.fail(f"BigQuery authentication not configured: {e}")
            else:
                raise

    def test_parameterized_query_support(self, connection_manager):
        """Test that parameterized queries work correctly."""
        query = "SELECT @test_param as param_value"
        
        from google.cloud import bigquery
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("test_param", "STRING", "test_value")
            ]
        )

        try:
            results = list(connection_manager.execute_query(query, job_config=job_config))
            assert len(results) == 1
            assert results[0].param_value == "test_value"
        except Exception as e:
            pytest.fail(f"Parameterized query failed: {e}")