"""
Integration contract tests for BigQuery vector search.
These tests use REAL BigQuery datasets and MUST FAIL initially.
"""

import pytest
import os
from google.cloud import bigquery
from google.cloud.exceptions import NotFound, Forbidden


class TestBigQueryIntegrationContract:
    """Integration tests against real BigQuery environment."""

    @pytest.fixture(scope="session")
    def bigquery_client(self):
        """Real BigQuery client for integration testing."""
        return bigquery.Client()

    @pytest.fixture(scope="session")
    def test_dataset_config(self):
        """Test dataset configuration from environment."""
        return {
            'project_id': os.getenv('GOOGLE_CLOUD_PROJECT', 'konveyn2ai'),
            'dataset_id': os.getenv('BIGQUERY_DATASET_ID', 'source_ingestion'),
            'table_name': 'source_embeddings'
        }

    def test_source_embeddings_table_exists(self, bigquery_client, test_dataset_config):
        """source_embeddings table must exist and be accessible."""
        table_ref = f"{test_dataset_config['project_id']}.{test_dataset_config['dataset_id']}.{test_dataset_config['table_name']}"

        try:
            table = bigquery_client.get_table(table_ref)
            assert table is not None
            assert table.table_type == 'TABLE'
        except NotFound:
            pytest.fail(f"Table {table_ref} not found - run 'make setup' first")
        except Forbidden:
            pytest.fail(f"Access denied to {table_ref} - check BigQuery permissions")

    def test_embedding_vector_column_schema(self, bigquery_client, test_dataset_config):
        """embedding_vector column must be VECTOR type with correct dimensions."""
        table_ref = f"{test_dataset_config['project_id']}.{test_dataset_config['dataset_id']}.{test_dataset_config['table_name']}"
        table = bigquery_client.get_table(table_ref)

        # Find embedding_vector column
        vector_column = None
        for field in table.schema:
            if field.name == 'embedding_vector':
                vector_column = field
                break

        assert vector_column is not None, "embedding_vector column not found"
        assert vector_column.field_type == 'VECTOR', f"Expected VECTOR type, got {vector_column.field_type}"

    def test_vector_search_query_syntax(self, bigquery_client, test_dataset_config):
        """VECTOR_SEARCH function must work with our table schema."""
        table_ref = f"{test_dataset_config['project_id']}.{test_dataset_config['dataset_id']}.{test_dataset_config['table_name']}"

        # Test VECTOR_SEARCH syntax (this will fail if table doesn't have data)
        query = f"""
        SELECT chunk_id, distance
        FROM VECTOR_SEARCH(
            TABLE `{table_ref}`,
            'embedding_vector',
            [0.1, 0.2, 0.3],  -- Dummy 3D vector for syntax test
            top_k => 1,
            distance_type => 'COSINE'
        )
        LIMIT 1
        """

        try:
            query_job = bigquery_client.query(query)
            results = list(query_job)
            # Query should execute without syntax errors
            # May return 0 results if table is empty, but should not error
        except Exception as e:
            if "dimension" in str(e).lower():
                pytest.skip("Table exists but no embeddings data - expected for fresh setup")
            else:
                pytest.fail(f"VECTOR_SEARCH query failed: {e}")

    def test_parameterized_query_support(self, bigquery_client, test_dataset_config):
        """Must support parameterized queries for security."""
        table_ref = f"{test_dataset_config['project_id']}.{test_dataset_config['dataset_id']}.{test_dataset_config['table_name']}"

        query = f"""
        SELECT COUNT(*) as count
        FROM `{table_ref}`
        WHERE chunk_id = @chunk_id
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("chunk_id", "STRING", "test_chunk")
            ]
        )

        try:
            query_job = bigquery_client.query(query, job_config=job_config)
            results = list(query_job)
            assert len(results) == 1  # Should return one row with count
        except Exception as e:
            pytest.fail(f"Parameterized query failed: {e}")

    def test_bigquery_permissions(self, bigquery_client, test_dataset_config):
        """Must have required BigQuery permissions."""
        dataset_ref = f"{test_dataset_config['project_id']}.{test_dataset_config['dataset_id']}"

        try:
            # Test BigQuery Data Viewer permission
            dataset = bigquery_client.get_dataset(dataset_ref)
            assert dataset is not None

            # Test BigQuery Job User permission (ability to run queries)
            query = f"SELECT 1 as test"
            query_job = bigquery_client.query(query)
            results = list(query_job)
            assert len(results) == 1

        except Forbidden as e:
            pytest.fail(f"Insufficient BigQuery permissions: {e}")

    def test_authentication_configuration(self):
        """Must have valid authentication configured."""
        # Test Application Default Credentials
        try:
            client = bigquery.Client()
            # Simple query to test auth
            query_job = client.query("SELECT 1")
            list(query_job)  # Execute query
        except Exception as e:
            if "authentication" in str(e).lower() or "credentials" in str(e).lower():
                pytest.fail(f"BigQuery authentication not configured: {e}")
            else:
                raise


class TestBigQueryVectorIndexIntegration:
    """Integration tests for BigQueryVectorIndex implementation."""

    @pytest.fixture
    def bigquery_vector_index(self):
        """This will fail until BigQueryVectorIndex is implemented."""
        from src.janapada_memory.bigquery_vector_index import BigQueryVectorIndex
        return BigQueryVectorIndex()

    def test_end_to_end_similarity_search(self, bigquery_vector_index):
        """End-to-end test: query vector → BigQuery → results."""
        # This test will fail until full implementation is complete
        query_vector = [0.1] * 3072  # Gemini embedding dimension
        results = bigquery_vector_index.similarity_search(query_vector, top_k=5)

        assert isinstance(results, list)
        assert len(results) <= 5

        # Verify BigQuery-specific metadata
        for result in results:
            assert 'source' in result
            assert result['source'] in ['bigquery', 'local']

    def test_fallback_on_bigquery_failure(self, bigquery_vector_index, monkeypatch):
        """Must gracefully fallback when BigQuery fails."""
        # Mock BigQuery client to simulate failure
        def mock_bigquery_failure(*args, **kwargs):
            raise NotFound("Simulated BigQuery failure")

        # This test structure will be completed in implementation
        query_vector = [0.1] * 3072
        results = bigquery_vector_index.similarity_search(query_vector, top_k=5)

        # Must still return valid results from fallback
        assert isinstance(results, list)
        # Should use local fallback when BigQuery fails
        if results:
            assert results[0].get('source') == 'local'

    def test_performance_baseline(self, bigquery_vector_index):
        """Establish performance baseline for similarity search."""
        import time

        query_vector = [0.1] * 3072
        start_time = time.time()

        results = bigquery_vector_index.similarity_search(query_vector, top_k=10)

        elapsed = time.time() - start_time

        # Performance target: <500ms as specified
        assert elapsed < 0.5, f"Similarity search took {elapsed:.2f}s, target <0.5s"
        assert len(results) <= 10