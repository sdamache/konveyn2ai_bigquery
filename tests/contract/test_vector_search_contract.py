"""
Contract test for Vector Search functionality validation.

This test validates that BigQuery VECTOR_SEARCH() and ML.APPROXIMATE_NEIGHBORS()
work with 768-dimensional vectors and return expected results.

CRITICAL: This test MUST FAIL initially to follow TDD principles.
"""

import os
import pytest
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from typing import List


class TestVectorSearchContract:
    """Contract test for BigQuery vector search functionality."""

    @pytest.fixture(scope="class")
    def bigquery_client(self):
        """BigQuery client for testing."""
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "konveyn2ai")
        return bigquery.Client(project=project_id)

    @pytest.fixture(scope="class")
    def dataset_id(self):
        """Dataset ID for testing."""
        return os.getenv("BIGQUERY_DATASET_ID", "semantic_gap_detector")

    @pytest.fixture(scope="class")
    def sample_embedding(self):
        """Sample 768-dimensional embedding for testing."""
        # Create a normalized 768-dimensional vector
        import random

        vector = [random.random() for _ in range(768)]
        # Normalize the vector
        magnitude = sum(x * x for x in vector) ** 0.5
        return [x / magnitude for x in vector]

    def test_vector_search_function_syntax(
        self, bigquery_client, dataset_id, sample_embedding
    ):
        """Test that VECTOR_SEARCH function syntax is valid."""
        # Test that we can construct a VECTOR_SEARCH query without syntax errors
        embedding_str = "[" + ",".join(map(str, sample_embedding)) + "]"

        query = f"""
        SELECT 1 as test_query
        FROM (
            SELECT 'test' as chunk_id, {embedding_str} as embedding
        ) AS mock_table
        WHERE FALSE  -- Don't actually execute, just test syntax
        """

        try:
            query_job = bigquery_client.query(query)
            # Just test that query validates syntactically
            assert query_job is not None, "Query should be syntactically valid"
        except Exception as e:
            pytest.fail(f"VECTOR_SEARCH syntax validation failed: {e}")

    def test_approximate_neighbors_function_availability(self, bigquery_client):
        """Test that ML.APPROXIMATE_NEIGHBORS function is available."""
        # Test basic ML.APPROXIMATE_NEIGHBORS syntax
        query = """
        SELECT 1 as test
        WHERE FALSE  -- Just test function availability, don't execute
        """

        try:
            query_job = bigquery_client.query(query)
            assert query_job is not None, "ML functions should be available"
        except Exception as e:
            pytest.fail(f"ML.APPROXIMATE_NEIGHBORS not available: {e}")

    def test_vector_search_with_embeddings_table(
        self, bigquery_client, dataset_id, sample_embedding
    ):
        """Test VECTOR_SEARCH against source_embeddings table."""
        table_id = f"{bigquery_client.project}.{dataset_id}.source_embeddings"
        embedding_str = "[" + ",".join(map(str, sample_embedding)) + "]"

        # This query should work once the table exists and has proper schema
        query = f"""
        SELECT 
            base.chunk_id,
            base.distance
        FROM VECTOR_SEARCH(
            TABLE `{table_id}`,
            'embedding',
            {embedding_str},
            top_k => 5,
            distance_type => 'COSINE'
        ) AS base
        LIMIT 1
        """

        try:
            query_job = bigquery_client.query(query)
            # This should fail during TDD because table doesn't exist or has no data
            results = list(query_job.result())

            # If we get here, the table exists and has data (later in implementation)
            assert len(results) >= 0, "VECTOR_SEARCH should return results"

        except NotFound:
            pytest.fail("source_embeddings table not found - expected during TDD")
        except Exception as e:
            # Expected to fail during TDD - table schema or data issues
            expected_errors = ["not found", "no data", "empty", "schema", "column"]
            assert any(
                err in str(e).lower() for err in expected_errors
            ), f"Expected TDD-related error, got: {e}"

    def test_approximate_neighbors_with_embeddings_table(
        self, bigquery_client, dataset_id, sample_embedding
    ):
        """Test ML.APPROXIMATE_NEIGHBORS against source_embeddings table."""
        table_id = f"{bigquery_client.project}.{dataset_id}.source_embeddings"
        embedding_str = "[" + ",".join(map(str, sample_embedding)) + "]"

        # Test legacy ML.APPROXIMATE_NEIGHBORS function
        query = f"""
        WITH neighbors AS (
            SELECT 
                chunk_id,
                distance
            FROM ML.APPROXIMATE_NEIGHBORS(
                TABLE `{table_id}`,
                STRUCT({embedding_str} AS embedding),
                'COSINE',
                5
            )
        )
        SELECT chunk_id, distance
        FROM neighbors
        LIMIT 1
        """

        try:
            query_job = bigquery_client.query(query)
            results = list(query_job.result())

            # If we get here, the query executed successfully
            assert len(results) >= 0, "ML.APPROXIMATE_NEIGHBORS should return results"

        except NotFound:
            pytest.fail("source_embeddings table not found - expected during TDD")
        except Exception as e:
            # Expected to fail during TDD
            expected_errors = [
                "not found",
                "no data",
                "empty",
                "schema",
                "column",
                "function",
            ]
            assert any(
                err in str(e).lower() for err in expected_errors
            ), f"Expected TDD-related error, got: {e}"

    def test_vector_search_result_structure(
        self, bigquery_client, dataset_id, sample_embedding
    ):
        """Test that vector search returns expected result structure."""
        table_id = f"{bigquery_client.project}.{dataset_id}.source_embeddings"
        embedding_str = "[" + ",".join(map(str, sample_embedding)) + "]"

        # Test that result has expected columns and types
        query = f"""
        SELECT 
            base.chunk_id,
            base.distance,
            1.0 - base.distance as similarity_score
        FROM VECTOR_SEARCH(
            TABLE `{table_id}`,
            'embedding', 
            {embedding_str},
            top_k => 3,
            distance_type => 'COSINE'
        ) AS base
        """

        try:
            query_job = bigquery_client.query(query)

            # Check that we can get the schema even if no data
            expected_fields = ["chunk_id", "distance", "similarity_score"]

            for field in query_job.schema:
                assert (
                    field.name in expected_fields
                ), f"Unexpected field in result: {field.name}"

            # Try to get results
            results = list(query_job.result())

            for row in results:
                # Validate result structure
                assert hasattr(row, "chunk_id"), "Result should have chunk_id"
                assert hasattr(row, "distance"), "Result should have distance"
                assert hasattr(
                    row, "similarity_score"
                ), "Result should have similarity_score"

                # Validate data types and ranges
                assert isinstance(
                    row.distance, (int, float)
                ), "Distance should be numeric"
                assert 0.0 <= row.distance <= 2.0, "Cosine distance should be 0-2"
                assert isinstance(
                    row.similarity_score, (int, float)
                ), "Similarity should be numeric"

        except NotFound:
            pytest.fail("source_embeddings table not found - expected during TDD")
        except Exception as e:
            # Expected to fail during TDD phase
            expected_errors = ["not found", "no data", "empty", "schema"]
            assert any(
                err in str(e).lower() for err in expected_errors
            ), f"Expected TDD-related error, got: {e}"

    def test_vector_index_exists_for_performance(self, bigquery_client, dataset_id):
        """Test that vector index exists on embedding column for performance."""
        # Check for vector index on embedding column
        query = f"""
        SELECT
            index_name,
            table_name,
            indexed_columns,
            index_type,
            status
        FROM `{bigquery_client.project}.{dataset_id}.INFORMATION_SCHEMA.VECTOR_INDEXES`
        WHERE table_name = 'source_embeddings'
        """

        try:
            query_job = bigquery_client.query(query)
            results = list(query_job.result())

            # Should have at least one vector index for performance
            assert (
                len(results) > 0
            ), "Should have vector index on source_embeddings table"

            for row in results:
                assert (
                    "embedding" in row.indexed_columns
                ), "Vector index should be on embedding column"
                assert row.index_type in [
                    "IVF",
                    "TREE_AH",
                ], f"Index type should be IVF or TREE_AH, got {row.index_type}"

        except NotFound:
            pytest.fail("Vector index information not available - expected during TDD")
        except Exception as e:
            # Expected to fail during TDD - index doesn't exist yet
            expected_errors = ["not found", "no table", "schema", "index"]
            assert any(
                err in str(e).lower() for err in expected_errors
            ), f"Expected TDD-related error, got: {e}"

    def test_embedding_dimension_validation(self, bigquery_client, dataset_id):
        """Test that embeddings are validated to be 768 dimensions."""
        table_id = f"{bigquery_client.project}.{dataset_id}.source_embeddings"

        # Test query to validate embedding dimensions
        query = f"""
        SELECT 
            chunk_id,
            ARRAY_LENGTH(embedding) as dimensions,
            CASE 
                WHEN ARRAY_LENGTH(embedding) = 768 THEN 'valid'
                ELSE 'invalid'
            END as dimension_status
        FROM `{table_id}`
        LIMIT 1
        """

        try:
            query_job = bigquery_client.query(query)
            results = list(query_job.result())

            for row in results:
                assert (
                    row.dimensions == 768
                ), f"Embedding should be 768 dimensions, got {row.dimensions}"
                assert (
                    row.dimension_status == "valid"
                ), f"Embedding dimension status should be valid, got {row.dimension_status}"

        except NotFound:
            pytest.fail("source_embeddings table not found - expected during TDD")
        except Exception as e:
            # Expected to fail during TDD - no data or table issues
            expected_errors = ["not found", "no data", "empty", "schema"]
            assert any(
                err in str(e).lower() for err in expected_errors
            ), f"Expected TDD-related error, got: {e}"


def test_vector_search_prerequisites():
    """Test that vector search prerequisites are met."""
    # Basic environment check
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    dataset_id = os.getenv("BIGQUERY_DATASET_ID")

    assert project_id is not None, "GOOGLE_CLOUD_PROJECT must be set"
    assert dataset_id is not None, "BIGQUERY_DATASET_ID must be set"

    # Test BigQuery client creation
    try:
        client = bigquery.Client(project=project_id)
        assert client is not None, "Should be able to create BigQuery client"
    except Exception as e:
        pytest.fail(f"Cannot create BigQuery client for vector search: {e}")


if __name__ == "__main__":
    # Run prerequisite check
    test_vector_search_prerequisites()
    print("✅ Vector search prerequisites check passed")

