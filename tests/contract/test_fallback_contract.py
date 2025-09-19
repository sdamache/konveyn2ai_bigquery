"""
Fallback behavior contract tests for BigQuery Memory Adapter.

This test suite verifies that the BigQueryVectorIndex implementation
gracefully falls back to local vector search when BigQuery operations fail.

CRITICAL: These tests MUST FAIL initially (TDD RED phase) before implementation.
"""

import pytest
from unittest.mock import Mock, patch
from google.cloud.exceptions import NotFound, Forbidden
from google.api_core.exceptions import GoogleAPICallError


class TestBigQueryFallbackContract:
    """Contract tests for BigQuery fallback behavior."""

    @pytest.fixture
    def bigquery_vector_index(self):
        """This will fail until BigQueryVectorIndex is implemented."""
        from src.janapada_memory.bigquery_vector_index import BigQueryVectorIndex

        return BigQueryVectorIndex()

    def test_fallback_on_table_not_found(self, bigquery_vector_index):
        """Must fallback to local search when BigQuery table is not found."""
        query_vector = [0.1] * 3072  # Gemini embedding dimension

        # Mock BigQuery adapter to raise NotFound exception
        with patch.object(
            bigquery_vector_index.bigquery_adapter, "search_similar_vectors"
        ) as mock_search:
            mock_search.side_effect = NotFound("Table not found")

            # Should still return results from local fallback
            results = bigquery_vector_index.similarity_search(query_vector, k=5)

            assert isinstance(results, list)
            # Verify fallback was used by checking VectorSearchResult objects
            if results:
                from src.janapada_memory.models.vector_search_result import (
                    VectorSearchResult,
                )

                assert isinstance(results[0], VectorSearchResult)
                assert results[0].source == "local"
                assert results[0].fallback_reason is not None
                assert "NotFound" in results[0].fallback_reason

    def test_fallback_on_permission_denied(self, bigquery_vector_index):
        """Must fallback to local search when BigQuery access is forbidden."""
        query_vector = [0.1] * 3072

        with patch.object(
            bigquery_vector_index.bigquery_adapter, "search_similar_vectors"
        ) as mock_search:
            mock_search.side_effect = Forbidden("Access denied")

            results = bigquery_vector_index.similarity_search(query_vector, k=5)

            assert isinstance(results, list)
            if results:
                from src.janapada_memory.models.vector_search_result import (
                    VectorSearchResult,
                )

                assert isinstance(results[0], VectorSearchResult)
                assert results[0].source == "local"
                assert results[0].fallback_reason is not None
                assert "Forbidden" in results[0].fallback_reason

    def test_fallback_on_connection_error(self, bigquery_vector_index):
        """Must fallback to local search when BigQuery connection fails."""
        query_vector = [0.1] * 3072

        with patch.object(
            bigquery_vector_index.bigquery_adapter, "search_similar_vectors"
        ) as mock_search:
            from src.janapada_memory.adapters.bigquery_adapter import (
                BigQueryAdapterError,
            )

            mock_search.side_effect = BigQueryAdapterError("Connection failed")

            results = bigquery_vector_index.similarity_search(query_vector, k=5)

            assert isinstance(results, list)
            if results:
                from src.janapada_memory.models.vector_search_result import (
                    VectorSearchResult,
                )

                assert isinstance(results[0], VectorSearchResult)
                assert results[0].source == "local"
                assert results[0].fallback_reason is not None

    def test_fallback_preserves_interface_contract(self, bigquery_vector_index):
        """Fallback must preserve the same interface contract as normal operation."""
        query_vector = [0.1] * 3072

        # Simulate BigQuery failure
        with patch.object(
            bigquery_vector_index.bigquery_adapter, "search_similar_vectors"
        ) as mock_search:
            mock_search.side_effect = NotFound("Simulated failure")

            # Test different k values
            for k in [1, 5, 10]:
                results = bigquery_vector_index.similarity_search(query_vector, k=k)

                assert isinstance(results, list)
                assert len(results) <= k

                # Verify result format consistency
                for result in results:
                    from src.janapada_memory.models.vector_search_result import (
                        VectorSearchResult,
                    )

                    assert isinstance(result, VectorSearchResult)
                    assert hasattr(result, "chunk_id")
                    assert hasattr(result, "distance")
                    assert isinstance(result.chunk_id, str)
                    assert isinstance(result.distance, (int, float))

    def test_fallback_maintains_distance_ordering(self, bigquery_vector_index):
        """Fallback results must maintain distance ordering requirement."""
        query_vector = [0.1] * 3072

        with patch.object(
            bigquery_vector_index.bigquery_adapter, "search_similar_vectors"
        ) as mock_search:
            mock_search.side_effect = NotFound("Simulated failure")

            results = bigquery_vector_index.similarity_search(query_vector, k=10)

            if len(results) > 1:
                distances = [r.distance for r in results]
                assert distances == sorted(
                    distances
                ), "Fallback results must be ordered by distance"

    def test_fallback_activation_logging(self, bigquery_vector_index, caplog):
        """Must log fallback activation with structured information."""
        query_vector = [0.1] * 3072

        with patch.object(bigquery_vector_index, "_bigquery_client") as mock_client:
            mock_client.query.side_effect = NotFound("Table missing")

            results = bigquery_vector_index.similarity_search(query_vector, top_k=5)

            # Verify structured logging
            assert "fallback" in caplog.text.lower()
            assert "bigquery" in caplog.text.lower()
            assert "local" in caplog.text.lower()

    def test_fallback_correlation_id_tracking(self, bigquery_vector_index):
        """Must include correlation IDs for fallback operation tracking."""
        query_vector = [0.1] * 3072

        with patch.object(bigquery_vector_index, "_bigquery_client") as mock_client:
            mock_client.query.side_effect = NotFound("Simulated failure")

            results = bigquery_vector_index.similarity_search(query_vector, top_k=5)

            if results:
                # Verify correlation tracking metadata
                assert "correlation_id" in results[0] or "request_id" in results[0]

    def test_no_fallback_on_successful_bigquery_operation(self, bigquery_vector_index):
        """Must NOT use fallback when BigQuery operations succeed."""
        query_vector = [0.1] * 3072

        # Mock successful BigQuery response
        mock_results = [
            {"chunk_id": "bq_chunk_1", "distance": 0.1, "source": "bigquery"},
            {"chunk_id": "bq_chunk_2", "distance": 0.2, "source": "bigquery"},
        ]

        with patch.object(bigquery_vector_index, "_bigquery_client") as mock_client:
            mock_query_job = Mock()
            mock_query_job.__iter__ = Mock(return_value=iter(mock_results))
            mock_client.query.return_value = mock_query_job

            results = bigquery_vector_index.similarity_search(query_vector, top_k=5)

            # Verify BigQuery was used, not fallback
            if results:
                assert all(r.get("source") == "bigquery" for r in results)
                assert not any("fallback_reason" in r for r in results)


class TestFallbackErrorHandling:
    """Test error handling in fallback scenarios."""

    @pytest.fixture
    def bigquery_vector_index(self):
        """This will fail until BigQueryVectorIndex is implemented."""
        from src.janapada_memory.bigquery_vector_index import BigQueryVectorIndex

        return BigQueryVectorIndex()

    def test_fallback_handles_local_index_failure(self, bigquery_vector_index):
        """Must handle gracefully when both BigQuery AND local fallback fail."""
        query_vector = [0.1] * 3072

        # Simulate both BigQuery and local index failures
        with patch.object(bigquery_vector_index, "_bigquery_client") as mock_bq_client:
            with patch.object(
                bigquery_vector_index, "_local_index"
            ) as mock_local_index:
                mock_bq_client.query.side_effect = NotFound("BigQuery failed")
                mock_local_index.similarity_search.side_effect = Exception(
                    "Local index failed"
                )

                # Should handle gracefully - either return empty list or raise informative exception
                try:
                    results = bigquery_vector_index.similarity_search(
                        query_vector, top_k=5
                    )
                    assert isinstance(results, list)  # Empty list is acceptable
                except Exception as e:
                    # If exception is raised, it should be informative
                    assert "fallback" in str(e).lower() or "local" in str(e).lower()

    def test_fallback_respects_timeout_limits(self, bigquery_vector_index):
        """Fallback operations must respect timeout constraints."""
        import time

        query_vector = [0.1] * 3072

        with patch.object(bigquery_vector_index, "_bigquery_client") as mock_client:
            mock_client.query.side_effect = NotFound("Triggering fallback")

            start_time = time.time()
            results = bigquery_vector_index.similarity_search(query_vector, top_k=5)
            elapsed = time.time() - start_time

            # Even with fallback, should complete within reasonable time
            assert elapsed < 1.0, f"Fallback took {elapsed:.2f}s, too slow"
