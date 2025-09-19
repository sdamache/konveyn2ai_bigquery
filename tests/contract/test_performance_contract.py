"""
Performance contract tests for BigQuery Memory Adapter.

This test suite verifies that the BigQueryVectorIndex implementation
meets performance requirements for similarity search operations.

CRITICAL: These tests MUST FAIL initially (TDD RED phase) before implementation.
Target: similarity_search completes within 500ms timeout.
"""

import pytest
import time
from unittest.mock import Mock, patch


class TestBigQueryPerformanceContract:
    """Contract tests for BigQuery performance requirements."""

    @pytest.fixture
    def bigquery_vector_index(self):
        """This will fail until BigQueryVectorIndex is implemented."""
        from src.janapada_memory.bigquery_vector_index import BigQueryVectorIndex

        return BigQueryVectorIndex()

    def test_similarity_search_500ms_timeout_requirement(self, bigquery_vector_index):
        """similarity_search must complete within 500ms timeout (primary requirement)."""
        query_vector = [0.1] * 3072  # Gemini embedding dimension

        start_time = time.time()
        results = bigquery_vector_index.similarity_search(query_vector, top_k=10)
        elapsed = time.time() - start_time

        # Primary performance requirement from specification
        assert (
            elapsed < 0.5
        ), f"similarity_search took {elapsed:.3f}s, requirement: <500ms"
        assert isinstance(results, list)

    def test_batch_similarity_search_performance(self, bigquery_vector_index):
        """Batch operations should maintain acceptable performance."""
        query_vectors = [[0.1] * 3072, [0.2] * 3072, [0.3] * 3072]

        start_time = time.time()

        for query_vector in query_vectors:
            results = bigquery_vector_index.similarity_search(query_vector, top_k=5)
            assert isinstance(results, list)

        elapsed = time.time() - start_time

        # Batch operations should complete within reasonable time
        assert (
            elapsed < 2.0
        ), f"Batch search took {elapsed:.3f}s, too slow for 3 queries"

    def test_large_top_k_performance(self, bigquery_vector_index):
        """Large top_k values should not cause significant performance degradation."""
        query_vector = [0.1] * 3072

        # Test with maximum reasonable top_k
        start_time = time.time()
        results = bigquery_vector_index.similarity_search(query_vector, top_k=100)
        elapsed = time.time() - start_time

        # Should still meet reasonable performance even with large result sets
        assert elapsed < 1.0, f"Large top_k search took {elapsed:.3f}s, too slow"
        assert isinstance(results, list)
        assert len(results) <= 100

    def test_concurrent_search_performance(self, bigquery_vector_index):
        """Concurrent similarity searches should maintain performance."""
        import threading

        query_vector = [0.1] * 3072
        results_container = []
        errors_container = []

        def search_worker():
            try:
                start_time = time.time()
                results = bigquery_vector_index.similarity_search(query_vector, top_k=5)
                elapsed = time.time() - start_time
                results_container.append((results, elapsed))
            except Exception as e:
                errors_container.append(e)

        # Run 3 concurrent searches
        threads = []
        overall_start = time.time()

        for _ in range(3):
            thread = threading.Thread(target=search_worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        overall_elapsed = time.time() - overall_start

        # Verify no errors occurred
        assert not errors_container, f"Concurrent search errors: {errors_container}"

        # Verify all searches completed
        assert len(results_container) == 3

        # Each individual search should meet timeout requirement
        for results, elapsed in results_container:
            assert (
                elapsed < 0.5
            ), f"Concurrent search took {elapsed:.3f}s, exceeds 500ms limit"
            assert isinstance(results, list)

        # Overall concurrent execution should be reasonable
        assert (
            overall_elapsed < 2.0
        ), f"Concurrent execution took {overall_elapsed:.3f}s, too slow"

    def test_cold_start_performance(self, bigquery_vector_index):
        """First search (cold start) should meet performance requirements."""
        query_vector = [0.1] * 3072

        # Simulate cold start by ensuring no prior operations
        start_time = time.time()
        results = bigquery_vector_index.similarity_search(query_vector, top_k=5)
        elapsed = time.time() - start_time

        # Cold start should still meet performance requirement
        assert elapsed < 1.0, f"Cold start search took {elapsed:.3f}s, too slow"
        assert isinstance(results, list)

    def test_fallback_performance_requirement(self, bigquery_vector_index):
        """Fallback to local search should also meet performance requirements."""
        query_vector = [0.1] * 3072

        # Force fallback by mocking BigQuery failure
        with patch.object(
            bigquery_vector_index.bigquery_adapter, "search_similar_vectors"
        ) as mock_search:
            from google.cloud.exceptions import NotFound

            mock_search.side_effect = NotFound("Force fallback")

            start_time = time.time()
            results = bigquery_vector_index.similarity_search(query_vector, k=10)
            elapsed = time.time() - start_time

            # Fallback should still meet performance requirement
            assert (
                elapsed < 0.5
            ), f"Fallback search took {elapsed:.3f}s, exceeds 500ms limit"
            assert isinstance(results, list)

    @pytest.mark.parametrize("top_k", [1, 5, 10, 25, 50])
    def test_performance_scaling_with_top_k(self, bigquery_vector_index, top_k):
        """Performance should scale reasonably with different k values."""
        query_vector = [0.1] * 3072

        start_time = time.time()
        results = bigquery_vector_index.similarity_search(query_vector, k=top_k)
        elapsed = time.time() - start_time

        # Base requirement: all top_k values should meet 500ms requirement
        assert (
            elapsed < 0.5
        ), f"top_k={top_k} search took {elapsed:.3f}s, exceeds 500ms limit"
        assert isinstance(results, list)
        assert len(results) <= top_k

    def test_memory_usage_during_search(self, bigquery_vector_index):
        """Memory usage should remain reasonable during similarity search."""
        import psutil
        import os

        query_vector = [0.1] * 3072
        process = psutil.Process(os.getpid())

        # Get baseline memory usage
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform multiple searches
        for _ in range(5):
            results = bigquery_vector_index.similarity_search(query_vector, top_k=10)
            assert isinstance(results, list)

        # Check memory usage after operations
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - baseline_memory

        # Memory increase should be reasonable (less than 100MB for test operations)
        assert (
            memory_increase < 100
        ), f"Memory usage increased by {memory_increase:.1f}MB, too high"


class TestPerformanceBaseline:
    """Establish performance baselines for monitoring."""

    @pytest.fixture
    def bigquery_vector_index(self):
        """This will fail until BigQueryVectorIndex is implemented."""
        from src.janapada_memory.bigquery_vector_index import BigQueryVectorIndex

        return BigQueryVectorIndex()

    def test_establish_bigquery_performance_baseline(self, bigquery_vector_index):
        """Establish baseline performance metrics for BigQuery operations."""
        query_vector = [0.1] * 3072
        measurements = []

        # Take multiple measurements for statistical validity
        for i in range(5):
            start_time = time.time()
            results = bigquery_vector_index.similarity_search(query_vector, top_k=10)
            elapsed = time.time() - start_time
            measurements.append(elapsed)
            assert isinstance(results, list)

        # Calculate performance statistics
        avg_time = sum(measurements) / len(measurements)
        max_time = max(measurements)
        min_time = min(measurements)

        # Log baseline metrics for monitoring
        print("\nPerformance Baseline Metrics:")
        print(f"Average: {avg_time:.3f}s")
        print(f"Maximum: {max_time:.3f}s")
        print(f"Minimum: {min_time:.3f}s")

        # All measurements should meet requirement
        assert max_time < 0.5, f"Maximum time {max_time:.3f}s exceeds 500ms requirement"
        assert (
            avg_time < 0.3
        ), f"Average time {avg_time:.3f}s too high for sustained performance"

    def test_establish_fallback_performance_baseline(self, bigquery_vector_index):
        """Establish baseline performance metrics for fallback operations."""
        query_vector = [0.1] * 3072
        measurements = []

        # Force fallback mode
        with patch.object(bigquery_vector_index, "_bigquery_client") as mock_client:
            from google.cloud.exceptions import NotFound

            mock_client.query.side_effect = NotFound("Baseline test fallback")

            # Take multiple measurements
            for i in range(5):
                start_time = time.time()
                results = bigquery_vector_index.similarity_search(
                    query_vector, top_k=10
                )
                elapsed = time.time() - start_time
                measurements.append(elapsed)
                assert isinstance(results, list)

        # Calculate fallback performance statistics
        avg_time = sum(measurements) / len(measurements)
        max_time = max(measurements)

        print("\nFallback Performance Baseline Metrics:")
        print(f"Average: {avg_time:.3f}s")
        print(f"Maximum: {max_time:.3f}s")

        # Fallback should also meet performance requirements
        assert (
            max_time < 0.5
        ), f"Fallback maximum time {max_time:.3f}s exceeds 500ms requirement"
