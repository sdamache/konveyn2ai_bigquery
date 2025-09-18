"""
Contract tests for VectorIndex interface preservation.
These tests MUST FAIL initially to enforce RED-GREEN-REFACTOR cycle.
"""

import pytest
from typing import List, Tuple
from abc import ABC, abstractmethod


class VectorIndex(ABC):
    """Abstract interface that MUST be preserved exactly."""

    @abstractmethod
    def similarity_search(self, query_vector: List[float], top_k: int) -> List[dict]:
        """Return top_k most similar vectors ordered by ascending distance."""
        pass

    @abstractmethod
    def add_vectors(self, vectors: List[Tuple[str, List[float]]]) -> None:
        """Add vectors with chunk IDs to the index."""
        pass

    @abstractmethod
    def remove_vector(self, chunk_id: str) -> bool:
        """Remove vector by chunk ID. Return True if removed, False if not found."""
        pass


class TestVectorIndexContract:
    """Contract tests that BigQueryVectorIndex MUST satisfy."""

    @pytest.fixture
    def vector_index(self):
        """This will fail until BigQueryVectorIndex is implemented."""
        from src.janapada_memory.bigquery_vector_index import BigQueryVectorIndex

        return BigQueryVectorIndex()

    def test_similarity_search_returns_correct_type(self, vector_index):
        """similarity_search must return List[dict] with required fields."""
        query_vector = [0.1] * 3072  # Gemini embedding dimension
        results = vector_index.similarity_search(query_vector, top_k=5)

        assert isinstance(results, list)
        assert len(results) <= 5

        for result in results:
            assert isinstance(result, dict)
            assert "chunk_id" in result
            assert "distance" in result
            assert isinstance(result["chunk_id"], str)
            assert isinstance(result["distance"], (int, float))

    def test_similarity_search_orders_by_distance(self, vector_index):
        """Results must be ordered by ascending distance."""
        query_vector = [0.1] * 3072
        results = vector_index.similarity_search(query_vector, top_k=10)

        if len(results) > 1:
            distances = [r["distance"] for r in results]
            assert distances == sorted(
                distances
            ), "Results must be ordered by ascending distance"

    def test_similarity_search_respects_top_k(self, vector_index):
        """Must not return more than top_k results."""
        query_vector = [0.1] * 3072

        for k in [1, 5, 10, 50]:
            results = vector_index.similarity_search(query_vector, top_k=k)
            assert len(results) <= k, f"Returned {len(results)} results for top_k={k}"

    def test_similarity_search_handles_empty_index(self, vector_index):
        """Must handle empty index gracefully."""
        query_vector = [0.1] * 3072
        results = vector_index.similarity_search(query_vector, top_k=5)
        assert isinstance(results, list)  # May be empty, but must be list

    def test_similarity_search_validates_query_vector(self, vector_index):
        """Must validate query vector dimensions."""
        with pytest.raises((ValueError, TypeError)):
            vector_index.similarity_search([], top_k=5)  # Empty vector

        with pytest.raises((ValueError, TypeError)):
            vector_index.similarity_search([0.1] * 10, top_k=5)  # Wrong dimensions

    def test_add_vectors_interface(self, vector_index):
        """add_vectors must accept List[Tuple[str, List[float]]]."""
        vectors = [("test_chunk_1", [0.1] * 3072), ("test_chunk_2", [0.2] * 3072)]

        # Should not raise exception
        vector_index.add_vectors(vectors)

    def test_remove_vector_interface(self, vector_index):
        """remove_vector must return bool."""
        result = vector_index.remove_vector("nonexistent_chunk")
        assert isinstance(result, bool)

    def test_bigquery_fallback_transparency(self, vector_index):
        """BigQuery failures must be transparent to interface contract."""
        # This test will verify fallback behavior maintains interface
        query_vector = [0.1] * 3072

        # Force BigQuery error (mocked in actual implementation)
        results = vector_index.similarity_search(query_vector, top_k=5)

        # Interface contract must be preserved regardless of backend
        assert isinstance(results, list)
        for result in results:
            assert "chunk_id" in result
            assert "distance" in result


class TestBigQuerySpecificContract:
    """Tests specific to BigQuery implementation details."""

    @pytest.fixture
    def bigquery_index(self):
        """This will fail until BigQueryVectorIndex is implemented."""
        from src.janapada_memory.bigquery_vector_index import BigQueryVectorIndex

        return BigQueryVectorIndex()

    def test_bigquery_configuration_loading(self, bigquery_index):
        """Must load configuration from environment variables."""
        # This will fail until configuration is implemented
        assert hasattr(bigquery_index, "config")
        assert bigquery_index.config.project_id is not None
        assert bigquery_index.config.dataset_id is not None

    def test_fallback_activation_logging(self, bigquery_index, caplog):
        """Must log fallback activation with structured data."""
        query_vector = [0.1] * 3072

        # Force BigQuery error to trigger fallback
        # (Actual error injection will be in implementation)
        results = bigquery_index.similarity_search(query_vector, top_k=5)

        # Must log fallback activation
        assert "fallback" in caplog.text.lower()
        assert "bigquery" in caplog.text.lower()

    def test_health_check_method(self, bigquery_index):
        """Must provide BigQuery health checking."""
        assert hasattr(bigquery_index, "check_bigquery_health")
        health = bigquery_index.check_bigquery_health()
        assert isinstance(health, bool)
