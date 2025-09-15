"""
Integration test: End-to-end vector workflow

This test validates the complete vector workflow:
Insert vectors → Search similar → Validate results with actual BigQuery.

IMPORTANT: This test MUST FAIL initially (TDD Red phase)
Implementation will be done in T015-T024.
"""

import time

import numpy as np
import pytest


class TestVectorWorkflow:
    """Integration tests for end-to-end vector operations."""

    @pytest.fixture(scope="class")
    def bigquery_client(self):
        """BigQuery client - will fail until implementation exists."""
        # This import will fail until implementation is done
        from src.janapada_memory.bigquery_vector_store import BigQueryVectorStore

        return BigQueryVectorStore()

    @pytest.fixture(scope="class")
    def schema_manager(self):
        """Schema manager - will fail until implementation exists."""
        # This import will fail until implementation is done
        from src.janapada_memory.schema_manager import SchemaManager

        return SchemaManager()

    @pytest.fixture(scope="class")
    def sample_embeddings(self):
        """Sample embeddings for testing."""
        return [
            {
                "chunk_id": "func_cosine_similarity_001",
                "text_content": "def cosine_similarity(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))",
                "source": "src/utils/similarity.py",
                "artifact_type": "code",
                "kind": "function",
                "record_name": "cosine_similarity",
                "embedding": np.random.rand(768).tolist(),
            },
            {
                "chunk_id": "func_euclidean_distance_001",
                "text_content": "def euclidean_distance(a, b): return np.sqrt(np.sum((a - b) ** 2))",
                "source": "src/utils/distance.py",
                "artifact_type": "code",
                "kind": "function",
                "record_name": "euclidean_distance",
                "embedding": np.random.rand(768).tolist(),
            },
            {
                "chunk_id": "api_similarity_endpoint_001",
                "text_content": "POST /api/v1/similarity - Calculate similarity between two vectors",
                "source": "api/endpoints.md",
                "artifact_type": "documentation",
                "kind": "api_doc",
                "embedding": np.random.rand(768).tolist(),
            },
            {
                "chunk_id": "class_vector_store_001",
                "text_content": "class VectorStore: def __init__(self): self.vectors = {}",
                "source": "src/models/vector_store.py",
                "artifact_type": "code",
                "kind": "class",
                "record_name": "VectorStore",
                "embedding": np.random.rand(768).tolist(),
            },
        ]

    def test_end_to_end_vector_workflow(
        self, schema_manager, bigquery_client, sample_embeddings
    ):
        """Test complete end-to-end vector workflow."""

        # Step 1: Setup schema (tables and indexes)
        print("Step 1: Setting up BigQuery schema...")
        schema_result = schema_manager.create_tables()
        assert schema_result is not None
        assert len(schema_result.get("tables_created", [])) == 3

        # Create vector index
        index_result = schema_manager.create_indexes(
            [
                {
                    "name": "embedding_similarity_index",
                    "table": "source_embeddings",
                    "column": "embedding",
                    "index_type": "IVF",
                    "distance_type": "COSINE",
                }
            ]
        )
        assert index_result is not None

        # Wait for index to be ready (might take time)
        time.sleep(5)

        # Step 2: Insert sample embeddings
        print("Step 2: Inserting sample embeddings...")
        insertion_results = []
        for embedding_data in sample_embeddings:
            result = bigquery_client.insert_embedding(
                chunk_data=embedding_data, embedding=embedding_data["embedding"]
            )
            assert result is not None
            assert result["chunk_id"] == embedding_data["chunk_id"]
            insertion_results.append(result)

        # Verify all insertions succeeded
        assert len(insertion_results) == len(sample_embeddings)

        # Step 3: Search for similar vectors (text-based)
        print("Step 3: Performing text-based similarity search...")
        text_query = "function that calculates similarity between vectors"
        text_search_results = bigquery_client.search_similar_text(
            query_text=text_query,
            limit=5,
            similarity_threshold=0.1,  # Low threshold to get results
        )

        # Should return some results
        assert text_search_results is not None
        assert isinstance(text_search_results, list)

        # Results should be sorted by similarity score
        if len(text_search_results) > 1:
            for i in range(len(text_search_results) - 1):
                assert (
                    text_search_results[i]["similarity_score"]
                    >= text_search_results[i + 1]["similarity_score"]
                )

        # Step 4: Search for similar vectors (vector-based)
        print("Step 4: Performing vector-based similarity search...")
        query_vector = np.random.rand(768).tolist()
        vector_search_results = bigquery_client.search_similar_vectors(
            query_embedding=query_vector, limit=3, similarity_threshold=0.1
        )

        # Should return some results
        assert vector_search_results is not None
        assert isinstance(vector_search_results, list)

        # Step 5: Validate search results
        print("Step 5: Validating search results...")
        for result in text_search_results + vector_search_results:
            # Each result should have required fields
            required_fields = [
                "chunk_id",
                "similarity_score",
                "text_content",
                "source",
                "artifact_type",
            ]
            for field in required_fields:
                assert field in result

            # Similarity score should be valid
            assert 0.0 <= result["similarity_score"] <= 1.0

            # Chunk ID should exist in our sample data
            chunk_ids = [e["chunk_id"] for e in sample_embeddings]
            assert result["chunk_id"] in chunk_ids

        # Step 6: Test specific chunk retrieval
        print("Step 6: Testing specific chunk retrieval...")
        test_chunk_id = sample_embeddings[0]["chunk_id"]
        retrieved_chunk = bigquery_client.get_embedding_by_id(test_chunk_id)

        assert retrieved_chunk is not None
        assert retrieved_chunk["chunk_id"] == test_chunk_id
        assert len(retrieved_chunk["embedding"]) == 768

        print("✅ End-to-end vector workflow completed successfully!")

    def test_vector_workflow_with_filtering(self, bigquery_client, sample_embeddings):
        """Test vector workflow with artifact type filtering."""

        # Search only for code artifacts
        code_results = bigquery_client.search_similar_text(
            query_text="function definition",
            limit=5,
            similarity_threshold=0.1,
            artifact_types=["code"],
        )

        # All results should be code artifacts
        for result in code_results:
            assert result["artifact_type"] == "code"

        # Search only for documentation artifacts
        doc_results = bigquery_client.search_similar_text(
            query_text="API endpoint documentation",
            limit=5,
            similarity_threshold=0.1,
            artifact_types=["documentation"],
        )

        # All results should be documentation artifacts
        for result in doc_results:
            assert result["artifact_type"] == "documentation"

    def test_vector_workflow_performance(self, bigquery_client):
        """Test vector workflow performance meets requirements."""

        # Generate query vector
        query_vector = np.random.rand(768).tolist()

        # Measure search latency
        start_time = time.time()
        results = bigquery_client.search_similar_vectors(
            query_embedding=query_vector, limit=10, similarity_threshold=0.1
        )
        search_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Should meet performance target (<200ms)
        assert search_time < 200, f"Search took {search_time:.1f}ms, expected <200ms"

        print(f"Search completed in {search_time:.1f}ms")

    def test_vector_workflow_data_integrity(self, bigquery_client, sample_embeddings):
        """Test data integrity throughout the workflow."""

        # Insert a known embedding
        test_data = sample_embeddings[0]
        insertion_result = bigquery_client.insert_embedding(
            chunk_data=test_data, embedding=test_data["embedding"]
        )

        # Retrieve the same embedding
        retrieved_data = bigquery_client.get_embedding_by_id(test_data["chunk_id"])

        # Verify data integrity
        assert retrieved_data["chunk_id"] == test_data["chunk_id"]
        assert retrieved_data["text_content"] == test_data["text_content"]
        assert retrieved_data["source"] == test_data["source"]
        assert retrieved_data["artifact_type"] == test_data["artifact_type"]

        # Verify embedding integrity (should be identical)
        original_embedding = test_data["embedding"]
        retrieved_embedding = retrieved_data["embedding"]

        assert len(retrieved_embedding) == len(original_embedding)
        for i in range(len(original_embedding)):
            assert abs(retrieved_embedding[i] - original_embedding[i]) < 1e-10

    def test_vector_workflow_error_handling(self, bigquery_client):
        """Test error handling in vector workflow."""

        # Test duplicate insertion
        duplicate_data = {
            "chunk_id": "duplicate_test_001",
            "text_content": "This is a test chunk for duplicate testing",
            "source": "test/duplicate.py",
            "artifact_type": "code",
            "embedding": np.random.rand(768).tolist(),
        }

        # First insertion should succeed
        result1 = bigquery_client.insert_embedding(
            chunk_data=duplicate_data, embedding=duplicate_data["embedding"]
        )
        assert result1 is not None

        # Second insertion should fail gracefully
        with pytest.raises(Exception) as exc_info:
            bigquery_client.insert_embedding(
                chunk_data=duplicate_data, embedding=duplicate_data["embedding"]
            )
        assert "already exists" in str(exc_info.value).lower()

        # Test retrieval of non-existent chunk
        with pytest.raises(Exception) as exc_info:
            bigquery_client.get_embedding_by_id("non_existent_chunk_999")
        assert "not found" in str(exc_info.value).lower()

    def test_vector_workflow_batch_operations(self, bigquery_client, sample_embeddings):
        """Test batch operations in vector workflow."""

        # Test batch insertion
        batch_results = bigquery_client.batch_insert_embeddings(sample_embeddings[:2])
        assert len(batch_results) == 2

        # Test batch retrieval
        chunk_ids = [e["chunk_id"] for e in sample_embeddings[:2]]
        batch_retrieved = bigquery_client.batch_get_embeddings(chunk_ids)
        assert len(batch_retrieved) == 2

        # Verify batch data integrity
        for retrieved in batch_retrieved:
            original = next(
                e for e in sample_embeddings if e["chunk_id"] == retrieved["chunk_id"]
            )
            assert retrieved["text_content"] == original["text_content"]

    def test_vector_workflow_cleanup(self, bigquery_client, sample_embeddings):
        """Test cleanup operations in vector workflow."""

        # Insert test data
        test_chunk_id = "cleanup_test_001"
        cleanup_data = {
            "chunk_id": test_chunk_id,
            "text_content": "This chunk will be deleted",
            "source": "test/cleanup.py",
            "artifact_type": "test",
            "embedding": np.random.rand(768).tolist(),
        }

        # Insert
        bigquery_client.insert_embedding(
            chunk_data=cleanup_data, embedding=cleanup_data["embedding"]
        )

        # Verify exists
        retrieved = bigquery_client.get_embedding_by_id(test_chunk_id)
        assert retrieved is not None

        # Delete
        delete_result = bigquery_client.delete_embedding(test_chunk_id)
        assert delete_result is True

        # Verify deleted
        with pytest.raises(Exception) as exc_info:
            bigquery_client.get_embedding_by_id(test_chunk_id)
        assert "not found" in str(exc_info.value).lower()
