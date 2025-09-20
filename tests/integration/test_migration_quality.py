"""
Integration test: Migration quality verification

This test validates migration quality and data integrity:
Vertex AI → BigQuery migration with dimension reduction and validation.

IMPORTANT: This test MUST FAIL initially (TDD Red phase)
Implementation will be done in T029-T037.
"""

import time

import numpy as np
import pytest


class TestMigrationQuality:
    """Integration tests for migration quality verification."""

    @pytest.fixture(scope="class")
    def legacy_vector_store(self):
        """Legacy Vertex AI vector store - will fail until implementation exists."""
        # This import will fail until implementation is done
        from src.janapada_memory.vertex_vector_store import VertexAIVectorStore

        return VertexAIVectorStore()

    @pytest.fixture(scope="class")
    def bigquery_client(self):
        """BigQuery client - will fail until implementation exists."""
        # This import will fail until implementation is done
        from src.janapada_memory.bigquery_vector_store import BigQueryVectorStore

        return BigQueryVectorStore()

    @pytest.fixture(scope="class")
    def migration_manager(self):
        """Migration manager - will fail until implementation exists."""
        # This import will fail until implementation is done
        from src.janapada_memory.migration_manager import MigrationManager

        return MigrationManager()

    @pytest.fixture(scope="class")
    def sample_legacy_data(self):
        """Sample legacy data with 768-dimensional embeddings."""
        return [
            {
                "chunk_id": "legacy_func_001",
                "text_content": "def calculate_similarity(vector_a, vector_b): return cosine_similarity(vector_a, vector_b)",
                "source": "src/legacy/similarity.py",
                "artifact_type": "code",
                "kind": "function",
                "record_name": "calculate_similarity",
                "embedding": np.random.rand(768).tolist(),  # Legacy 768 dimensions
                "metadata": {
                    "author": "legacy_user",
                    "created_at": "2023-01-01T00:00:00Z",
                    "version": "1.0",
                },
            },
            {
                "chunk_id": "legacy_class_001",
                "text_content": "class VectorProcessor: def __init__(self): self.vectors = []",
                "source": "src/legacy/processor.py",
                "artifact_type": "code",
                "kind": "class",
                "record_name": "VectorProcessor",
                "embedding": np.random.rand(768).tolist(),
                "metadata": {
                    "author": "legacy_user",
                    "created_at": "2023-02-01T00:00:00Z",
                    "version": "1.1",
                },
            },
            {
                "chunk_id": "legacy_doc_001",
                "text_content": "Vector similarity computation using cosine distance for semantic search",
                "source": "docs/legacy/similarity.md",
                "artifact_type": "documentation",
                "kind": "api_doc",
                "embedding": np.random.rand(768).tolist(),
                "metadata": {
                    "author": "docs_team",
                    "created_at": "2023-03-01T00:00:00Z",
                    "version": "2.0",
                },
            },
        ]

    def test_full_migration_workflow(
        self,
        migration_manager,
        legacy_vector_store,
        bigquery_client,
        sample_legacy_data,
    ):
        """Test complete migration workflow with quality verification."""

        # Step 1: Setup legacy data in Vertex AI
        print("Step 1: Setting up legacy data in Vertex AI...")
        for data in sample_legacy_data:
            legacy_vector_store.insert_embedding(
                chunk_data=data, embedding=data["embedding"]
            )

        # Verify legacy data exists
        legacy_count = legacy_vector_store.count_embeddings()
        assert legacy_count == len(sample_legacy_data)

        # Step 2: Initialize migration
        print("Step 2: Initializing migration...")
        migration_result = migration_manager.start_migration(
            source_type="vertex_ai",
            target_type="bigquery",
            migration_options={
                "batch_size": 100,
                "dimension_reduction": True,
                "target_dimensions": 768,
                "preserve_metadata": True,
                "validate_similarity": True,
            },
        )

        assert migration_result is not None
        assert migration_result["status"] == "STARTED"
        migration_id = migration_result["migration_id"]

        # Step 3: Monitor migration progress
        print("Step 3: Monitoring migration progress...")
        max_wait_time = 300  # 5 minutes
        wait_time = 0

        while wait_time < max_wait_time:
            status = migration_manager.get_migration_status(migration_id)

            if status["status"] == "COMPLETED":
                break
            elif status["status"] == "FAILED":
                pytest.fail(
                    f"Migration failed: {status.get('error_message', 'Unknown error')}"
                )

            time.sleep(10)
            wait_time += 10

        # Final status check
        final_status = migration_manager.get_migration_status(migration_id)
        assert final_status["status"] == "COMPLETED"

        # Step 4: Verify data count and integrity
        print("Step 4: Verifying data count and integrity...")
        bigquery_count = bigquery_client.count_embeddings()
        assert bigquery_count == len(sample_legacy_data)

        # Verify all chunks were migrated
        for data in sample_legacy_data:
            migrated_chunk = bigquery_client.get_embedding_by_id(data["chunk_id"])
            assert migrated_chunk is not None
            assert migrated_chunk["chunk_id"] == data["chunk_id"]
            assert migrated_chunk["text_content"] == data["text_content"]
            assert migrated_chunk["source"] == data["source"]
            assert migrated_chunk["artifact_type"] == data["artifact_type"]

    def test_dimension_reduction_quality(self, migration_manager, sample_legacy_data):
        """Test dimension reduction maintains semantic quality."""

        print("Testing PCA dimension reduction quality...")

        # Get dimension reduction results
        reduction_results = migration_manager.test_dimension_reduction(
            embeddings=[data["embedding"] for data in sample_legacy_data],
            target_dimensions=768,
            method="PCA",
        )

        assert reduction_results is not None
        assert "reduced_embeddings" in reduction_results
        assert "quality_metrics" in reduction_results

        # Verify dimensions are correct
        reduced_embeddings = reduction_results["reduced_embeddings"]
        for embedding in reduced_embeddings:
            assert len(embedding) == 768

        # Verify quality metrics
        quality_metrics = reduction_results["quality_metrics"]
        assert "variance_preserved" in quality_metrics
        assert "similarity_preservation" in quality_metrics

        # Quality thresholds
        assert (
            quality_metrics["variance_preserved"] >= 0.85
        )  # Should preserve 85%+ variance
        assert (
            quality_metrics["similarity_preservation"] >= 0.90
        )  # Should preserve 90%+ similarity

    def test_similarity_preservation_after_migration(
        self, legacy_vector_store, bigquery_client, sample_legacy_data
    ):
        """Test similarity relationships are preserved after migration."""

        print("Testing similarity preservation...")

        # Get similarity in legacy system
        query_text = "vector similarity calculation"
        legacy_results = legacy_vector_store.search_similar_text(
            query_text=query_text, limit=3, similarity_threshold=0.1
        )

        # Get similarity in new system
        bigquery_results = bigquery_client.search_similar_text(
            query_text=query_text, limit=3, similarity_threshold=0.1
        )

        # Verify same top results (order may vary slightly due to dimension reduction)
        legacy_chunk_ids = {result["chunk_id"] for result in legacy_results}
        bigquery_chunk_ids = {result["chunk_id"] for result in bigquery_results}

        # At least 80% of top results should match
        intersection = legacy_chunk_ids.intersection(bigquery_chunk_ids)
        overlap_ratio = len(intersection) / max(
            len(legacy_chunk_ids), len(bigquery_chunk_ids)
        )
        assert overlap_ratio >= 0.8

    def test_metadata_preservation(self, bigquery_client, sample_legacy_data):
        """Test metadata is properly preserved during migration."""

        print("Testing metadata preservation...")

        for original_data in sample_legacy_data:
            migrated_chunk = bigquery_client.get_embedding_by_id(
                original_data["chunk_id"]
            )

            # Core fields should match exactly
            assert migrated_chunk["chunk_id"] == original_data["chunk_id"]
            assert migrated_chunk["text_content"] == original_data["text_content"]
            assert migrated_chunk["source"] == original_data["source"]
            assert migrated_chunk["artifact_type"] == original_data["artifact_type"]

            # Optional fields should be preserved if they exist
            if "kind" in original_data:
                assert migrated_chunk["kind"] == original_data["kind"]
            if "record_name" in original_data:
                assert migrated_chunk["record_name"] == original_data["record_name"]

            # Custom metadata should be preserved
            if "metadata" in original_data:
                assert "metadata" in migrated_chunk
                for key, value in original_data["metadata"].items():
                    assert key in migrated_chunk["metadata"]
                    assert migrated_chunk["metadata"][key] == value

    def test_migration_performance_metrics(self, migration_manager, migration_id=None):
        """Test migration performance meets requirements."""

        if migration_id is None:
            # Use a test migration
            migration_result = migration_manager.start_migration(
                source_type="vertex_ai",
                target_type="bigquery",
                migration_options={"batch_size": 50},
            )
            migration_id = migration_result["migration_id"]

        print("Testing migration performance...")

        # Get performance metrics
        performance = migration_manager.get_migration_performance(migration_id)
        assert performance is not None

        # Check throughput (should process at least 10 embeddings/second)
        assert performance["embeddings_per_second"] >= 10

        # Check memory usage (should not exceed reasonable limits)
        assert performance["peak_memory_mb"] <= 1024  # 1GB limit

        # Check error rate (should be less than 1%)
        assert performance["error_rate"] <= 0.01

    def test_migration_rollback_capability(self, migration_manager):
        """Test migration can be rolled back if issues occur."""

        print("Testing migration rollback...")

        # Start a test migration
        migration_result = migration_manager.start_migration(
            source_type="vertex_ai",
            target_type="bigquery",
            migration_options={"enable_rollback": True},
        )

        migration_id = migration_result["migration_id"]

        # Wait for migration to start processing
        time.sleep(5)

        # Initiate rollback
        rollback_result = migration_manager.rollback_migration(migration_id)
        assert rollback_result is not None
        assert rollback_result["status"] == "ROLLBACK_INITIATED"

        # Wait for rollback to complete
        max_wait_time = 60  # 1 minute
        wait_time = 0

        while wait_time < max_wait_time:
            status = migration_manager.get_migration_status(migration_id)
            if status["status"] == "ROLLED_BACK":
                break
            time.sleep(5)
            wait_time += 5

        final_status = migration_manager.get_migration_status(migration_id)
        assert final_status["status"] == "ROLLED_BACK"

    def test_migration_validation_reports(self, migration_manager, migration_id=None):
        """Test migration validation reports are generated."""

        if migration_id is None:
            # Use completed migration from previous tests
            migrations = migration_manager.list_migrations(status="COMPLETED")
            assert len(migrations) > 0
            migration_id = migrations[0]["migration_id"]

        print("Testing migration validation reports...")

        # Get validation report
        validation_report = migration_manager.get_validation_report(migration_id)
        assert validation_report is not None

        # Check report sections
        required_sections = [
            "data_integrity",
            "similarity_preservation",
            "performance_metrics",
            "quality_assessment",
        ]

        for section in required_sections:
            assert section in validation_report

        # Check data integrity section
        data_integrity = validation_report["data_integrity"]
        assert "total_records_migrated" in data_integrity
        assert "successful_migrations" in data_integrity
        assert "failed_migrations" in data_integrity
        assert (
            data_integrity["successful_migrations"]
            >= data_integrity["failed_migrations"]
        )

        # Check similarity preservation section
        similarity_preservation = validation_report["similarity_preservation"]
        assert "average_similarity_score" in similarity_preservation
        assert "similarity_correlation" in similarity_preservation
        assert similarity_preservation["similarity_correlation"] >= 0.85

    def test_migration_cleanup_and_archival(self, migration_manager):
        """Test migration cleanup and data archival processes."""

        print("Testing migration cleanup and archival...")

        # List all completed migrations
        completed_migrations = migration_manager.list_migrations(status="COMPLETED")

        if len(completed_migrations) > 0:
            migration_id = completed_migrations[0]["migration_id"]

            # Test archival
            archive_result = migration_manager.archive_migration_data(
                migration_id=migration_id,
                keep_validation_report=True,
                archive_location="gs://konveyn2ai-backups/migrations/",
            )

            assert archive_result is not None
            assert archive_result["status"] == "ARCHIVED"
            assert "archive_location" in archive_result

            # Test cleanup
            cleanup_result = migration_manager.cleanup_migration_workspace(migration_id)
            assert cleanup_result is not None
            assert cleanup_result["status"] == "CLEANED"

    def test_migration_error_handling(self, migration_manager):
        """Test migration error handling and recovery."""

        print("Testing migration error handling...")

        # Start migration with invalid configuration to trigger errors
        invalid_migration = migration_manager.start_migration(
            source_type="vertex_ai",
            target_type="bigquery",
            migration_options={
                "batch_size": -1,  # Invalid batch size
                "target_dimensions": 999999,  # Invalid dimensions
            },
        )

        # Should handle errors gracefully
        assert invalid_migration["status"] in ["FAILED", "VALIDATION_ERROR"]
        assert "error_message" in invalid_migration
        assert len(invalid_migration["error_message"]) > 0

    def test_migration_concurrent_operations(self, migration_manager, bigquery_client):
        """Test system handles concurrent operations during migration."""

        print("Testing concurrent operations during migration...")

        # Start a migration
        migration_result = migration_manager.start_migration(
            source_type="vertex_ai",
            target_type="bigquery",
            migration_options={"batch_size": 10},
        )

        migration_id = migration_result["migration_id"]

        # Try concurrent operations
        test_chunk = {
            "chunk_id": "concurrent_test_001",
            "text_content": "Test chunk during migration",
            "source": "test/concurrent.py",
            "artifact_type": "test",
            "embedding": np.random.rand(768).tolist(),
        }

        # This should work (new system should handle concurrent operations)
        try:
            result = bigquery_client.insert_embedding(
                chunk_data=test_chunk, embedding=test_chunk["embedding"]
            )
            assert result is not None
        except Exception as e:
            # If it fails, should fail gracefully with clear error
            assert "migration in progress" in str(e).lower()

    def test_migration_quality_thresholds(self, migration_manager):
        """Test migration meets quality thresholds."""

        print("Testing migration quality thresholds...")

        # Get completed migrations
        completed_migrations = migration_manager.list_migrations(status="COMPLETED")

        for migration in completed_migrations:
            migration_id = migration["migration_id"]
            quality_report = migration_manager.get_quality_report(migration_id)

            # Quality thresholds
            assert quality_report["data_integrity_score"] >= 0.99  # 99%+ integrity
            assert (
                quality_report["similarity_preservation_score"] >= 0.90
            )  # 90%+ similarity
            assert quality_report["performance_score"] >= 0.85  # 85%+ performance
            assert quality_report["overall_quality_score"] >= 0.90  # 90%+ overall

    def test_migration_audit_trail(self, migration_manager):
        """Test migration audit trail and logging."""

        print("Testing migration audit trail...")

        # Get audit trail for a migration
        completed_migrations = migration_manager.list_migrations(status="COMPLETED")

        if len(completed_migrations) > 0:
            migration_id = completed_migrations[0]["migration_id"]
            audit_trail = migration_manager.get_audit_trail(migration_id)

            assert audit_trail is not None
            assert "events" in audit_trail
            assert len(audit_trail["events"]) > 0

            # Check required audit events
            event_types = [event["type"] for event in audit_trail["events"]]
            required_events = [
                "MIGRATION_STARTED",
                "DATA_VALIDATION",
                "MIGRATION_COMPLETED",
            ]

            for event_type in required_events:
                assert event_type in event_types

            # Each event should have timestamp and details
            for event in audit_trail["events"]:
                assert "timestamp" in event
                assert "type" in event
                assert "details" in event
