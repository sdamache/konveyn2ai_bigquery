"""
Unit tests for BigQuery writer functionality

T036: Comprehensive test coverage for BigQuery integration, schema handling, and data operations
Tests batch writing, error handling, schema validation, and context managers
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any

# Import test dependencies
try:
    from google.cloud import bigquery
    from google.cloud.exceptions import GoogleCloudError, NotFound, Conflict
    from google.api_core import exceptions
except ImportError:
    pytest.skip("Google Cloud BigQuery not available", allow_module_level=True)

# Import modules under test
from common.bq_writer import (
    BigQueryWriter,
    WriteResult,
    BatchConfig,
    SourceType,
    ErrorClass,
    ChunkMetadata,
    ParseError
)


class TestWriteResult:
    """Test WriteResult dataclass"""

    def test_write_result_creation(self):
        """Test WriteResult instantiation"""
        result = WriteResult(
            rows_written=100,
            errors_count=5,
            processing_time_ms=1500,
            run_id="01H8XY2K3MNBQJFGASDFGHJKLQ",
            table_name="source_metadata",
            success=True
        )

        assert result.rows_written == 100
        assert result.errors_count == 5
        assert result.processing_time_ms == 1500
        assert result.run_id == "01H8XY2K3MNBQJFGASDFGHJKLQ"
        assert result.table_name == "source_metadata"
        assert result.success is True
        assert result.error_message is None

    def test_write_result_with_error(self):
        """Test WriteResult with error message"""
        result = WriteResult(
            rows_written=0,
            errors_count=10,
            processing_time_ms=500,
            run_id="01H8XY2K3MNBQJFGASDFGHJKLQ",
            table_name="source_metadata",
            success=False,
            error_message="Connection timeout"
        )

        assert result.success is False
        assert result.error_message == "Connection timeout"


class TestBatchConfig:
    """Test BatchConfig dataclass"""

    def test_default_batch_config(self):
        """Test default BatchConfig values"""
        config = BatchConfig()

        assert config.batch_size == 1000
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0
        assert config.timeout_seconds == 300
        assert config.use_storage_write_api is True

    def test_custom_batch_config(self):
        """Test custom BatchConfig values"""
        config = BatchConfig(
            batch_size=500,
            max_retries=5,
            retry_delay_seconds=2.0,
            timeout_seconds=600,
            use_storage_write_api=False
        )

        assert config.batch_size == 500
        assert config.max_retries == 5
        assert config.retry_delay_seconds == 2.0
        assert config.timeout_seconds == 600
        assert config.use_storage_write_api is False


class TestBigQueryWriter:
    """Test BigQuery writer functionality"""

    @patch('common.bq_writer.bigquery.Client')
    def test_initialization_default(self, mock_client_class):
        """Test BigQueryWriter initialization with defaults"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        assert writer.project_id == "konveyn2ai"
        assert writer.dataset_id == "source_ingestion"
        assert writer.location == "US"
        assert isinstance(writer.batch_config, BatchConfig)
        assert writer.client == mock_client
        mock_client_class.assert_called_once_with(project="konveyn2ai")

    @patch('common.bq_writer.bigquery.Client')
    def test_initialization_custom(self, mock_client_class):
        """Test BigQueryWriter initialization with custom values"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        custom_config = BatchConfig(batch_size=500, max_retries=5)
        writer = BigQueryWriter(
            project_id="test-project",
            dataset_id="test_dataset",
            location="EU",
            batch_config=custom_config
        )

        assert writer.project_id == "test-project"
        assert writer.dataset_id == "test_dataset"
        assert writer.location == "EU"
        assert writer.batch_config.batch_size == 500
        assert writer.batch_config.max_retries == 5

    @patch.dict('os.environ', {'BQ_PROJECT': 'env-project', 'BQ_DATASET': 'env_dataset'})
    @patch('common.bq_writer.bigquery.Client')
    def test_initialization_from_env(self, mock_client_class):
        """Test BigQueryWriter initialization from environment variables"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        assert writer.project_id == "env-project"
        assert writer.dataset_id == "env_dataset"

    @patch('common.bq_writer.bigquery.Client')
    def test_setup_table_schemas(self, mock_client_class):
        """Test table schema setup"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        # Check that schemas are set up correctly
        assert "source_metadata" in writer.schemas
        assert "source_metadata_errors" in writer.schemas
        assert "ingestion_log" in writer.schemas

        # Check source_metadata schema fields
        source_metadata_schema = writer.schemas["source_metadata"]
        field_names = [field.name for field in source_metadata_schema]

        expected_fields = [
            "source_type", "artifact_id", "parent_id", "parent_type",
            "content_text", "content_tokens", "content_hash", "source_uri",
            "repo_ref", "collected_at", "source_metadata"
        ]

        for field in expected_fields:
            assert field in field_names

    @patch('common.bq_writer.bigquery.Client')
    def test_create_dataset_if_not_exists_new(self, mock_client_class):
        """Test dataset creation when dataset doesn't exist"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Simulate dataset not found
        mock_client.get_dataset.side_effect = NotFound("Dataset not found")
        mock_dataset = Mock()
        mock_client.create_dataset.return_value = mock_dataset

        writer = BigQueryWriter(project_id="test-project", dataset_id="test_dataset")
        result = writer.create_dataset_if_not_exists()

        assert result is True
        mock_client.get_dataset.assert_called_once()
        mock_client.create_dataset.assert_called_once()

    @patch('common.bq_writer.bigquery.Client')
    def test_create_dataset_if_not_exists_existing(self, mock_client_class):
        """Test dataset creation when dataset already exists"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Simulate dataset exists
        mock_dataset = Mock()
        mock_client.get_dataset.return_value = mock_dataset

        writer = BigQueryWriter(project_id="test-project", dataset_id="test_dataset")
        result = writer.create_dataset_if_not_exists()

        assert result is False
        mock_client.get_dataset.assert_called_once()
        mock_client.create_dataset.assert_not_called()

    @patch('common.bq_writer.bigquery.Client')
    def test_create_tables_if_not_exist(self, mock_client_class):
        """Test table creation"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Simulate tables don't exist
        mock_client.get_table.side_effect = NotFound("Table not found")
        mock_table = Mock()
        mock_client.create_table.return_value = mock_table

        writer = BigQueryWriter()
        result = writer.create_tables_if_not_exist()

        assert "source_metadata" in result
        assert "source_metadata_errors" in result
        assert "ingestion_log" in result
        assert all(result.values())  # All should be True (created)

        # Should call create_table for each schema
        assert mock_client.create_table.call_count == len(writer.schemas)

    @patch('common.bq_writer.bigquery.Client')
    def test_chunk_to_row_conversion(self, mock_client_class):
        """Test ChunkMetadata to row conversion"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        chunk = ChunkMetadata(
            source_type=SourceType.KUBERNETES,
            artifact_id="k8s://production/Deployment/web-app",
            parent_id=None,
            parent_type=None,
            content_text="apiVersion: v1\nkind: Deployment",
            content_tokens=50,
            content_hash="abc123def456",
            source_uri="file:///manifests/deployment.yaml",
            repo_ref="main",
            collected_at=datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc),
            source_metadata={"namespace": "production", "labels": {"app": "web"}}
        )

        row = writer._chunk_to_row(chunk)

        assert row["source_type"] == "kubernetes"
        assert row["artifact_id"] == "k8s://production/Deployment/web-app"
        assert row["content_text"] == "apiVersion: v1\nkind: Deployment"
        assert row["content_tokens"] == 50
        assert row["content_hash"] == "abc123def456"
        assert row["source_uri"] == "file:///manifests/deployment.yaml"
        assert row["repo_ref"] == "main"

        # Check collected_at is ISO format string
        assert isinstance(row["collected_at"], str)
        assert "2024-01-15T10:30:45" in row["collected_at"]

        # Check source_metadata is JSON string
        source_metadata = json.loads(row["source_metadata"])
        assert source_metadata["namespace"] == "production"
        assert source_metadata["labels"]["app"] == "web"

    @patch('common.bq_writer.bigquery.Client')
    def test_error_to_row_conversion(self, mock_client_class):
        """Test ParseError to row conversion"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        error = ParseError(
            source_type=SourceType.FASTAPI,
            source_uri="file:///src/api.py",
            error_class=ErrorClass.PARSING,
            error_msg="Syntax error in Python code",
            sample_text="def broken_func(",
            stack_trace="Traceback...",
            collected_at=datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        )

        row = writer._error_to_row(error)

        assert row["source_type"] == "fastapi"
        assert row["source_uri"] == "file:///src/api.py"
        assert row["error_class"] == "parsing"
        assert row["error_msg"] == "Syntax error in Python code"
        assert row["sample_text"] == "def broken_func("
        assert row["stack_trace"] == "Traceback..."

        # Check collected_at is ISO format string
        assert isinstance(row["collected_at"], str)
        assert "2024-01-15T10:30:45" in row["collected_at"]

    @patch('common.bq_writer.bigquery.Client')
    def test_run_info_to_row_conversion(self, mock_client_class):
        """Test run info to row conversion"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        run_info = {
            "run_id": "01H8XY2K3MNBQJFGASDFGHJKLQ",
            "source_type": "kubernetes",
            "started_at": datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            "completed_at": datetime(2024, 1, 15, 10, 35, 0, tzinfo=timezone.utc),
            "status": "completed",
            "files_processed": 25,
            "chunks_created": 150,
            "errors_encountered": 2,
            "processing_duration_ms": 300000,
            "config_used": {"batch_size": 1000},
            "error_summary": "2 minor parsing errors"
        }

        row = writer._run_info_to_row(run_info)

        assert row["run_id"] == "01H8XY2K3MNBQJFGASDFGHJKLQ"
        assert row["source_type"] == "kubernetes"
        assert row["status"] == "completed"
        assert row["files_processed"] == 25
        assert row["chunks_created"] == 150
        assert row["errors_encountered"] == 2
        assert row["processing_duration_ms"] == 300000

        # Check timestamps are ISO format strings
        assert isinstance(row["started_at"], str)
        assert isinstance(row["completed_at"], str)
        assert "2024-01-15T10:30:00" in row["started_at"]
        assert "2024-01-15T10:35:00" in row["completed_at"]

        # Check config_used is JSON string
        config = json.loads(row["config_used"])
        assert config["batch_size"] == 1000

    @patch('common.bq_writer.bigquery.Client')
    def test_batch_iterator(self, mock_client_class):
        """Test batch iterator functionality"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        items = list(range(25))  # 25 items
        batches = list(writer._batch_iterator(items, batch_size=10))

        assert len(batches) == 3  # 10, 10, 5
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5
        assert batches[0] == list(range(10))
        assert batches[1] == list(range(10, 20))
        assert batches[2] == list(range(20, 25))

    @patch('common.bq_writer.bigquery.Client')
    def test_clustering_fields(self, mock_client_class):
        """Test clustering fields configuration"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        # Test source_metadata clustering
        clustering = writer._get_clustering_fields("source_metadata")
        assert clustering == ["source_type", "content_hash"]

        # Test source_metadata_errors clustering
        clustering = writer._get_clustering_fields("source_metadata_errors")
        assert clustering == ["source_type", "error_class"]

        # Test ingestion_log clustering
        clustering = writer._get_clustering_fields("ingestion_log")
        assert clustering == ["source_type", "status"]

        # Test unknown table
        clustering = writer._get_clustering_fields("unknown_table")
        assert clustering is None

    @patch('common.bq_writer.bigquery.Client')
    def test_partitioning(self, mock_client_class):
        """Test table partitioning configuration"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        # Test source_metadata partitioning
        partitioning = writer._get_partitioning("source_metadata")
        assert partitioning is not None
        assert partitioning.field == "collected_at"
        assert partitioning.type_ == bigquery.TimePartitioningType.DAY

        # Test source_metadata_errors partitioning
        partitioning = writer._get_partitioning("source_metadata_errors")
        assert partitioning is not None
        assert partitioning.field == "collected_at"

        # Test ingestion_log partitioning
        partitioning = writer._get_partitioning("ingestion_log")
        assert partitioning is not None
        assert partitioning.field == "started_at"

        # Test unknown table
        partitioning = writer._get_partitioning("unknown_table")
        assert partitioning is None


class TestBigQueryWriterWriteOperations:
    """Test BigQuery writer write operations with mocking"""

    @patch('common.bq_writer.bigquery.Client')
    def test_write_chunks_success(self, mock_client_class):
        """Test successful chunk writing"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock successful write batch
        mock_write_result = WriteResult(
            rows_written=2,
            errors_count=0,
            processing_time_ms=500,
            run_id="test-run-id",
            table_name="source_metadata",
            success=True
        )

        writer = BigQueryWriter()
        with patch.object(writer, '_write_batch', return_value=mock_write_result):
            chunks = [
                ChunkMetadata(
                    source_type=SourceType.KUBERNETES,
                    artifact_id="k8s://test/Pod/test-pod",
                    content_text="apiVersion: v1",
                    content_hash="hash1",
                    source_uri="file:///test.yaml",
                    collected_at=datetime.now(timezone.utc)
                ),
                ChunkMetadata(
                    source_type=SourceType.FASTAPI,
                    artifact_id="py://api.py#1-10",
                    content_text="from fastapi import FastAPI",
                    content_hash="hash2",
                    source_uri="file:///api.py",
                    collected_at=datetime.now(timezone.utc)
                )
            ]

            result = writer.write_chunks(chunks, "test-run-id")

            assert result.success is True
            assert result.rows_written == 2
            assert result.errors_count == 0
            assert result.table_name == "source_metadata"

    @patch('common.bq_writer.bigquery.Client')
    def test_write_errors_success(self, mock_client_class):
        """Test successful error writing"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock successful write batch
        mock_write_result = WriteResult(
            rows_written=1,
            errors_count=0,
            processing_time_ms=300,
            run_id="test-run-id",
            table_name="source_metadata_errors",
            success=True
        )

        writer = BigQueryWriter()
        with patch.object(writer, '_write_batch', return_value=mock_write_result):
            errors = [
                ParseError(
                    source_type=SourceType.COBOL,
                    source_uri="file:///copybook.cbl",
                    error_class=ErrorClass.PARSING,
                    error_msg="Invalid COBOL syntax",
                    collected_at=datetime.now(timezone.utc)
                )
            ]

            result = writer.write_errors(errors, "test-run-id")

            assert result.success is True
            assert result.rows_written == 1
            assert result.table_name == "source_metadata_errors"

    @patch('common.bq_writer.bigquery.Client')
    def test_log_ingestion_run_success(self, mock_client_class):
        """Test successful ingestion run logging"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock successful write batch
        mock_write_result = WriteResult(
            rows_written=1,
            errors_count=0,
            processing_time_ms=200,
            run_id="test-run-id",
            table_name="ingestion_log",
            success=True
        )

        writer = BigQueryWriter()
        with patch.object(writer, '_write_batch', return_value=mock_write_result):
            run_info = {
                "run_id": "test-run-id",
                "source_type": "kubernetes",
                "started_at": datetime.now(timezone.utc),
                "status": "completed",
                "files_processed": 10
            }

            result = writer.log_ingestion_run(run_info)

            assert result.success is True
            assert result.rows_written == 1
            assert result.table_name == "ingestion_log"

    @patch('common.bq_writer.bigquery.Client')
    def test_ingestion_run_context_success(self, mock_client_class):
        """Test ingestion run context manager"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        # Mock the log_ingestion_run method
        mock_start_result = WriteResult(
            rows_written=1, errors_count=0, processing_time_ms=100,
            run_id="test-run", table_name="ingestion_log", success=True
        )
        mock_end_result = WriteResult(
            rows_written=1, errors_count=0, processing_time_ms=100,
            run_id="test-run", table_name="ingestion_log", success=True
        )

        with patch.object(writer, 'log_ingestion_run', side_effect=[mock_start_result, mock_end_result]):
            with writer.ingestion_run_context(
                run_id="test-run",
                source_type="kubernetes",
                source_uri="file:///test",
                config_used={"batch_size": 1000}
            ) as context:
                # Simulate some work
                context["files_processed"] = 5
                context["chunks_created"] = 25

            # Context manager should call log_ingestion_run twice (start and end)
            assert writer.log_ingestion_run.call_count == 2

    @patch('common.bq_writer.bigquery.Client')
    def test_ingestion_run_context_with_exception(self, mock_client_class):
        """Test ingestion run context manager with exception"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        # Mock the log_ingestion_run method
        mock_result = WriteResult(
            rows_written=1, errors_count=0, processing_time_ms=100,
            run_id="test-run", table_name="ingestion_log", success=True
        )

        with patch.object(writer, 'log_ingestion_run', return_value=mock_result):
            with pytest.raises(ValueError):
                with writer.ingestion_run_context(
                    run_id="test-run",
                    source_type="kubernetes",
                    source_uri="file:///test"
                ) as context:
                    raise ValueError("Test exception")

            # Should still log the run completion with error status
            assert writer.log_ingestion_run.call_count == 2

            # Check that the final call logged the error
            final_call = writer.log_ingestion_run.call_args_list[1]
            run_info = final_call[0][0]  # First argument of second call
            assert run_info["status"] == "failed"
            assert "Test exception" in run_info["error_summary"]


class TestBigQueryWriterEdgeCases:
    """Test edge cases and error conditions"""

    @patch('common.bq_writer.bigquery.Client')
    def test_empty_chunks_list(self, mock_client_class):
        """Test writing empty chunks list"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        result = writer.write_chunks([], "test-run-id")

        assert result.rows_written == 0
        assert result.success is True
        assert result.table_name == "source_metadata"

    @patch('common.bq_writer.bigquery.Client')
    def test_empty_errors_list(self, mock_client_class):
        """Test writing empty errors list"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        result = writer.write_errors([], "test-run-id")

        assert result.rows_written == 0
        assert result.success is True
        assert result.table_name == "source_metadata_errors"

    @patch('common.bq_writer.bigquery.Client')
    def test_chunk_with_none_values(self, mock_client_class):
        """Test chunk conversion with None values"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        chunk = ChunkMetadata(
            source_type=SourceType.MUMPS,
            artifact_id="mumps://TEST/node",
            content_text="TEST CONTENT",
            content_hash="hash123",
            source_uri="file:///test.m",
            collected_at=datetime.now(timezone.utc),
            # All optional fields as None
            parent_id=None,
            parent_type=None,
            content_tokens=None,
            repo_ref=None,
            source_metadata=None
        )

        row = writer._chunk_to_row(chunk)

        assert row["parent_id"] is None
        assert row["parent_type"] is None
        assert row["content_tokens"] is None
        assert row["repo_ref"] is None
        assert row["source_metadata"] is None

    @patch('common.bq_writer.bigquery.Client')
    def test_error_with_none_values(self, mock_client_class):
        """Test error conversion with None values"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        error = ParseError(
            source_type=SourceType.IRS,
            source_uri="file:///layout.txt",
            error_class=ErrorClass.VALIDATION,
            error_msg="Invalid layout format",
            collected_at=datetime.now(timezone.utc),
            # Optional fields as None
            sample_text=None,
            stack_trace=None
        )

        row = writer._error_to_row(error)

        assert row["sample_text"] is None
        assert row["stack_trace"] is None

    @patch('common.bq_writer.bigquery.Client')
    def test_enum_serialization(self, mock_client_class):
        """Test proper enum serialization"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        # Test all source types
        for source_type in SourceType:
            chunk = ChunkMetadata(
                source_type=source_type,
                artifact_id=f"{source_type.value}://test",
                content_text="test",
                content_hash="hash",
                source_uri="file:///test",
                collected_at=datetime.now(timezone.utc)
            )

            row = writer._chunk_to_row(chunk)
            assert row["source_type"] == source_type.value

        # Test all error classes
        for error_class in ErrorClass:
            error = ParseError(
                source_type=SourceType.KUBERNETES,
                source_uri="file:///test",
                error_class=error_class,
                error_msg="test error",
                collected_at=datetime.now(timezone.utc)
            )

            row = writer._error_to_row(error)
            assert row["error_class"] == error_class.value

    @patch('common.bq_writer.bigquery.Client')
    def test_large_content_handling(self, mock_client_class):
        """Test handling of large content"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        # Create chunk with very large content
        large_content = "x" * 100000  # 100KB content

        chunk = ChunkMetadata(
            source_type=SourceType.KUBERNETES,
            artifact_id="k8s://test/large",
            content_text=large_content,
            content_hash="large_hash",
            source_uri="file:///large.yaml",
            collected_at=datetime.now(timezone.utc)
        )

        row = writer._chunk_to_row(chunk)

        assert len(row["content_text"]) == 100000
        assert row["content_text"] == large_content

    @patch('common.bq_writer.bigquery.Client')
    def test_unicode_content_handling(self, mock_client_class):
        """Test handling of Unicode content"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        writer = BigQueryWriter()

        unicode_content = "Hello 世界! 🌍 Ñiño café"

        chunk = ChunkMetadata(
            source_type=SourceType.FASTAPI,
            artifact_id="py://unicode_test.py#1-5",
            content_text=unicode_content,
            content_hash="unicode_hash",
            source_uri="file:///unicode_test.py",
            collected_at=datetime.now(timezone.utc)
        )

        row = writer._chunk_to_row(chunk)

        assert row["content_text"] == unicode_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])