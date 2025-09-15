"""
Contract test for ingestion_log table schema validation
Tests BigQuery ingestion tracking table structure against expected schema

This test MUST FAIL initially (TDD) until BigQuery tables are created
"""

import os

import pytest
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


@pytest.fixture
def bigquery_client():
    """Create BigQuery client for testing"""
    project_id = os.getenv("BQ_PROJECT", "konveyn2ai")
    return bigquery.Client(project=project_id)


@pytest.fixture
def dataset_id():
    """Get BigQuery dataset ID from environment"""
    return os.getenv("BQ_DATASET", "source_ingestion")


@pytest.mark.contract
@pytest.mark.bigquery
def test_ingestion_log_table_exists(bigquery_client, dataset_id):
    """Test that ingestion_log table exists"""
    table_id = f"{bigquery_client.project}.{dataset_id}.ingestion_log"

    try:
        table = bigquery_client.get_table(table_id)
        assert table.table_id == "ingestion_log"
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")


@pytest.mark.contract
@pytest.mark.bigquery
def test_ingestion_log_required_fields(bigquery_client, dataset_id):
    """Test that all required ingestion tracking fields exist with correct types"""
    table_id = f"{bigquery_client.project}.{dataset_id}.ingestion_log"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Expected ingestion log fields with their types
    expected_fields = {
        "run_id": "STRING",
        "source_type": "STRING",
        "started_at": "TIMESTAMP",
        "completed_at": "TIMESTAMP",
        "status": "STRING",
        "files_processed": "INTEGER",
        "rows_written": "INTEGER",
        "rows_skipped": "INTEGER",
        "errors_count": "INTEGER",
        "bytes_written": "INTEGER",
        "processing_duration_ms": "INTEGER",
        "avg_chunk_size_tokens": "FLOAT",
        "tool_version": "STRING",
        "config_params": "JSON",
    }

    # Get actual fields
    actual_fields = {field.name: field.field_type for field in table.schema}

    # Verify each expected field exists with correct type
    for field_name, expected_type in expected_fields.items():
        assert (
            field_name in actual_fields
        ), f"Required field '{field_name}' missing from ingestion log schema"
        assert (
            actual_fields[field_name] == expected_type
        ), f"Field '{field_name}' has type '{actual_fields[field_name]}', expected '{expected_type}'"


@pytest.mark.contract
@pytest.mark.bigquery
def test_ingestion_log_field_modes(bigquery_client, dataset_id):
    """Test that required fields are marked as REQUIRED and optional fields as NULLABLE"""
    table_id = f"{bigquery_client.project}.{dataset_id}.ingestion_log"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Required fields for ingestion tracking
    required_fields = {
        "run_id",
        "source_type",
        "started_at",
        "status",
        "files_processed",
        "rows_written",
        "rows_skipped",
        "errors_count",
        "bytes_written",
        "tool_version",
    }

    # Optional fields (can be NULL)
    nullable_fields = {
        "completed_at",
        "processing_duration_ms",
        "avg_chunk_size_tokens",
        "config_params",
    }

    field_modes = {field.name: field.mode for field in table.schema}

    # Check required fields
    for field_name in required_fields:
        assert (
            field_name in field_modes
        ), f"Required field '{field_name}' missing from ingestion log schema"
        assert (
            field_modes[field_name] == "REQUIRED"
        ), f"Ingestion log field '{field_name}' should be REQUIRED, got '{field_modes[field_name]}'"

    # Check nullable fields
    for field_name in nullable_fields:
        if field_name in field_modes:  # Field exists
            assert (
                field_modes[field_name] == "NULLABLE"
            ), f"Ingestion log field '{field_name}' should be NULLABLE, got '{field_modes[field_name]}'"


@pytest.mark.contract
@pytest.mark.bigquery
def test_ingestion_log_partitioning_and_clustering(bigquery_client, dataset_id):
    """Test that ingestion log table has correct partitioning and clustering configuration"""
    table_id = f"{bigquery_client.project}.{dataset_id}.ingestion_log"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Check partitioning (by started_at)
    assert (
        table.time_partitioning is not None
    ), "Ingestion log table should be partitioned"
    assert (
        table.time_partitioning.field == "started_at"
    ), f"Expected partitioning on 'started_at', got '{table.time_partitioning.field}'"

    # Check clustering (by source_type, status)
    assert (
        table.clustering_fields is not None
    ), "Ingestion log table should have clustering"
    expected_clustering = ["source_type", "status"]
    assert (
        table.clustering_fields == expected_clustering
    ), f"Expected clustering on {expected_clustering}, got {table.clustering_fields}"


@pytest.mark.contract
@pytest.mark.bigquery
def test_ingestion_log_performance_fields(bigquery_client, dataset_id):
    """Test that performance tracking fields have appropriate types for metrics"""
    table_id = f"{bigquery_client.project}.{dataset_id}.ingestion_log"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Performance fields with expected types
    performance_fields = {
        "files_processed": "INTEGER",
        "rows_written": "INTEGER",
        "rows_skipped": "INTEGER",
        "errors_count": "INTEGER",
        "bytes_written": "INTEGER",
        "processing_duration_ms": "INTEGER",
        "avg_chunk_size_tokens": "FLOAT",
    }

    actual_fields = {field.name: field.field_type for field in table.schema}

    for field_name, expected_type in performance_fields.items():
        assert field_name in actual_fields, f"Performance field '{field_name}' missing"
        assert (
            actual_fields[field_name] == expected_type
        ), f"Performance field '{field_name}' has type '{actual_fields[field_name]}', expected '{expected_type}'"


@pytest.mark.contract
@pytest.mark.bigquery
def test_ingestion_log_status_field_constraints(bigquery_client, dataset_id):
    """Test that status field can handle expected status values"""
    table_id = f"{bigquery_client.project}.{dataset_id}.ingestion_log"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Verify status field exists and is STRING
    status_field = next(
        (field for field in table.schema if field.name == "status"), None
    )
    assert status_field is not None, "Status field missing from ingestion log"
    assert status_field.field_type == "STRING", "Status field should be STRING type"
    assert status_field.mode == "REQUIRED", "Status field should be REQUIRED"

    # Note: Expected status values are 'running', 'completed', 'failed', 'partial'
    # This constraint would typically be enforced at application level


@pytest.mark.contract
@pytest.mark.bigquery
def test_ingestion_log_run_id_uniqueness(bigquery_client, dataset_id):
    """Test that run_id field is suitable as primary key"""
    table_id = f"{bigquery_client.project}.{dataset_id}.ingestion_log"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Verify run_id field characteristics
    run_id_field = next(
        (field for field in table.schema if field.name == "run_id"), None
    )
    assert run_id_field is not None, "run_id field missing from ingestion log"
    assert run_id_field.field_type == "STRING", "run_id field should be STRING type"
    assert run_id_field.mode == "REQUIRED", "run_id field should be REQUIRED"

    # Note: ULID format ensures uniqueness and sortability


@pytest.mark.contract
@pytest.mark.bigquery
def test_ingestion_log_timestamp_fields(bigquery_client, dataset_id):
    """Test that timestamp fields are properly configured for time tracking"""
    table_id = f"{bigquery_client.project}.{dataset_id}.ingestion_log"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Timestamp fields with requirements
    timestamp_fields = {
        "started_at": "REQUIRED",  # Always set when run begins
        "completed_at": "NULLABLE",  # Set when run completes (may be NULL for failed runs)
    }

    field_info = {field.name: (field.field_type, field.mode) for field in table.schema}

    for field_name, expected_mode in timestamp_fields.items():
        assert field_name in field_info, f"Timestamp field '{field_name}' missing"
        field_type, field_mode = field_info[field_name]
        assert (
            field_type == "TIMESTAMP"
        ), f"Timestamp field '{field_name}' has type '{field_type}', expected 'TIMESTAMP'"
        assert (
            field_mode == expected_mode
        ), f"Timestamp field '{field_name}' has mode '{field_mode}', expected '{expected_mode}'"


@pytest.mark.contract
@pytest.mark.bigquery
def test_ingestion_log_config_params_json(bigquery_client, dataset_id):
    """Test that config_params field supports JSON for flexible configuration storage"""
    table_id = f"{bigquery_client.project}.{dataset_id}.ingestion_log"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Verify config_params field
    config_field = next(
        (field for field in table.schema if field.name == "config_params"), None
    )
    assert config_field is not None, "config_params field missing from ingestion log"
    assert config_field.field_type == "JSON", "config_params field should be JSON type"
    assert config_field.mode == "NULLABLE", "config_params field should be NULLABLE"
