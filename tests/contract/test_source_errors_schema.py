"""
Contract test for source_metadata_errors table schema validation
Tests BigQuery error tracking table structure against expected schema

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
def test_source_metadata_errors_table_exists(bigquery_client, dataset_id):
    """Test that source_metadata_errors table exists"""
    table_id = f"{bigquery_client.project}.{dataset_id}.source_metadata_errors"

    try:
        table = bigquery_client.get_table(table_id)
        assert table.table_id == "source_metadata_errors"
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")


@pytest.mark.contract
@pytest.mark.bigquery
def test_source_metadata_errors_required_fields(bigquery_client, dataset_id):
    """Test that all required error tracking fields exist with correct types"""
    table_id = f"{bigquery_client.project}.{dataset_id}.source_metadata_errors"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Expected error tracking fields with their types
    expected_fields = {
        "error_id": "STRING",
        "source_type": "STRING",
        "source_uri": "STRING",
        "error_class": "STRING",
        "error_msg": "STRING",
        "sample_text": "STRING",
        "collected_at": "TIMESTAMP",
        "tool_version": "STRING",
        "stack_trace": "STRING",
    }

    # Get actual fields
    actual_fields = {field.name: field.field_type for field in table.schema}

    # Verify each expected field exists with correct type
    for field_name, expected_type in expected_fields.items():
        assert (
            field_name in actual_fields
        ), f"Required field '{field_name}' missing from error schema"
        assert (
            actual_fields[field_name] == expected_type
        ), f"Field '{field_name}' has type '{actual_fields[field_name]}', expected '{expected_type}'"


@pytest.mark.contract
@pytest.mark.bigquery
def test_source_metadata_errors_field_modes(bigquery_client, dataset_id):
    """Test that required fields are marked as REQUIRED and optional fields as NULLABLE"""
    table_id = f"{bigquery_client.project}.{dataset_id}.source_metadata_errors"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Required fields for error tracking
    required_fields = {
        "error_id",
        "source_type",
        "source_uri",
        "error_class",
        "error_msg",
        "collected_at",
        "tool_version",
    }

    # Optional fields (can be NULL)
    nullable_fields = {"sample_text", "stack_trace"}

    field_modes = {field.name: field.mode for field in table.schema}

    # Check required fields
    for field_name in required_fields:
        assert (
            field_name in field_modes
        ), f"Required field '{field_name}' missing from error schema"
        assert (
            field_modes[field_name] == "REQUIRED"
        ), f"Error field '{field_name}' should be REQUIRED, got '{field_modes[field_name]}'"

    # Check nullable fields
    for field_name in nullable_fields:
        if field_name in field_modes:  # Field exists
            assert (
                field_modes[field_name] == "NULLABLE"
            ), f"Error field '{field_name}' should be NULLABLE, got '{field_modes[field_name]}'"


@pytest.mark.contract
@pytest.mark.bigquery
def test_source_metadata_errors_partitioning_and_clustering(
    bigquery_client, dataset_id
):
    """Test that errors table has correct partitioning and clustering configuration"""
    table_id = f"{bigquery_client.project}.{dataset_id}.source_metadata_errors"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Check partitioning (by collected_at)
    assert table.time_partitioning is not None, "Errors table should be partitioned"
    assert (
        table.time_partitioning.field == "collected_at"
    ), f"Expected partitioning on 'collected_at', got '{table.time_partitioning.field}'"

    # Check clustering (by error_class, source_type)
    assert table.clustering_fields is not None, "Errors table should have clustering"
    expected_clustering = ["error_class", "source_type"]
    assert (
        table.clustering_fields == expected_clustering
    ), f"Expected clustering on {expected_clustering}, got {table.clustering_fields}"


@pytest.mark.contract
@pytest.mark.bigquery
def test_source_metadata_errors_constraints(bigquery_client, dataset_id):
    """Test that error fields have appropriate constraints for data quality"""
    table_id = f"{bigquery_client.project}.{dataset_id}.source_metadata_errors"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Verify field presence for constraint checking
    field_names = {field.name for field in table.schema}

    # Essential error tracking fields must be present
    essential_fields = {
        "error_id",
        "source_type",
        "source_uri",
        "error_class",
        "error_msg",
        "collected_at",
        "tool_version",
    }

    for field_name in essential_fields:
        assert (
            field_name in field_names
        ), f"Essential error field '{field_name}' missing"

    # Verify error_class field can handle expected error types
    error_class_field = next(
        field for field in table.schema if field.name == "error_class"
    )
    assert error_class_field.field_type == "STRING", "error_class should be STRING type"

    # Verify source_type field matches main table
    source_type_field = next(
        field for field in table.schema if field.name == "source_type"
    )
    assert source_type_field.field_type == "STRING", "source_type should be STRING type"


@pytest.mark.contract
@pytest.mark.bigquery
def test_source_metadata_errors_field_descriptions(bigquery_client, dataset_id):
    """Test that error fields have appropriate descriptions for documentation"""
    table_id = f"{bigquery_client.project}.{dataset_id}.source_metadata_errors"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Check that key fields have descriptions (if implemented in DDL)
    key_fields_requiring_description = {
        "error_id",
        "error_class",
        "error_msg",
        "sample_text",
    }

    field_descriptions = {field.name: field.description for field in table.schema}

    # Note: This test may pass even without descriptions if DDL doesn't include them
    # It serves as documentation of the expectation
    for field_name in key_fields_requiring_description:
        if field_name in field_descriptions and field_descriptions[field_name]:
            # If description exists, it should be meaningful
            description = field_descriptions[field_name]
            assert (
                len(description) > 10
            ), f"Field '{field_name}' description too short: '{description}'"

