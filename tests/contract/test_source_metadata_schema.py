"""
Contract test for source_metadata table schema validation
Tests BigQuery table structure against expected schema definition

This test MUST FAIL initially (TDD) until BigQuery tables are created
"""

import os

import pytest
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


@pytest.fixture
def bigquery_client():
    """Create BigQuery client for testing"""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "konveyn2ai")
    return bigquery.Client(project=project_id)


@pytest.fixture
def dataset_id():
    """Get BigQuery dataset ID from environment"""
    return os.getenv("BIGQUERY_INGESTION_DATASET_ID", "source_ingestion")


@pytest.mark.contract
@pytest.mark.bigquery
def test_source_metadata_table_exists(bigquery_client, dataset_id):
    """Test that source_metadata table exists"""
    table_id = f"{bigquery_client.project}.{dataset_id}.source_metadata"

    try:
        table = bigquery_client.get_table(table_id)
        assert table.table_id == "source_metadata"
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")


@pytest.mark.contract
@pytest.mark.bigquery
def test_source_metadata_required_common_fields(bigquery_client, dataset_id):
    """Test that all required common fields exist with correct types"""
    table_id = f"{bigquery_client.project}.{dataset_id}.source_metadata"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Expected common fields with their types
    expected_fields = {
        "source_type": "STRING",
        "artifact_id": "STRING",
        "parent_id": "STRING",
        "parent_type": "STRING",
        "content_text": "STRING",
        "content_tokens": "INTEGER",
        "content_hash": "STRING",
        "created_at": "TIMESTAMP",
        "updated_at": "TIMESTAMP",
        "collected_at": "TIMESTAMP",
        "source_uri": "STRING",
        "repo_ref": "STRING",
        "tool_version": "STRING",
    }

    # Get actual fields
    actual_fields = {field.name: field.field_type for field in table.schema}

    # Verify each expected field exists with correct type
    for field_name, expected_type in expected_fields.items():
        assert (
            field_name in actual_fields
        ), f"Required field '{field_name}' missing from schema"
        assert (
            actual_fields[field_name] == expected_type
        ), f"Field '{field_name}' has type '{actual_fields[field_name]}', expected '{expected_type}'"


@pytest.mark.contract
@pytest.mark.bigquery
def test_source_metadata_kubernetes_fields(bigquery_client, dataset_id):
    """Test that Kubernetes-specific fields exist with correct types"""
    table_id = f"{bigquery_client.project}.{dataset_id}.source_metadata"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Expected Kubernetes fields
    expected_k8s_fields = {
        "k8s_api_version": "STRING",
        "k8s_kind": "STRING",
        "k8s_namespace": "STRING",
        "k8s_resource_name": "STRING",
        "k8s_labels": "JSON",
        "k8s_annotations": "JSON",
        "k8s_resource_version": "STRING",
    }

    actual_fields = {field.name: field.field_type for field in table.schema}

    for field_name, expected_type in expected_k8s_fields.items():
        assert (
            field_name in actual_fields
        ), f"Kubernetes field '{field_name}' missing from schema"
        assert (
            actual_fields[field_name] == expected_type
        ), f"Field '{field_name}' has type '{actual_fields[field_name]}', expected '{expected_type}'"


@pytest.mark.contract
@pytest.mark.bigquery
def test_source_metadata_fastapi_fields(bigquery_client, dataset_id):
    """Test that FastAPI-specific fields exist with correct types"""
    table_id = f"{bigquery_client.project}.{dataset_id}.source_metadata"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Expected FastAPI fields
    expected_fastapi_fields = {
        "fastapi_http_method": "STRING",
        "fastapi_route_path": "STRING",
        "fastapi_operation_id": "STRING",
        "fastapi_request_model": "STRING",
        "fastapi_response_model": "STRING",
        "fastapi_status_codes": "JSON",
        "fastapi_dependencies": "JSON",
        "fastapi_src_path": "STRING",
        "fastapi_start_line": "INTEGER",
        "fastapi_end_line": "INTEGER",
    }

    actual_fields = {field.name: field.field_type for field in table.schema}

    for field_name, expected_type in expected_fastapi_fields.items():
        assert (
            field_name in actual_fields
        ), f"FastAPI field '{field_name}' missing from schema"
        assert (
            actual_fields[field_name] == expected_type
        ), f"Field '{field_name}' has type '{actual_fields[field_name]}', expected '{expected_type}'"


@pytest.mark.contract
@pytest.mark.bigquery
def test_source_metadata_cobol_fields(bigquery_client, dataset_id):
    """Test that COBOL-specific fields exist with correct types"""
    table_id = f"{bigquery_client.project}.{dataset_id}.source_metadata"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Expected COBOL fields
    expected_cobol_fields = {
        "cobol_structure_level": "INTEGER",
        "cobol_field_names": "JSON",
        "cobol_pic_clauses": "JSON",
        "cobol_occurs_count": "INTEGER",
        "cobol_redefines": "STRING",
        "cobol_usage": "STRING",
    }

    actual_fields = {field.name: field.field_type for field in table.schema}

    for field_name, expected_type in expected_cobol_fields.items():
        assert (
            field_name in actual_fields
        ), f"COBOL field '{field_name}' missing from schema"
        assert (
            actual_fields[field_name] == expected_type
        ), f"Field '{field_name}' has type '{actual_fields[field_name]}', expected '{expected_type}'"


@pytest.mark.contract
@pytest.mark.bigquery
def test_source_metadata_irs_fields(bigquery_client, dataset_id):
    """Test that IRS-specific fields exist with correct types"""
    table_id = f"{bigquery_client.project}.{dataset_id}.source_metadata"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Expected IRS fields
    expected_irs_fields = {
        "irs_record_type": "STRING",
        "irs_layout_version": "STRING",
        "irs_start_position": "INTEGER",
        "irs_field_length": "INTEGER",
        "irs_data_type": "STRING",
        "irs_section": "STRING",
    }

    actual_fields = {field.name: field.field_type for field in table.schema}

    for field_name, expected_type in expected_irs_fields.items():
        assert (
            field_name in actual_fields
        ), f"IRS field '{field_name}' missing from schema"
        assert (
            actual_fields[field_name] == expected_type
        ), f"Field '{field_name}' has type '{actual_fields[field_name]}', expected '{expected_type}'"


@pytest.mark.contract
@pytest.mark.bigquery
def test_source_metadata_mumps_fields(bigquery_client, dataset_id):
    """Test that MUMPS-specific fields exist with correct types"""
    table_id = f"{bigquery_client.project}.{dataset_id}.source_metadata"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Expected MUMPS fields
    expected_mumps_fields = {
        "mumps_global_name": "STRING",
        "mumps_node_path": "STRING",
        "mumps_file_no": "INTEGER",
        "mumps_field_no": "FLOAT",
        "mumps_xrefs": "JSON",
        "mumps_input_transform": "STRING",
    }

    actual_fields = {field.name: field.field_type for field in table.schema}

    for field_name, expected_type in expected_mumps_fields.items():
        assert (
            field_name in actual_fields
        ), f"MUMPS field '{field_name}' missing from schema"
        assert (
            actual_fields[field_name] == expected_type
        ), f"Field '{field_name}' has type '{actual_fields[field_name]}', expected '{expected_type}'"


@pytest.mark.contract
@pytest.mark.bigquery
def test_source_metadata_partitioning_and_clustering(bigquery_client, dataset_id):
    """Test that table has correct partitioning and clustering configuration"""
    table_id = f"{bigquery_client.project}.{dataset_id}.source_metadata"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Check partitioning (by collected_at)
    assert table.time_partitioning is not None, "Table should be partitioned"
    assert (
        table.time_partitioning.field == "collected_at"
    ), f"Expected partitioning on 'collected_at', got '{table.time_partitioning.field}'"

    # Check clustering (by source_type, artifact_id)
    assert table.clustering_fields is not None, "Table should have clustering"
    expected_clustering = ["source_type", "artifact_id"]
    assert (
        table.clustering_fields == expected_clustering
    ), f"Expected clustering on {expected_clustering}, got {table.clustering_fields}"


@pytest.mark.contract
@pytest.mark.bigquery
def test_source_metadata_field_modes(bigquery_client, dataset_id):
    """Test that required fields are marked as REQUIRED and optional fields as NULLABLE"""
    table_id = f"{bigquery_client.project}.{dataset_id}.source_metadata"

    try:
        table = bigquery_client.get_table(table_id)
    except NotFound:
        pytest.fail(f"Table {table_id} does not exist - run 'make setup' first")

    # Required fields (should have mode REQUIRED)
    required_fields = {
        "source_type",
        "artifact_id",
        "content_text",
        "content_hash",
        "created_at",
        "updated_at",
        "collected_at",
        "source_uri",
        "tool_version",
    }

    # All source-specific fields should be NULLABLE
    nullable_fields = {
        "parent_id",
        "parent_type",
        "content_tokens",
        "repo_ref",
        # Kubernetes fields
        "k8s_api_version",
        "k8s_kind",
        "k8s_namespace",
        "k8s_resource_name",
        "k8s_labels",
        "k8s_annotations",
        "k8s_resource_version",
        # FastAPI fields
        "fastapi_http_method",
        "fastapi_route_path",
        "fastapi_operation_id",
        "fastapi_request_model",
        "fastapi_response_model",
        "fastapi_status_codes",
        "fastapi_dependencies",
        "fastapi_src_path",
        "fastapi_start_line",
        "fastapi_end_line",
        # COBOL fields
        "cobol_structure_level",
        "cobol_field_names",
        "cobol_pic_clauses",
        "cobol_occurs_count",
        "cobol_redefines",
        "cobol_usage",
        # IRS fields
        "irs_record_type",
        "irs_layout_version",
        "irs_start_position",
        "irs_field_length",
        "irs_data_type",
        "irs_section",
        # MUMPS fields
        "mumps_global_name",
        "mumps_node_path",
        "mumps_file_no",
        "mumps_field_no",
        "mumps_xrefs",
        "mumps_input_transform",
    }

    field_modes = {field.name: field.mode for field in table.schema}

    # Check required fields
    for field_name in required_fields:
        assert field_name in field_modes, f"Required field '{field_name}' missing"
        assert (
            field_modes[field_name] == "REQUIRED"
        ), f"Field '{field_name}' should be REQUIRED, got '{field_modes[field_name]}'"

    # Check nullable fields
    for field_name in nullable_fields:
        if field_name in field_modes:  # Field exists
            assert (
                field_modes[field_name] == "NULLABLE"
            ), f"Field '{field_name}' should be NULLABLE, got '{field_modes[field_name]}'"
