"""
Integration test for idempotent ingestion (repeated processing)
T018: Validates that repeated ingestion of same content produces consistent results

This test ensures:
1. Same content produces same content_hash across runs
2. Repeated ingestion doesn't create duplicate rows
3. Upsert logic works correctly with content hash comparison
4. Ingestion_log tracks runs but source_metadata remains deduplicated
"""

import os
import tempfile
import pytest
from datetime import datetime, timedelta
from google.cloud import bigquery
from typing import Dict, Any, List


# TDD RED Phase: These imports will fail initially - this is expected and correct
try:
    from src.parsers.kubernetes_parser import KubernetesParserImpl
    from src.parsers.fastapi_parser import FastAPIParserImpl
    from src.parsers.cobol_parser import COBOLParserImpl
    from src.bigquery.writer import BigQueryWriterImpl
    from src.ingestion.orchestrator import IngestionOrchestrator
    PARSERS_AVAILABLE = True
except ImportError:
    PARSERS_AVAILABLE = False


@pytest.fixture
def bigquery_client():
    """BigQuery client for testing"""
    project_id = os.getenv("BQ_PROJECT", "konveyn2ai")
    return bigquery.Client(project=project_id)


@pytest.fixture
def temp_bigquery_dataset(bigquery_client):
    """Create temporary BigQuery dataset for testing"""
    dataset_id = f"test_idempotency_{int(datetime.now().timestamp())}"
    dataset_ref = bigquery_client.dataset(dataset_id)

    dataset = bigquery.Dataset(dataset_ref)
    dataset.default_table_expiration_ms = 24 * 60 * 60 * 1000  # 24 hours
    dataset.location = "US"

    try:
        dataset = bigquery_client.create_dataset(dataset, timeout=30)
        yield dataset_id
    finally:
        bigquery_client.delete_dataset(dataset_ref, delete_contents=True, not_found_ok=True)


@pytest.fixture
def create_test_tables(bigquery_client, temp_bigquery_dataset):
    """Create test tables in temporary dataset"""
    dataset_id = temp_bigquery_dataset

    # Read DDL from contract
    ddl_path = "specs/002-m1-parse-and/contracts/bigquery-ddl.sql"
    with open(ddl_path, "r") as f:
        ddl_content = f.read()

    # Replace placeholders
    ddl_content = ddl_content.replace("${BQ_PROJECT}", bigquery_client.project)
    ddl_content = ddl_content.replace("${BQ_DATASET}", dataset_id)

    # Execute DDL statements
    for statement in ddl_content.split(";"):
        if statement.strip():
            bigquery_client.query(statement).result()

    return dataset_id


@pytest.fixture
def sample_k8s_manifest():
    """Sample Kubernetes manifest for testing"""
    return """
apiVersion: v1
kind: ConfigMap
metadata:
  name: test-config
  namespace: default
data:
  app.yaml: |
    name: test-app
    version: 1.0.0
  nginx.conf: |
    server {
        listen 80;
        server_name example.com;
    }
"""


@pytest.fixture
def sample_fastapi_code():
    """Sample FastAPI code for testing"""
    return '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    email: str

@app.get("/users/{user_id}")
async def get_user(user_id: int) -> User:
    """Get user by ID"""
    return User(id=user_id, name="Test User", email="test@example.com")

@app.post("/users")
async def create_user(user: User) -> User:
    """Create new user"""
    return user
'''


@pytest.fixture
def temp_test_files(sample_k8s_manifest, sample_fastapi_code):
    """Create temporary test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # K8s manifest
        k8s_file = os.path.join(temp_dir, "configmap.yaml")
        with open(k8s_file, "w") as f:
            f.write(sample_k8s_manifest)

        # FastAPI code
        fastapi_file = os.path.join(temp_dir, "main.py")
        with open(fastapi_file, "w") as f:
            f.write(sample_fastapi_code)

        yield {
            "k8s_file": k8s_file,
            "fastapi_file": fastapi_file,
            "temp_dir": temp_dir
        }


class TestIdempotencyIntegration:
    """Integration tests for idempotent ingestion behavior"""

    @pytest.mark.integration
    @pytest.mark.bigquery
    def test_parser_implementations_not_available(self):
        """TDD RED: Verify parser implementations don't exist yet"""
        assert not PARSERS_AVAILABLE, (
            "Parser implementations should not exist yet in TDD RED phase. "
            "This test should fail until implementations are created."
        )

    @pytest.mark.integration
    @pytest.mark.bigquery
    @pytest.mark.skipif(not PARSERS_AVAILABLE, reason="Parser implementations not available")
    def test_kubernetes_idempotent_ingestion(self, bigquery_client, create_test_tables, temp_test_files):
        """Test that repeated K8s ingestion produces consistent results"""
        dataset_id = create_test_tables
        k8s_file = temp_test_files["k8s_file"]

        # Initialize components
        parser = KubernetesParserImpl()
        writer = BigQueryWriterImpl(bigquery_client, dataset_id)
        orchestrator = IngestionOrchestrator(parser, writer)

        # First ingestion run
        result1 = orchestrator.ingest_file(k8s_file)
        run_id1 = result1.run_id

        # Verify first run created data
        metadata_query = f"""
        SELECT artifact_id, content_hash, content_text
        FROM `{bigquery_client.project}.{dataset_id}.source_metadata`
        WHERE source_type = 'kubernetes'
        """
        metadata_rows1 = list(bigquery_client.query(metadata_query).result())
        assert len(metadata_rows1) > 0, "First ingestion should create metadata rows"

        # Second ingestion run (same file)
        result2 = orchestrator.ingest_file(k8s_file)
        run_id2 = result2.run_id

        # Verify runs are different but metadata is same
        assert run_id1 != run_id2, "Each ingestion should get unique run_id"

        metadata_rows2 = list(bigquery_client.query(metadata_query).result())
        assert len(metadata_rows2) == len(metadata_rows1), (
            "Repeated ingestion should not create duplicate metadata rows"
        )

        # Compare content hashes - should be identical
        for row1, row2 in zip(metadata_rows1, metadata_rows2):
            assert row1.artifact_id == row2.artifact_id, "Artifact IDs should be consistent"
            assert row1.content_hash == row2.content_hash, "Content hashes should be identical"
            assert row1.content_text == row2.content_text, "Content should be identical"

        # Verify ingestion_log has both runs
        log_query = f"""
        SELECT run_id, source_type, files_processed, chunks_created
        FROM `{bigquery_client.project}.{dataset_id}.ingestion_log`
        WHERE source_type = 'kubernetes'
        ORDER BY started_at
        """
        log_rows = list(bigquery_client.query(log_query).result())
        assert len(log_rows) == 2, "Both ingestion runs should be logged"
        assert log_rows[0].run_id == run_id1
        assert log_rows[1].run_id == run_id2

    @pytest.mark.integration
    @pytest.mark.bigquery
    @pytest.mark.skipif(not PARSERS_AVAILABLE, reason="Parser implementations not available")
    def test_fastapi_idempotent_ingestion(self, bigquery_client, create_test_tables, temp_test_files):
        """Test that repeated FastAPI ingestion produces consistent results"""
        dataset_id = create_test_tables
        fastapi_file = temp_test_files["fastapi_file"]

        # Initialize components
        parser = FastAPIParserImpl()
        writer = BigQueryWriterImpl(bigquery_client, dataset_id)
        orchestrator = IngestionOrchestrator(parser, writer)

        # First ingestion run
        result1 = orchestrator.ingest_file(fastapi_file)

        # Second ingestion run (same file)
        result2 = orchestrator.ingest_file(fastapi_file)

        # Verify metadata consistency
        metadata_query = f"""
        SELECT artifact_id, content_hash, COUNT(*) as row_count
        FROM `{bigquery_client.project}.{dataset_id}.source_metadata`
        WHERE source_type = 'fastapi'
        GROUP BY artifact_id, content_hash
        HAVING COUNT(*) > 1
        """
        duplicate_rows = list(bigquery_client.query(metadata_query).result())
        assert len(duplicate_rows) == 0, "No duplicate metadata rows should exist"

        # Verify both runs logged but metadata deduplicated
        total_metadata = bigquery_client.query(f"""
        SELECT COUNT(*) as count
        FROM `{bigquery_client.project}.{dataset_id}.source_metadata`
        WHERE source_type = 'fastapi'
        """).result()
        metadata_count = list(total_metadata)[0].count

        total_runs = bigquery_client.query(f"""
        SELECT COUNT(*) as count
        FROM `{bigquery_client.project}.{dataset_id}.ingestion_log`
        WHERE source_type = 'fastapi'
        """).result()
        run_count = list(total_runs)[0].count

        assert run_count == 2, "Both ingestion runs should be logged"
        assert metadata_count >= 1, "At least one metadata row should exist"

    @pytest.mark.integration
    @pytest.mark.bigquery
    @pytest.mark.skipif(not PARSERS_AVAILABLE, reason="Parser implementations not available")
    def test_modified_file_creates_new_hash(self, bigquery_client, create_test_tables, temp_test_files):
        """Test that modified files create new content hashes"""
        dataset_id = create_test_tables
        k8s_file = temp_test_files["k8s_file"]

        # Initialize components
        parser = KubernetesParserImpl()
        writer = BigQueryWriterImpl(bigquery_client, dataset_id)
        orchestrator = IngestionOrchestrator(parser, writer)

        # First ingestion
        result1 = orchestrator.ingest_file(k8s_file)

        # Modify the file
        with open(k8s_file, "a") as f:
            f.write("\n  modified: true\n")

        # Second ingestion (modified file)
        result2 = orchestrator.ingest_file(k8s_file)

        # Verify different content hashes exist
        hash_query = f"""
        SELECT DISTINCT content_hash
        FROM `{bigquery_client.project}.{dataset_id}.source_metadata`
        WHERE source_type = 'kubernetes'
        """
        hashes = list(bigquery_client.query(hash_query).result())
        assert len(hashes) >= 2, "Modified file should produce different content hash"

    @pytest.mark.integration
    @pytest.mark.bigquery
    @pytest.mark.skipif(not PARSERS_AVAILABLE, reason="Parser implementations not available")
    def test_concurrent_ingestion_safety(self, bigquery_client, create_test_tables, temp_test_files):
        """Test that concurrent ingestion of same file is safe"""
        import concurrent.futures
        import threading

        dataset_id = create_test_tables
        k8s_file = temp_test_files["k8s_file"]

        def ingest_file():
            """Ingest file in separate thread"""
            parser = KubernetesParserImpl()
            writer = BigQueryWriterImpl(bigquery_client, dataset_id)
            orchestrator = IngestionOrchestrator(parser, writer)
            return orchestrator.ingest_file(k8s_file)

        # Run 3 concurrent ingestions
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(ingest_file) for _ in range(3)]
            results = [future.result() for future in futures]

        # Verify all runs completed
        assert len(results) == 3
        assert all(result.run_id for result in results)

        # Verify unique run IDs
        run_ids = [result.run_id for result in results]
        assert len(set(run_ids)) == 3, "All runs should have unique IDs"

        # Verify metadata not duplicated
        metadata_query = f"""
        SELECT artifact_id, content_hash, COUNT(*) as row_count
        FROM `{bigquery_client.project}.{dataset_id}.source_metadata`
        WHERE source_type = 'kubernetes'
        GROUP BY artifact_id, content_hash
        HAVING COUNT(*) > 1
        """
        duplicate_rows = list(bigquery_client.query(metadata_query).result())
        assert len(duplicate_rows) == 0, "Concurrent ingestion should not create duplicates"

        # Verify all runs logged
        log_count_query = f"""
        SELECT COUNT(*) as count
        FROM `{bigquery_client.project}.{dataset_id}.ingestion_log`
        WHERE source_type = 'kubernetes'
        """
        log_count = list(bigquery_client.query(log_count_query).result())[0].count
        assert log_count == 3, "All concurrent runs should be logged"

    @pytest.mark.integration
    @pytest.mark.bigquery
    @pytest.mark.skipif(not PARSERS_AVAILABLE, reason="Parser implementations not available")
    def test_cross_parser_artifact_id_uniqueness(self, bigquery_client, create_test_tables, temp_test_files):
        """Test that artifact IDs are unique across different parser types"""
        dataset_id = create_test_tables

        # Ingest with different parsers
        k8s_parser = KubernetesParserImpl()
        fastapi_parser = FastAPIParserImpl()
        writer = BigQueryWriterImpl(bigquery_client, dataset_id)

        k8s_orchestrator = IngestionOrchestrator(k8s_parser, writer)
        fastapi_orchestrator = IngestionOrchestrator(fastapi_parser, writer)

        # Ingest files with same name but different parsers
        k8s_result = k8s_orchestrator.ingest_file(temp_test_files["k8s_file"])
        fastapi_result = fastapi_orchestrator.ingest_file(temp_test_files["fastapi_file"])

        # Verify artifact IDs are unique across parsers
        artifact_query = f"""
        SELECT artifact_id, source_type, COUNT(*) as count
        FROM `{bigquery_client.project}.{dataset_id}.source_metadata`
        GROUP BY artifact_id, source_type
        HAVING COUNT(*) > 1
        """
        duplicate_artifacts = list(bigquery_client.query(artifact_query).result())
        assert len(duplicate_artifacts) == 0, "Artifact IDs should be unique across parsers"

        # Verify source types are correctly set
        source_types_query = f"""
        SELECT DISTINCT source_type
        FROM `{bigquery_client.project}.{dataset_id}.source_metadata`
        ORDER BY source_type
        """
        source_types = [row.source_type for row in bigquery_client.query(source_types_query).result()]
        assert "kubernetes" in source_types
        assert "fastapi" in source_types

    @pytest.mark.integration
    @pytest.mark.bigquery
    @pytest.mark.skipif(not PARSERS_AVAILABLE, reason="Parser implementations not available")
    def test_error_logging_idempotency(self, bigquery_client, create_test_tables, temp_test_files):
        """Test that error logging is also idempotent"""
        dataset_id = create_test_tables

        # Create an invalid file
        invalid_file = os.path.join(temp_test_files["temp_dir"], "invalid.yaml")
        with open(invalid_file, "w") as f:
            f.write("invalid: yaml: content: [\n")  # Malformed YAML

        parser = KubernetesParserImpl()
        writer = BigQueryWriterImpl(bigquery_client, dataset_id)
        orchestrator = IngestionOrchestrator(parser, writer)

        # First ingestion (should log error)
        result1 = orchestrator.ingest_file(invalid_file)

        # Second ingestion (should log same error)
        result2 = orchestrator.ingest_file(invalid_file)

        # Verify errors logged but not duplicated by content
        error_query = f"""
        SELECT source_uri, error_msg, COUNT(*) as count
        FROM `{bigquery_client.project}.{dataset_id}.source_metadata_errors`
        WHERE source_type = 'kubernetes'
        GROUP BY source_uri, error_msg
        """
        error_counts = list(bigquery_client.query(error_query).result())

        # Should have errors logged (2 runs) but content-based deduplication depends on implementation
        assert len(error_counts) >= 1, "Errors should be logged"

        # Verify run tracking shows both attempts
        run_query = f"""
        SELECT COUNT(*) as count
        FROM `{bigquery_client.project}.{dataset_id}.ingestion_log`
        WHERE source_type = 'kubernetes'
        """
        run_count = list(bigquery_client.query(run_query).result())[0].count
        assert run_count == 2, "Both error runs should be tracked"

    @pytest.mark.integration
    @pytest.mark.bigquery
    def test_sample_files_are_valid(self, sample_k8s_manifest, sample_fastapi_code):
        """Verify test sample files are syntactically valid"""
        import yaml
        import ast

        # Validate K8s YAML
        try:
            k8s_data = yaml.safe_load(sample_k8s_manifest)
            assert k8s_data["kind"] == "ConfigMap"
            assert k8s_data["metadata"]["name"] == "test-config"
        except yaml.YAMLError as e:
            pytest.fail(f"Sample K8s manifest is invalid YAML: {e}")

        # Validate FastAPI Python code
        try:
            ast.parse(sample_fastapi_code)
        except SyntaxError as e:
            pytest.fail(f"Sample FastAPI code has syntax errors: {e}")

    @pytest.mark.integration
    @pytest.mark.bigquery
    def test_bigquery_tables_exist(self, bigquery_client, create_test_tables):
        """Verify BigQuery test tables are created correctly"""
        dataset_id = create_test_tables

        expected_tables = ["source_metadata", "source_metadata_errors", "ingestion_log"]

        for table_name in expected_tables:
            table_ref = bigquery_client.dataset(dataset_id).table(table_name)
            table = bigquery_client.get_table(table_ref)
            assert table.table_id == table_name
            assert len(table.schema) > 0, f"Table {table_name} should have schema fields"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])