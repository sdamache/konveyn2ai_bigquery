"""
Integration tests for Kubernetes manifest ingestion to BigQuery.

This is an end-to-end test that validates the complete flow from K8s YAML/JSON manifests
to BigQuery source_metadata table. Tests use real BigQuery but with temporary datasets
for isolation.

Requirements:
- Follows TDD - tests MUST FAIL initially (no implementation exists yet)
- Uses real BigQuery (no mocks) but temp datasets for isolation
- Tests manifest parsing, chunk generation, metadata extraction, BigQuery ingestion
- Includes sample K8s manifests (deployment, service, configmap examples)
- Tests both file and directory ingestion modes
- Validates row counts, content hashing, artifact IDs
- Uses pytest markers: @pytest.mark.integration, @pytest.mark.bigquery
"""

import os
import json
import yaml
import tempfile
import shutil
import pytest
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock

# Import BigQuery client
try:
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

# Import parser interface contracts
try:
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0, project_root)

    import importlib.util
    parser_interfaces_path = os.path.join(project_root, "specs", "002-m1-parse-and", "contracts", "parser-interfaces.py")
    spec = importlib.util.spec_from_file_location("parser_interfaces", parser_interfaces_path)
    parser_interfaces = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_interfaces)

    SourceType = parser_interfaces.SourceType
    ChunkMetadata = parser_interfaces.ChunkMetadata
    ParseResult = parser_interfaces.ParseResult
    ParseError = parser_interfaces.ParseError
    ErrorClass = parser_interfaces.ErrorClass
    KubernetesParser = parser_interfaces.KubernetesParser
    BigQueryWriter = parser_interfaces.BigQueryWriter

    PARSER_INTERFACES_AVAILABLE = True
except (ImportError, AttributeError, FileNotFoundError) as e:
    PARSER_INTERFACES_AVAILABLE = False
    print(f"Warning: Could not import parser interfaces: {e}")

# Try to import the actual implementation (expected to fail initially)
try:
    from src.parsers.kubernetes_parser import KubernetesParserImpl
    K8S_PARSER_AVAILABLE = True
except ImportError:
    K8S_PARSER_AVAILABLE = False

# Try to import BigQuery writer implementation (expected to fail initially)
try:
    from src.ingestion.bigquery_writer import BigQueryWriterImpl
    BIGQUERY_WRITER_AVAILABLE = True
except ImportError:
    BIGQUERY_WRITER_AVAILABLE = False

# Try to import ingestion orchestrator (expected to fail initially)
try:
    from src.ingestion.k8s_ingestion import KubernetesIngestionOrchestrator
    INGESTION_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    INGESTION_ORCHESTRATOR_AVAILABLE = False


# Sample Kubernetes manifests for testing
SAMPLE_DEPLOYMENT_MANIFEST = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  namespace: production
  labels:
    app: web-app
    version: "2.1.0"
    tier: frontend
  annotations:
    deployment.kubernetes.io/revision: "3"
    konveyn2ai.com/parser: "kubernetes"
    konveyn2ai.com/ingested-at: "2025-01-15T10:30:00Z"
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
        version: "2.1.0"
    spec:
      containers:
      - name: web-app
        image: nginx:1.21-alpine
        ports:
        - containerPort: 8080
          protocol: TCP
        env:
        - name: APP_ENV
          value: "production"
        - name: DB_CONNECTION
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: connection-string
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      imagePullSecrets:
      - name: registry-secret
"""

SAMPLE_SERVICE_MANIFEST = """
apiVersion: v1
kind: Service
metadata:
  name: web-app-service
  namespace: production
  labels:
    app: web-app
    component: load-balancer
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    external-dns.alpha.kubernetes.io/hostname: "web-app.example.com"
spec:
  type: LoadBalancer
  selector:
    app: web-app
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  - name: https
    port: 443
    targetPort: 8080
    protocol: TCP
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800
"""

SAMPLE_CONFIGMAP_MANIFEST = """
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: production
  labels:
    app: web-app
    config-type: application
data:
  application.yml: |
    server:
      port: 8080
      servlet:
        context-path: /api
    spring:
      profiles:
        active: production
      datasource:
        driver-class-name: org.postgresql.Driver
        url: jdbc:postgresql://db.production.svc.cluster.local:5432/webapp
    logging:
      level:
        root: INFO
        com.example: DEBUG
      pattern:
        file: "%d{ISO8601} [%thread] %-5level %logger{36} - %msg%n"
  nginx.conf: |
    server {
        listen 8080;
        server_name _;
        location / {
            proxy_pass http://localhost:3000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        location /health {
            access_log off;
            return 200 "healthy\\n";
        }
    }
  feature-flags.json: |
    {
      "authentication": {
        "enable_oauth": true,
        "enable_ldap": false,
        "session_timeout": 3600
      },
      "features": {
        "enable_analytics": true,
        "enable_caching": true,
        "enable_rate_limiting": false
      }
    }
"""

SAMPLE_SECRET_MANIFEST = """
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
  namespace: production
  labels:
    app: web-app
    secret-type: database
type: Opaque
data:
  connection-string: cG9zdGdyZXNxbDovL3VzZXI6cGFzc3dvcmRAZGIucHJvZHVjdGlvbi5zdmMuY2x1c3Rlci5sb2NhbDo1NDMyL3dlYmFwcA==
  username: dXNlcg==
  password: cGFzc3dvcmQ=
"""

INVALID_YAML_MANIFEST = """
apiVersion: v1
kind: Pod
metadata:
  name: invalid-pod
  namespace: default
spec:
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
      invalidKey: [unclosed bracket and invalid yaml
"""

MULTI_DOCUMENT_MANIFEST = f"""
{SAMPLE_DEPLOYMENT_MANIFEST}
---
{SAMPLE_SERVICE_MANIFEST}
---
{SAMPLE_CONFIGMAP_MANIFEST}
---
{SAMPLE_SECRET_MANIFEST}
"""

# BigQuery test configuration
TEST_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "konveyn2ai")
TEST_DATASET_PREFIX = "test_k8s_ingestion"


@pytest.fixture(scope="session")
def bigquery_client():
    """BigQuery client for integration tests"""
    if not BIGQUERY_AVAILABLE:
        pytest.skip("BigQuery client not available")

    return bigquery.Client(project=TEST_PROJECT_ID)


@pytest.fixture(scope="function")
def temp_dataset(bigquery_client):
    """Create temporary BigQuery dataset for each test"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    dataset_id = f"{TEST_DATASET_PREFIX}_{timestamp}"

    dataset = bigquery.Dataset(f"{TEST_PROJECT_ID}.{dataset_id}")
    dataset.location = "US"
    dataset.description = "Temporary dataset for K8s ingestion integration tests"

    # Set short expiration for cleanup
    dataset.default_table_expiration_ms = 3600000  # 1 hour

    created_dataset = bigquery_client.create_dataset(dataset, timeout=30)

    yield created_dataset

    # Cleanup: Delete the dataset
    try:
        bigquery_client.delete_dataset(dataset_id, delete_contents=True, not_found_ok=True)
    except Exception as e:
        print(f"Warning: Failed to cleanup test dataset {dataset_id}: {e}")


@pytest.fixture(scope="function")
def temp_manifest_dir():
    """Create temporary directory with sample K8s manifests"""
    temp_dir = tempfile.mkdtemp(prefix="k8s_manifests_")

    try:
        # Create sample manifest files
        manifest_files = {
            "deployment.yaml": SAMPLE_DEPLOYMENT_MANIFEST,
            "service.yml": SAMPLE_SERVICE_MANIFEST,
            "configmap.yaml": SAMPLE_CONFIGMAP_MANIFEST,
            "secret.yaml": SAMPLE_SECRET_MANIFEST,
            "multi-resource.yaml": MULTI_DOCUMENT_MANIFEST,
            "invalid.yaml": INVALID_YAML_MANIFEST,
            "not-a-manifest.txt": "This is not a Kubernetes manifest file",
            "README.md": "# Sample K8s manifests for testing"
        }

        for filename, content in manifest_files.items():
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w') as f:
                f.write(content)

        # Create subdirectory with more manifests
        sub_dir = os.path.join(temp_dir, "overlays", "production")
        os.makedirs(sub_dir, exist_ok=True)

        with open(os.path.join(sub_dir, "production-overrides.yaml"), 'w') as f:
            f.write(SAMPLE_DEPLOYMENT_MANIFEST.replace("replicas: 5", "replicas: 10"))

        yield temp_dir

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.bigquery
class TestKubernetesIngestionEndToEnd:
    """End-to-end integration tests for Kubernetes manifest ingestion to BigQuery"""

    def test_parser_interfaces_available(self):
        """Test that parser interface contracts are available"""
        assert PARSER_INTERFACES_AVAILABLE, "Parser interface contracts should be available"

    def test_bigquery_client_available(self, bigquery_client):
        """Test that BigQuery client is properly configured"""
        assert bigquery_client is not None
        assert bigquery_client.project == TEST_PROJECT_ID

    def test_temp_dataset_creation(self, temp_dataset):
        """Test that temporary dataset is created successfully"""
        assert temp_dataset is not None
        assert temp_dataset.dataset_id.startswith(TEST_DATASET_PREFIX)
        assert temp_dataset.project == TEST_PROJECT_ID

    @pytest.mark.skipif(not K8S_PARSER_AVAILABLE, reason="Kubernetes parser not implemented yet")
    def test_kubernetes_parser_available(self):
        """Test that Kubernetes parser implementation is available"""
        parser = KubernetesParserImpl()
        assert parser.source_type == SourceType.KUBERNETES

    @pytest.mark.skipif(not BIGQUERY_WRITER_AVAILABLE, reason="BigQuery writer not implemented yet")
    def test_bigquery_writer_available(self):
        """Test that BigQuery writer implementation is available"""
        writer = BigQueryWriterImpl(project_id=TEST_PROJECT_ID)
        assert isinstance(writer, BigQueryWriter)

    @pytest.mark.skipif(not INGESTION_ORCHESTRATOR_AVAILABLE, reason="Ingestion orchestrator not implemented yet")
    def test_ingestion_orchestrator_available(self):
        """Test that ingestion orchestrator is available"""
        orchestrator = KubernetesIngestionOrchestrator(
            project_id=TEST_PROJECT_ID,
            dataset_id="test_dataset"
        )
        assert orchestrator is not None

    @pytest.mark.skipif(
        not all([K8S_PARSER_AVAILABLE, BIGQUERY_WRITER_AVAILABLE, INGESTION_ORCHESTRATOR_AVAILABLE]),
        reason="Required components not implemented yet"
    )
    def test_single_manifest_file_ingestion(self, temp_dataset, temp_manifest_dir):
        """Test ingesting a single Kubernetes manifest file to BigQuery"""
        orchestrator = KubernetesIngestionOrchestrator(
            project_id=TEST_PROJECT_ID,
            dataset_id=temp_dataset.dataset_id
        )

        # Ingest single deployment manifest
        deployment_file = os.path.join(temp_manifest_dir, "deployment.yaml")
        result = orchestrator.ingest_file(deployment_file)

        # Verify ParseResult structure
        assert isinstance(result, ParseResult)
        assert result.files_processed == 1
        assert len(result.chunks) > 0
        assert len(result.errors) == 0
        assert result.processing_duration_ms > 0

        # Verify chunks content
        chunk = result.chunks[0]
        assert isinstance(chunk, ChunkMetadata)
        assert chunk.source_type == SourceType.KUBERNETES
        assert chunk.artifact_id == "k8s://production/Deployment/web-app"
        assert chunk.content_text.strip() != ""
        assert len(chunk.content_hash) == 64  # SHA256 hex digest
        assert chunk.source_uri == deployment_file

        # Verify K8s-specific metadata
        k8s_metadata = chunk.source_metadata
        assert k8s_metadata['kind'] == 'Deployment'
        assert k8s_metadata['api_version'] == 'apps/v1'
        assert k8s_metadata['namespace'] == 'production'
        assert k8s_metadata['resource_name'] == 'web-app'
        assert k8s_metadata['labels']['app'] == 'web-app'
        assert k8s_metadata['labels']['version'] == '2.1.0'

        # Verify data was written to BigQuery
        self._verify_bigquery_data(temp_dataset.dataset_id, result.chunks, expected_count=len(result.chunks))

    @pytest.mark.skipif(
        not all([K8S_PARSER_AVAILABLE, BIGQUERY_WRITER_AVAILABLE, INGESTION_ORCHESTRATOR_AVAILABLE]),
        reason="Required components not implemented yet"
    )
    def test_multi_document_manifest_ingestion(self, temp_dataset, temp_manifest_dir):
        """Test ingesting multi-document YAML manifest to BigQuery"""
        orchestrator = KubernetesIngestionOrchestrator(
            project_id=TEST_PROJECT_ID,
            dataset_id=temp_dataset.dataset_id
        )

        # Ingest multi-document manifest
        multi_file = os.path.join(temp_manifest_dir, "multi-resource.yaml")
        result = orchestrator.ingest_file(multi_file)

        # Should process multiple resources from single file
        assert result.files_processed == 1
        assert len(result.chunks) >= 4  # At least 4 resources (Deployment, Service, ConfigMap, Secret)

        # Verify we got all expected resource types
        kinds = [chunk.source_metadata['kind'] for chunk in result.chunks]
        expected_kinds = ['Deployment', 'Service', 'ConfigMap', 'Secret']
        for expected_kind in expected_kinds:
            assert expected_kind in kinds, f"Missing expected resource kind: {expected_kind}"

        # Verify artifact IDs are unique and correctly formatted
        artifact_ids = [chunk.artifact_id for chunk in result.chunks]
        assert len(set(artifact_ids)) == len(artifact_ids), "Artifact IDs should be unique"

        expected_artifact_patterns = [
            "k8s://production/Deployment/web-app",
            "k8s://production/Service/web-app-service",
            "k8s://production/ConfigMap/app-config",
            "k8s://production/Secret/db-secret"
        ]

        for pattern in expected_artifact_patterns:
            assert any(pattern in artifact_id for artifact_id in artifact_ids), f"Missing expected artifact ID pattern: {pattern}"

        # Verify BigQuery ingestion
        self._verify_bigquery_data(temp_dataset.dataset_id, result.chunks, expected_count=len(result.chunks))

    @pytest.mark.skipif(
        not all([K8S_PARSER_AVAILABLE, BIGQUERY_WRITER_AVAILABLE, INGESTION_ORCHESTRATOR_AVAILABLE]),
        reason="Required components not implemented yet"
    )
    def test_directory_ingestion_with_mixed_files(self, temp_dataset, temp_manifest_dir):
        """Test ingesting entire directory with mixed K8s and non-K8s files"""
        orchestrator = KubernetesIngestionOrchestrator(
            project_id=TEST_PROJECT_ID,
            dataset_id=temp_dataset.dataset_id
        )

        # Ingest entire directory
        result = orchestrator.ingest_directory(temp_manifest_dir)

        # Should process multiple files but skip non-manifest files
        assert result.files_processed >= 5  # Valid manifest files
        assert len(result.chunks) >= 8  # Multiple resources across files

        # Should have some errors for invalid files
        assert len(result.errors) >= 1  # At least invalid.yaml should cause errors

        # Verify chunks from different files
        source_uris = set(chunk.source_uri for chunk in result.chunks)
        expected_files = [
            os.path.join(temp_manifest_dir, "deployment.yaml"),
            os.path.join(temp_manifest_dir, "service.yml"),
            os.path.join(temp_manifest_dir, "configmap.yaml"),
            os.path.join(temp_manifest_dir, "secret.yaml"),
        ]

        for expected_file in expected_files:
            assert any(expected_file in uri for uri in source_uris), f"Missing chunks from file: {expected_file}"

        # Verify BigQuery ingestion
        self._verify_bigquery_data(temp_dataset.dataset_id, result.chunks, expected_count=len(result.chunks))

    @pytest.mark.skipif(
        not all([K8S_PARSER_AVAILABLE, BIGQUERY_WRITER_AVAILABLE, INGESTION_ORCHESTRATOR_AVAILABLE]),
        reason="Required components not implemented yet"
    )
    def test_recursive_directory_ingestion(self, temp_dataset, temp_manifest_dir):
        """Test recursive directory ingestion including subdirectories"""
        orchestrator = KubernetesIngestionOrchestrator(
            project_id=TEST_PROJECT_ID,
            dataset_id=temp_dataset.dataset_id
        )

        # Ingest directory recursively
        result = orchestrator.ingest_directory(temp_manifest_dir, recursive=True)

        # Should find files in subdirectories too
        source_uris = [chunk.source_uri for chunk in result.chunks]
        subdirectory_file = os.path.join(temp_manifest_dir, "overlays", "production", "production-overrides.yaml")

        assert any(subdirectory_file in uri for uri in source_uris), "Should find files in subdirectories"

        # Verify BigQuery ingestion
        self._verify_bigquery_data(temp_dataset.dataset_id, result.chunks, expected_count=len(result.chunks))

    @pytest.mark.skipif(
        not all([K8S_PARSER_AVAILABLE, BIGQUERY_WRITER_AVAILABLE, INGESTION_ORCHESTRATOR_AVAILABLE]),
        reason="Required components not implemented yet"
    )
    def test_error_handling_and_logging(self, temp_dataset, temp_manifest_dir):
        """Test error handling for invalid manifests and BigQuery error logging"""
        orchestrator = KubernetesIngestionOrchestrator(
            project_id=TEST_PROJECT_ID,
            dataset_id=temp_dataset.dataset_id
        )

        # Ingest file with invalid YAML
        invalid_file = os.path.join(temp_manifest_dir, "invalid.yaml")
        result = orchestrator.ingest_file(invalid_file)

        # Should have parsing errors
        assert len(result.errors) > 0

        # Verify error structure
        error = result.errors[0]
        assert isinstance(error, ParseError)
        assert error.source_type == SourceType.KUBERNETES
        assert error.source_uri == invalid_file
        assert error.error_class in [ErrorClass.PARSING, ErrorClass.VALIDATION]
        assert error.error_msg != ""

        # Verify errors are logged to BigQuery
        self._verify_bigquery_errors(temp_dataset.dataset_id, result.errors)

    @pytest.mark.skipif(
        not all([K8S_PARSER_AVAILABLE, BIGQUERY_WRITER_AVAILABLE, INGESTION_ORCHESTRATOR_AVAILABLE]),
        reason="Required components not implemented yet"
    )
    def test_content_hashing_and_deduplication(self, temp_dataset, temp_manifest_dir):
        """Test content hashing and deduplication behavior"""
        orchestrator = KubernetesIngestionOrchestrator(
            project_id=TEST_PROJECT_ID,
            dataset_id=temp_dataset.dataset_id
        )

        # Ingest same file twice
        deployment_file = os.path.join(temp_manifest_dir, "deployment.yaml")

        result1 = orchestrator.ingest_file(deployment_file)
        result2 = orchestrator.ingest_file(deployment_file)

        # Content hashes should be identical
        hash1 = result1.chunks[0].content_hash
        hash2 = result2.chunks[0].content_hash
        assert hash1 == hash2, "Same content should produce same hash"

        # Artifact IDs should be identical
        artifact_id1 = result1.chunks[0].artifact_id
        artifact_id2 = result2.chunks[0].artifact_id
        assert artifact_id1 == artifact_id2, "Same resource should produce same artifact ID"

    @pytest.mark.skipif(
        not all([K8S_PARSER_AVAILABLE, BIGQUERY_WRITER_AVAILABLE, INGESTION_ORCHESTRATOR_AVAILABLE]),
        reason="Required components not implemented yet"
    )
    def test_large_manifest_chunking(self, temp_dataset):
        """Test chunking behavior for very large Kubernetes manifests"""
        orchestrator = KubernetesIngestionOrchestrator(
            project_id=TEST_PROJECT_ID,
            dataset_id=temp_dataset.dataset_id
        )

        # Create a very large ConfigMap with lots of data
        large_data = {}
        for i in range(200):  # Create 200 config entries
            large_data[f"config_key_{i}"] = "x" * 1000  # 1KB per entry

        large_configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "large-config",
                "namespace": "default",
                "labels": {"size": "large"}
            },
            "data": large_data
        }

        large_manifest = yaml.dump(large_configmap)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(large_manifest)
            temp_file = f.name

        try:
            result = orchestrator.ingest_file(temp_file)

            # Should generate multiple chunks for large content
            assert len(result.chunks) > 1, "Large manifest should be split into multiple chunks"

            # All chunks should have same artifact_id but different content
            artifact_ids = [chunk.artifact_id for chunk in result.chunks]
            assert all(aid == "k8s://default/ConfigMap/large-config" for aid in artifact_ids), "All chunks should have same artifact ID"

            # Content should be different across chunks
            content_texts = [chunk.content_text for chunk in result.chunks]
            assert len(set(content_texts)) > 1, "Chunks should have different content"

            # Verify BigQuery ingestion
            self._verify_bigquery_data(temp_dataset.dataset_id, result.chunks, expected_count=len(result.chunks))

        finally:
            os.unlink(temp_file)

    @pytest.mark.skipif(
        not all([K8S_PARSER_AVAILABLE, BIGQUERY_WRITER_AVAILABLE, INGESTION_ORCHESTRATOR_AVAILABLE]),
        reason="Required components not implemented yet"
    )
    def test_ingestion_run_logging(self, temp_dataset, temp_manifest_dir):
        """Test that ingestion runs are properly logged with metadata"""
        orchestrator = KubernetesIngestionOrchestrator(
            project_id=TEST_PROJECT_ID,
            dataset_id=temp_dataset.dataset_id
        )

        # Perform ingestion
        result = orchestrator.ingest_directory(temp_manifest_dir)

        # Verify run logging
        run_info = {
            "source_type": "kubernetes",
            "source_path": temp_manifest_dir,
            "files_processed": result.files_processed,
            "chunks_generated": len(result.chunks),
            "errors_encountered": len(result.errors),
            "processing_duration_ms": result.processing_duration_ms,
            "parser_version": "1.0.0",
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Log the run and get run_id
        writer = BigQueryWriterImpl(project_id=TEST_PROJECT_ID)
        run_id = writer.log_ingestion_run(run_info, f"{temp_dataset.dataset_id}.ingestion_runs")

        assert run_id is not None and run_id != ""

        # Verify run was logged to BigQuery
        self._verify_ingestion_run_logged(temp_dataset.dataset_id, run_id, run_info)

    def _verify_bigquery_data(self, dataset_id: str, chunks: List[ChunkMetadata], expected_count: int):
        """Verify that chunks were properly written to BigQuery source_metadata table"""
        if not BIGQUERY_WRITER_AVAILABLE:
            pytest.skip("BigQuery writer not available for verification")

        client = bigquery.Client(project=TEST_PROJECT_ID)

        # Query the source_metadata table
        query = f"""
        SELECT
            source_type,
            artifact_id,
            content_hash,
            source_uri,
            source_metadata,
            collected_at
        FROM `{TEST_PROJECT_ID}.{dataset_id}.source_metadata`
        WHERE source_type = 'kubernetes'
        ORDER BY collected_at DESC
        """

        query_job = client.query(query)
        results = list(query_job.result())

        # Verify row count
        assert len(results) >= expected_count, f"Expected at least {expected_count} rows, got {len(results)}"

        # Verify data integrity
        for row in results:
            assert row.source_type == "kubernetes"
            assert row.artifact_id.startswith("k8s://")
            assert len(row.content_hash) == 64  # SHA256 hex digest
            assert row.source_uri != ""
            assert row.source_metadata is not None
            assert row.collected_at is not None

            # Parse and verify source_metadata JSON
            metadata = json.loads(row.source_metadata) if isinstance(row.source_metadata, str) else row.source_metadata
            required_fields = ['kind', 'api_version', 'namespace', 'resource_name', 'labels', 'annotations']
            for field in required_fields:
                assert field in metadata, f"Missing required metadata field: {field}"

    def _verify_bigquery_errors(self, dataset_id: str, errors: List[ParseError]):
        """Verify that parsing errors were properly logged to BigQuery"""
        if not BIGQUERY_WRITER_AVAILABLE or not errors:
            return

        client = bigquery.Client(project=TEST_PROJECT_ID)

        # Query the source_metadata_errors table
        query = f"""
        SELECT
            source_type,
            source_uri,
            error_class,
            error_msg,
            sample_text,
            collected_at
        FROM `{TEST_PROJECT_ID}.{dataset_id}.source_metadata_errors`
        WHERE source_type = 'kubernetes'
        ORDER BY collected_at DESC
        """

        query_job = client.query(query)
        results = list(query_job.result())

        # Verify error logging
        assert len(results) >= len(errors), f"Expected at least {len(errors)} error rows"

        for row in results:
            assert row.source_type == "kubernetes"
            assert row.error_class in ["parsing", "validation", "ingestion"]
            assert row.error_msg != ""
            assert row.collected_at is not None

    def _verify_ingestion_run_logged(self, dataset_id: str, run_id: str, run_info: Dict[str, Any]):
        """Verify that ingestion run was properly logged"""
        if not BIGQUERY_WRITER_AVAILABLE:
            return

        client = bigquery.Client(project=TEST_PROJECT_ID)

        # Query the ingestion_runs table
        query = f"""
        SELECT
            run_id,
            source_type,
            source_path,
            files_processed,
            chunks_generated,
            errors_encountered,
            processing_duration_ms,
            parser_version,
            ingestion_timestamp
        FROM `{TEST_PROJECT_ID}.{dataset_id}.ingestion_runs`
        WHERE run_id = @run_id
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id", "STRING", run_id)
            ]
        )

        query_job = client.query(query, job_config=job_config)
        results = list(query_job.result())

        assert len(results) == 1, f"Expected exactly 1 run log entry, got {len(results)}"

        row = results[0]
        assert row.run_id == run_id
        assert row.source_type == run_info["source_type"]
        assert row.files_processed == run_info["files_processed"]
        assert row.chunks_generated == run_info["chunks_generated"]


@pytest.mark.integration
@pytest.mark.bigquery
class TestKubernetesIngestionTDDFailures:
    """Tests that should fail until implementation is complete (TDD verification)"""

    def test_kubernetes_parser_not_implemented(self):
        """Verify that Kubernetes parser is not implemented yet (TDD requirement)"""
        assert not K8S_PARSER_AVAILABLE, (
            "KubernetesParserImpl should not exist yet (TDD requirement). "
            "This test should fail once the parser is implemented."
        )

    def test_bigquery_writer_not_implemented(self):
        """Verify that BigQuery writer is not implemented yet (TDD requirement)"""
        assert not BIGQUERY_WRITER_AVAILABLE, (
            "BigQueryWriterImpl should not exist yet (TDD requirement). "
            "This test should fail once the writer is implemented."
        )

    def test_ingestion_orchestrator_not_implemented(self):
        """Verify that ingestion orchestrator is not implemented yet (TDD requirement)"""
        assert not INGESTION_ORCHESTRATOR_AVAILABLE, (
            "KubernetesIngestionOrchestrator should not exist yet (TDD requirement). "
            "This test should fail once the orchestrator is implemented."
        )

    def test_integration_will_fail_without_implementation(self):
        """Verify that the integration tests will fail without implementation"""
        missing_components = []

        if not K8S_PARSER_AVAILABLE:
            missing_components.append("KubernetesParserImpl")
        if not BIGQUERY_WRITER_AVAILABLE:
            missing_components.append("BigQueryWriterImpl")
        if not INGESTION_ORCHESTRATOR_AVAILABLE:
            missing_components.append("KubernetesIngestionOrchestrator")

        assert len(missing_components) > 0, (
            f"Missing implementation components: {missing_components}. "
            "Integration tests should fail until these are implemented."
        )

    def test_sample_manifests_valid(self):
        """Verify that sample manifests are valid YAML/JSON (these should always pass)"""
        # These tests should pass to ensure our test data is valid

        # Test individual manifests
        for manifest_name, manifest_content in [
            ("deployment", SAMPLE_DEPLOYMENT_MANIFEST),
            ("service", SAMPLE_SERVICE_MANIFEST),
            ("configmap", SAMPLE_CONFIGMAP_MANIFEST),
            ("secret", SAMPLE_SECRET_MANIFEST),
        ]:
            try:
                parsed = yaml.safe_load(manifest_content)
                assert parsed is not None, f"{manifest_name} manifest should parse as valid YAML"
                assert 'apiVersion' in parsed, f"{manifest_name} should have apiVersion"
                assert 'kind' in parsed, f"{manifest_name} should have kind"
                assert 'metadata' in parsed, f"{manifest_name} should have metadata"
                assert 'name' in parsed['metadata'], f"{manifest_name} should have metadata.name"
            except yaml.YAMLError as e:
                pytest.fail(f"Sample {manifest_name} manifest is invalid YAML: {e}")

        # Test multi-document manifest
        try:
            documents = list(yaml.safe_load_all(MULTI_DOCUMENT_MANIFEST))
            assert len(documents) == 4, "Multi-document manifest should contain 4 documents"
            for i, doc in enumerate(documents):
                assert doc is not None, f"Document {i} should not be None"
                assert 'apiVersion' in doc, f"Document {i} should have apiVersion"
        except yaml.YAMLError as e:
            pytest.fail(f"Multi-document manifest is invalid YAML: {e}")

        # Test that invalid manifest is indeed invalid
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(INVALID_YAML_MANIFEST)


if __name__ == "__main__":
    # Run the integration tests
    pytest.main([__file__, "-v", "-m", "integration"])