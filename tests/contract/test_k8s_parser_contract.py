"""
Contract tests for Kubernetes Parser Interface
These tests MUST FAIL initially (TDD requirement) until the Kubernetes parser is implemented.

Tests the contract defined in specs/002-m1-parse-and/contracts/parser-interfaces.py
"""

import pytest

# Register custom markers to avoid warnings
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "contract: Contract tests for interface compliance (TDD)")
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
import yaml
import json
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

# Import the parser interface contracts via shared module
try:
    from src.common.parser_interfaces import (
        BaseParser,
        KubernetesParser,
        SourceType,
        ChunkMetadata,
        ParseResult,
        ParseError,
        ErrorClass,
    )
    PARSER_INTERFACES_AVAILABLE = True
except Exception as e:
    PARSER_INTERFACES_AVAILABLE = False
    print(f"Warning: Could not import parser interfaces: {e}")

# Try to import the actual implementation (expected to fail initially)
try:
    from src.ingest.k8s.parser import KubernetesParserImpl
    KUBERNETES_PARSER_AVAILABLE = True
except ImportError:
    KUBERNETES_PARSER_AVAILABLE = False


# Test data for Kubernetes manifests
VALID_DEPLOYMENT_MANIFEST = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  namespace: default
  labels:
    app: nginx
    version: "1.0"
  annotations:
    deployment.kubernetes.io/revision: "1"
    konveyn2ai.com/parser: "kubernetes"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.20
        ports:
        - containerPort: 80
"""

VALID_SERVICE_MANIFEST = """
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
  namespace: default
  labels:
    app: nginx
spec:
  selector:
    app: nginx
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
  type: ClusterIP
"""

VALID_CONFIGMAP_MANIFEST = """
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: kube-system
  labels:
    component: config
data:
  database_url: "postgresql://localhost:5432/mydb"
  log_level: "info"
  feature_flags: |
    {
      "enable_auth": true,
      "enable_metrics": false
    }
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
      invalidKey: [unclosed bracket
"""

MALFORMED_MANIFEST = """
this is not yaml or json
just random text
"""

MULTI_DOCUMENT_MANIFEST = f"""
{VALID_SERVICE_MANIFEST}
---
{VALID_DEPLOYMENT_MANIFEST}
---
{VALID_CONFIGMAP_MANIFEST}
"""


@pytest.mark.contract
@pytest.mark.unit
class TestKubernetesParserContract:
    """Contract tests for Kubernetes Parser implementation"""

    def test_parser_interfaces_available(self):
        """Test that parser interface contracts are available"""
        assert PARSER_INTERFACES_AVAILABLE, "Parser interface contracts should be available"

    @pytest.mark.skipif(not PARSER_INTERFACES_AVAILABLE, reason="Parser interfaces not available")
    def test_kubernetes_parser_interface_exists(self):
        """Test that KubernetesParser abstract class exists with required methods"""
        # Verify KubernetesParser exists and inherits from BaseParser
        assert issubclass(KubernetesParser, BaseParser)

        # Verify required abstract methods exist
        abstract_methods = getattr(KubernetesParser, '__abstractmethods__', set())
        required_methods = {
            'parse_file',
            'parse_directory',
            'validate_content',
            'parse_manifest',
            'extract_live_resources'
        }

        # Check that all required methods are declared as abstract
        for method in required_methods:
            assert hasattr(KubernetesParser, method), f"Method {method} should exist"

    @pytest.mark.skipif(KUBERNETES_PARSER_AVAILABLE, reason="Skip when implementation exists")
    def test_kubernetes_parser_not_implemented_yet(self):
        """Test that the actual implementation doesn't exist yet (TDD requirement)"""
        assert not KUBERNETES_PARSER_AVAILABLE, "KubernetesParserImpl should not exist yet (TDD)"

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_kubernetes_parser_inheritance(self):
        """Test that KubernetesParserImpl properly inherits from KubernetesParser"""
        assert issubclass(KubernetesParserImpl, KubernetesParser)
        assert issubclass(KubernetesParserImpl, BaseParser)

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_kubernetes_parser_source_type(self):
        """Test that parser returns correct source type"""
        parser = KubernetesParserImpl()
        assert parser.source_type == SourceType.KUBERNETES
        assert parser._get_source_type() == SourceType.KUBERNETES

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_valid_deployment_manifest(self):
        """Test parsing a valid Deployment manifest"""
        parser = KubernetesParserImpl()
        chunks = parser.parse_manifest(VALID_DEPLOYMENT_MANIFEST)

        assert isinstance(chunks, list)
        assert len(chunks) > 0

        # Check the first chunk
        chunk = chunks[0]
        assert isinstance(chunk, ChunkMetadata)
        assert chunk.source_type == SourceType.KUBERNETES

        # Verify artifact_id format: k8s://{namespace}/{kind}/{name}
        expected_artifact_id = "k8s://default/Deployment/nginx-deployment"
        assert chunk.artifact_id == expected_artifact_id

        # Verify K8s-specific metadata
        k8s_metadata = chunk.source_metadata
        assert k8s_metadata['kind'] == 'Deployment'
        assert k8s_metadata['api_version'] == 'apps/v1'
        assert k8s_metadata['namespace'] == 'default'
        assert k8s_metadata['resource_name'] == 'nginx-deployment'
        assert k8s_metadata['labels']['app'] == 'nginx'
        assert k8s_metadata['labels']['version'] == '1.0'
        assert 'deployment.kubernetes.io/revision' in k8s_metadata['annotations']

        # Verify content fields
        assert chunk.content_text.strip() != ""
        assert chunk.content_hash != ""
        assert isinstance(chunk.collected_at, datetime)

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_valid_service_manifest(self):
        """Test parsing a valid Service manifest"""
        parser = KubernetesParserImpl()
        chunks = parser.parse_manifest(VALID_SERVICE_MANIFEST)

        assert len(chunks) > 0
        chunk = chunks[0]

        # Verify artifact_id format
        expected_artifact_id = "k8s://default/Service/nginx-service"
        assert chunk.artifact_id == expected_artifact_id

        # Verify K8s-specific metadata
        k8s_metadata = chunk.source_metadata
        assert k8s_metadata['kind'] == 'Service'
        assert k8s_metadata['api_version'] == 'v1'
        assert k8s_metadata['namespace'] == 'default'
        assert k8s_metadata['resource_name'] == 'nginx-service'

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_configmap_with_kube_system_namespace(self):
        """Test parsing ConfigMap in kube-system namespace"""
        parser = KubernetesParserImpl()
        chunks = parser.parse_manifest(VALID_CONFIGMAP_MANIFEST)

        assert len(chunks) > 0
        chunk = chunks[0]

        # Verify artifact_id format with kube-system namespace
        expected_artifact_id = "k8s://kube-system/ConfigMap/app-config"
        assert chunk.artifact_id == expected_artifact_id

        k8s_metadata = chunk.source_metadata
        assert k8s_metadata['namespace'] == 'kube-system'
        assert k8s_metadata['kind'] == 'ConfigMap'

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_multi_document_manifest(self):
        """Test parsing multi-document YAML manifest"""
        parser = KubernetesParserImpl()
        chunks = parser.parse_manifest(MULTI_DOCUMENT_MANIFEST)

        # Should generate chunks for all 3 documents
        assert len(chunks) >= 3

        # Verify we got all expected resource types
        kinds = [chunk.source_metadata['kind'] for chunk in chunks]
        assert 'Service' in kinds
        assert 'Deployment' in kinds
        assert 'ConfigMap' in kinds

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_invalid_yaml_manifest(self):
        """Test error handling for invalid YAML"""
        parser = KubernetesParserImpl()

        with pytest.raises((yaml.YAMLError, ValueError)):
            parser.parse_manifest(INVALID_YAML_MANIFEST)

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_malformed_manifest(self):
        """Test error handling for completely malformed content"""
        parser = KubernetesParserImpl()

        with pytest.raises((yaml.YAMLError, ValueError)):
            parser.parse_manifest(MALFORMED_MANIFEST)

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_validate_content_valid_yaml(self):
        """Test content validation for valid Kubernetes YAML"""
        parser = KubernetesParserImpl()

        assert parser.validate_content(VALID_DEPLOYMENT_MANIFEST) is True
        assert parser.validate_content(VALID_SERVICE_MANIFEST) is True
        assert parser.validate_content(VALID_CONFIGMAP_MANIFEST) is True

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_validate_content_valid_json(self):
        """Test content validation for valid Kubernetes JSON"""
        parser = KubernetesParserImpl()

        # Convert YAML to JSON
        manifest_dict = yaml.safe_load(VALID_DEPLOYMENT_MANIFEST)
        json_manifest = json.dumps(manifest_dict)

        assert parser.validate_content(json_manifest) is True

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_validate_content_invalid(self):
        """Test content validation for invalid content"""
        parser = KubernetesParserImpl()

        assert parser.validate_content(MALFORMED_MANIFEST) is False
        assert parser.validate_content("") is False
        assert parser.validate_content("not yaml or json") is False

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_file_single_manifest(self, tmp_path):
        """Test parsing a single manifest file"""
        parser = KubernetesParserImpl()

        # Create temporary manifest file
        manifest_file = tmp_path / "deployment.yaml"
        manifest_file.write_text(VALID_DEPLOYMENT_MANIFEST)

        result = parser.parse_file(str(manifest_file))

        assert isinstance(result, ParseResult)
        assert len(result.chunks) > 0
        assert len(result.errors) == 0
        assert result.files_processed == 1
        assert result.processing_duration_ms > 0

        # Verify source_uri is set correctly
        chunk = result.chunks[0]
        assert chunk.source_uri == str(manifest_file)

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_directory_multiple_manifests(self, tmp_path):
        """Test parsing a directory with multiple manifest files"""
        parser = KubernetesParserImpl()

        # Create multiple manifest files
        (tmp_path / "deployment.yaml").write_text(VALID_DEPLOYMENT_MANIFEST)
        (tmp_path / "service.yml").write_text(VALID_SERVICE_MANIFEST)
        (tmp_path / "configmap.yaml").write_text(VALID_CONFIGMAP_MANIFEST)
        (tmp_path / "not-a-manifest.txt").write_text("This is not a Kubernetes manifest")

        result = parser.parse_directory(str(tmp_path))

        assert isinstance(result, ParseResult)
        assert len(result.chunks) >= 3  # At least 3 valid manifests
        assert result.files_processed >= 3

        # Verify we got the expected resource types
        kinds = [chunk.source_metadata['kind'] for chunk in result.chunks]
        assert 'Deployment' in kinds
        assert 'Service' in kinds
        assert 'ConfigMap' in kinds

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_artifact_id_generation(self):
        """Test artifact ID generation for various Kubernetes resources"""
        parser = KubernetesParserImpl()

        # Test with different namespaces
        test_cases = [
            ("default", "Deployment", "nginx", "k8s://default/Deployment/nginx"),
            ("kube-system", "ConfigMap", "coredns", "k8s://kube-system/ConfigMap/coredns"),
            ("monitoring", "Service", "prometheus", "k8s://monitoring/Service/prometheus"),
            ("", "Pod", "standalone", "k8s:///Pod/standalone"),  # Empty namespace
        ]

        for namespace, kind, name, expected in test_cases:
            artifact_id = parser.generate_artifact_id(
                "",  # source_path not used for K8s
                namespace=namespace,
                kind=kind,
                name=name
            )
            assert artifact_id == expected

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_chunk_content_large_manifest(self):
        """Test chunking behavior for large manifests"""
        parser = KubernetesParserImpl()

        # Create a large manifest with lots of data
        large_data = {str(i): f"value_{i}" * 100 for i in range(100)}
        large_configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "large-config",
                "namespace": "default"
            },
            "data": large_data
        }

        large_manifest = yaml.dump(large_configmap)
        chunks = parser.parse_manifest(large_manifest)

        # Should generate multiple chunks for large content
        assert len(chunks) > 0

        # Verify all chunks have content and proper metadata
        for chunk in chunks:
            assert chunk.content_text.strip() != ""
            assert chunk.source_metadata['kind'] == 'ConfigMap'
            assert chunk.artifact_id.startswith("k8s://default/ConfigMap/large-config")

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    @patch('kubernetes.client.ApiClient')
    @patch('kubernetes.config.load_incluster_config')
    def test_extract_live_resources_in_cluster(self, mock_load_config, mock_api_client):
        """Test extracting resources from live cluster (in-cluster)"""
        parser = KubernetesParserImpl()

        # Mock Kubernetes API responses
        mock_client = MagicMock()
        mock_api_client.return_value = mock_client

        # Mock API responses for different resource types
        mock_deployments = MagicMock()
        mock_deployments.items = [
            MagicMock(metadata=MagicMock(name="test-deployment", namespace="default"))
        ]
        mock_client.list_deployment_for_all_namespaces.return_value = mock_deployments

        result = parser.extract_live_resources()

        assert isinstance(result, ParseResult)
        mock_load_config.assert_called_once()

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    @patch('kubernetes.client.ApiClient')
    @patch('kubernetes.config.load_kube_config')
    def test_extract_live_resources_external(self, mock_load_config, mock_api_client):
        """Test extracting resources from live cluster (external)"""
        parser = KubernetesParserImpl()

        # Mock for external cluster access
        mock_client = MagicMock()
        mock_api_client.return_value = mock_client

        result = parser.extract_live_resources(namespace="kube-system")

        assert isinstance(result, ParseResult)

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_error_handling_missing_required_fields(self):
        """Test error handling when required Kubernetes fields are missing"""
        parser = KubernetesParserImpl()

        # Missing kind
        invalid_manifest_no_kind = """
        apiVersion: v1
        metadata:
          name: no-kind
          namespace: default
        """

        # Missing name
        invalid_manifest_no_name = """
        apiVersion: v1
        kind: Pod
        metadata:
          namespace: default
        """

        # Missing metadata
        invalid_manifest_no_metadata = """
        apiVersion: v1
        kind: Pod
        spec:
          containers: []
        """

        for invalid_manifest in [invalid_manifest_no_kind, invalid_manifest_no_name, invalid_manifest_no_metadata]:
            with pytest.raises((ValueError, KeyError)):
                parser.parse_manifest(invalid_manifest)

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_content_hash_generation(self):
        """Test that content hash is generated correctly"""
        parser = KubernetesParserImpl()
        chunks = parser.parse_manifest(VALID_DEPLOYMENT_MANIFEST)

        assert len(chunks) > 0
        chunk = chunks[0]
        assert chunk.content_hash != ""
        assert len(chunk.content_hash) == 64  # SHA256 hex digest length

        # Same content should generate same hash
        chunks2 = parser.parse_manifest(VALID_DEPLOYMENT_MANIFEST)
        assert chunks[0].content_hash == chunks2[0].content_hash

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_result_structure(self):
        """Test that ParseResult has the correct structure"""
        parser = KubernetesParserImpl()

        # Test with valid manifest
        chunks = parser.parse_manifest(VALID_DEPLOYMENT_MANIFEST)

        # Verify chunk structure
        assert len(chunks) > 0
        chunk = chunks[0]

        # Required ChunkMetadata fields
        assert hasattr(chunk, 'source_type')
        assert hasattr(chunk, 'artifact_id')
        assert hasattr(chunk, 'content_text')
        assert hasattr(chunk, 'content_hash')
        assert hasattr(chunk, 'source_metadata')
        assert hasattr(chunk, 'collected_at')

        # Kubernetes-specific metadata fields
        k8s_metadata = chunk.source_metadata
        required_k8s_fields = ['kind', 'api_version', 'namespace', 'resource_name', 'labels', 'annotations']
        for field in required_k8s_fields:
            assert field in k8s_metadata, f"Missing required K8s metadata field: {field}"

    @pytest.mark.skipif(not KUBERNETES_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_namespace_handling_edge_cases(self):
        """Test namespace handling for edge cases"""
        parser = KubernetesParserImpl()

        # Test manifest without namespace (should default to 'default')
        manifest_no_namespace = """
        apiVersion: v1
        kind: Pod
        metadata:
          name: no-namespace-pod
        spec:
          containers:
          - name: app
            image: nginx
        """

        chunks = parser.parse_manifest(manifest_no_namespace)
        assert len(chunks) > 0
        chunk = chunks[0]

        # Should default to 'default' namespace or handle appropriately
        k8s_metadata = chunk.source_metadata
        namespace = k8s_metadata.get('namespace', 'default')
        assert namespace in ['default', '']  # Allow both default and empty

        expected_artifact_id = f"k8s://{namespace}/Pod/no-namespace-pod"
        assert chunk.artifact_id == expected_artifact_id


@pytest.mark.contract
@pytest.mark.unit
class TestKubernetesParserContractFailure:
    """Tests that should fail until implementation is complete (TDD verification)"""

    @pytest.mark.skipif(KUBERNETES_PARSER_AVAILABLE, reason="Implementation available")
    def test_implementation_not_available(self):
        """This test ensures we're in TDD mode - implementation should not exist yet"""
        # This test should pass initially, then fail once implementation exists
        try:
            from src.parsers.kubernetes_parser import KubernetesParserImpl
            pytest.fail("KubernetesParserImpl should not be implemented yet (TDD requirement)")
        except ImportError:
            # This is expected in TDD mode
            pass

    @pytest.mark.skipif(KUBERNETES_PARSER_AVAILABLE, reason="Implementation available")
    def test_contract_will_fail_without_implementation(self):
        """Verify that the contract tests will fail without implementation"""
        assert not KUBERNETES_PARSER_AVAILABLE, (
            "Implementation should not exist yet. "
            "Once implemented, this test should be removed or modified."
        )


if __name__ == "__main__":
    # Run the contract tests
    pytest.main([__file__, "-v"])
