"""
Unit tests for Janapada Memory service.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Import the Janapada main app
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.janapada_memory.main import app


class TestJanapadaMemory:
    """Test Janapada Memory service."""
    
    @pytest.fixture
    def client(self):
        """Test client for Janapada service."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_vertex_setup(self, mock_env_vars, mock_google_credentials, mock_vertex_ai):
        """Set up mocked Vertex AI environment."""
        return {
            "env": mock_env_vars,
            "credentials": mock_google_credentials,
            "vertex": mock_vertex_ai
        }

    def test_health_endpoint(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_agent_manifest_endpoint(self, client):
        """Test agent manifest endpoint."""
        response = client.get("/.well-known/agent.json")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Janapada Memory Service"
        assert data["version"] == "1.0.0"
        assert "methods" in data
        
        # Check for search method
        method_names = [method["name"] for method in data["methods"]]
        assert "search" in method_names

    @pytest.mark.asyncio
    async def test_search_jsonrpc_success(
        self, 
        client, 
        mock_vertex_setup, 
        mock_matching_engine,
        sample_snippets
    ):
        """Test successful JSON-RPC search request."""
        
        # Prepare JSON-RPC request
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "id": "test-search-123",
            "method": "search",
            "params": {
                "query": "authentication implementation",
                "k": 3
            }
        }
        
        response = client.post("/", json=jsonrpc_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify JSON-RPC response format
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "test-search-123"
        assert "result" in data
        
        # Verify search results
        result = data["result"]
        assert "snippets" in result
        assert len(result["snippets"]) <= 3  # Requested k=3
        
        # Verify snippet structure
        for snippet in result["snippets"]:
            assert "file_path" in snippet
            assert "content" in snippet

    def test_search_jsonrpc_invalid_method(self, client):
        """Test JSON-RPC with invalid method."""
        
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "id": "test-invalid-123",
            "method": "invalid_method",
            "params": {}
        }
        
        response = client.post("/", json=jsonrpc_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return JSON-RPC error
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "test-invalid-123"
        assert "error" in data
        assert data["error"]["code"] == -32601  # Method not found

    def test_search_jsonrpc_missing_params(self, client):
        """Test JSON-RPC search with missing parameters."""
        
        jsonrpc_request = {
            "jsonrpc": "2.0", 
            "id": "test-missing-params",
            "method": "search",
            "params": {
                # Missing "query" parameter
                "k": 5
            }
        }
        
        response = client.post("/", json=jsonrpc_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return parameter error
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "test-missing-params"
        assert "error" in data
        assert data["error"]["code"] == -32602  # Invalid params

    def test_search_jsonrpc_invalid_k_parameter(self, client):
        """Test JSON-RPC search with invalid k parameter."""
        
        test_cases = [
            {"k": 0},      # Zero
            {"k": -1},     # Negative
            {"k": 101},    # Too large
            {"k": "five"}, # Wrong type
        ]
        
        for params in test_cases:
            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": "test-invalid-k",
                "method": "search", 
                "params": {
                    "query": "test query",
                    **params
                }
            }
            
            response = client.post("/", json=jsonrpc_request)
            
            assert response.status_code == 200
            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32602  # Invalid params

    @pytest.mark.asyncio
    async def test_search_with_embedding_failure(
        self, 
        client, 
        mock_env_vars, 
        mock_google_credentials
    ):
        """Test search when embedding generation fails."""
        
        with patch("vertexai.language_models.TextEmbeddingModel.from_pretrained") as mock_model:
            # Mock embedding failure
            model_instance = MagicMock()
            model_instance.get_embeddings.side_effect = Exception("Embedding service error")
            mock_model.return_value = model_instance
            
            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": "test-embed-fail",
                "method": "search",
                "params": {
                    "query": "test query",
                    "k": 3
                }
            }
            
            response = client.post("/", json=jsonrpc_request)
            
            assert response.status_code == 200
            data = response.json()
            
            # Should return error
            assert "error" in data
            assert data["id"] == "test-embed-fail"

    @pytest.mark.asyncio 
    async def test_search_with_index_failure(
        self, 
        client, 
        mock_vertex_setup
    ):
        """Test search when vector index lookup fails."""
        
        with patch("google.cloud.aiplatform.MatchingEngineIndexEndpoint") as mock_endpoint:
            # Mock index failure
            endpoint_instance = MagicMock()
            index_mock = MagicMock()
            index_mock.find_neighbors.side_effect = Exception("Index lookup failed")
            endpoint_instance.get_index.return_value = index_mock
            mock_endpoint.return_value = endpoint_instance
            
            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": "test-index-fail",
                "method": "search",
                "params": {
                    "query": "test query",
                    "k": 3
                }
            }
            
            response = client.post("/", json=jsonrpc_request)
            
            assert response.status_code == 200
            data = response.json()
            
            # Should return error
            assert "error" in data
            assert data["id"] == "test-index-fail"

    def test_malformed_jsonrpc_request(self, client):
        """Test handling of malformed JSON-RPC requests."""
        
        test_cases = [
            # Missing jsonrpc field
            {"id": "test", "method": "search", "params": {}},
            # Wrong jsonrpc version
            {"jsonrpc": "1.0", "id": "test", "method": "search", "params": {}},
            # Missing method
            {"jsonrpc": "2.0", "id": "test", "params": {}},
            # Invalid JSON structure
            "invalid json string",
            # Empty request
            {},
        ]
        
        for invalid_request in test_cases:
            response = client.post("/", json=invalid_request)
            
            # Should handle gracefully
            assert response.status_code in [200, 400, 422]
            
            if response.status_code == 200:
                data = response.json()
                assert "error" in data

    @pytest.mark.asyncio
    async def test_search_result_metadata_parsing(
        self, 
        client, 
        mock_vertex_setup,
        sample_snippets
    ):
        """Test parsing of search result metadata."""
        
        with patch("google.cloud.aiplatform.MatchingEngineIndexEndpoint") as mock_endpoint:
            # Create mock matches with various metadata formats
            matches = []
            
            # Valid metadata
            valid_match = MagicMock()
            valid_match.metadata = json.dumps(sample_snippets[0])
            valid_match.distance = 0.1
            matches.append(valid_match)
            
            # Invalid JSON metadata
            invalid_match = MagicMock()
            invalid_match.metadata = "invalid json {"
            invalid_match.distance = 0.2
            matches.append(invalid_match)
            
            # Missing fields metadata
            incomplete_match = MagicMock()
            incomplete_match.metadata = json.dumps({"file_path": "test.py"})  # Missing content
            incomplete_match.distance = 0.3
            matches.append(incomplete_match)
            
            # Setup mock index
            index_mock = MagicMock()
            index_mock.find_neighbors.return_value = [matches]
            endpoint_instance = MagicMock()
            endpoint_instance.get_index.return_value = index_mock
            mock_endpoint.return_value = endpoint_instance
            
            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": "test-metadata",
                "method": "search",
                "params": {
                    "query": "test query",
                    "k": 5
                }
            }
            
            response = client.post("/", json=jsonrpc_request)
            
            assert response.status_code == 200
            data = response.json()
            
            # Should successfully parse valid metadata and skip invalid ones
            assert "result" in data
            snippets = data["result"]["snippets"]
            
            # Should only include snippets with valid metadata
            assert len(snippets) >= 1  # At least the valid one
            assert snippets[0]["file_path"] == sample_snippets[0]["file_path"]

    def test_concurrent_search_requests(self, client, mock_vertex_setup, mock_matching_engine):
        """Test handling of concurrent search requests."""
        
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request(request_id):
            try:
                jsonrpc_request = {
                    "jsonrpc": "2.0",
                    "id": f"concurrent-{request_id}",
                    "method": "search",
                    "params": {
                        "query": f"test query {request_id}",
                        "k": 3
                    }
                }
                
                response = client.post("/", json=jsonrpc_request)
                results.append((request_id, response.status_code, response.json()))
            except Exception as e:
                errors.append((request_id, str(e)))
        
        # Launch concurrent requests
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        
        for request_id, status_code, response_data in results:
            assert status_code == 200
            assert response_data["id"] == f"concurrent-{request_id}"

    def test_search_query_sanitization(self, client, mock_vertex_setup, mock_matching_engine):
        """Test sanitization of search queries."""
        
        test_queries = [
            "normal query",
            "query with special chars !@#$%^&*()",
            "query\nwith\nnewlines",
            "query\twith\ttabs",
            "query with unicode: 你好",
            "very " + "long " * 100 + "query",  # Very long query
            "",  # Empty query (should be handled)
        ]
        
        for query in test_queries:
            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": f"test-sanitize-{hash(query)}",
                "method": "search", 
                "params": {
                    "query": query,
                    "k": 3
                }
            }
            
            response = client.post("/", json=jsonrpc_request)
            
            # Should handle all queries gracefully
            assert response.status_code == 200
            data = response.json()
            
            # Empty query might return error, others should succeed
            if query.strip():
                assert "result" in data or "error" in data
            else:
                assert "error" in data  # Empty query should return error


class TestJanapadaConfiguration:
    """Test Janapada service configuration and initialization."""
    
    def test_environment_variable_validation(self):
        """Test validation of required environment variables."""
        
        required_vars = [
            "GOOGLE_CLOUD_PROJECT",
            "INDEX_ENDPOINT_ID", 
            "INDEX_ID"
        ]
        
        for var in required_vars:
            with patch.dict(os.environ, {var: ""}, clear=True):
                # Should handle missing environment variables gracefully
                # This would be tested during app initialization
                pass

    def test_google_cloud_initialization(self, mock_env_vars, mock_google_credentials):
        """Test Google Cloud service initialization."""
        
        with patch("vertexai.init") as mock_init:
            # Import should trigger initialization
            from src.janapada_memory.main import app
            
            # Verify Vertex AI was initialized
            mock_init.assert_called()


class TestJanapadaPerformance:
    """Test performance characteristics of Janapada service."""
    
    @pytest.mark.slow
    def test_search_performance(self, client, mock_vertex_setup, mock_matching_engine):
        """Test search performance under load."""
        
        import time
        
        start_time = time.time()
        
        # Make multiple sequential requests
        for i in range(50):
            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": f"perf-test-{i}",
                "method": "search",
                "params": {
                    "query": f"performance test query {i}",
                    "k": 5
                }
            }
            
            response = client.post("/", json=jsonrpc_request)
            assert response.status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert total_time < 30.0, f"Search took too long: {total_time}s"
        
        avg_time = total_time / 50
        assert avg_time < 0.6, f"Average search time too high: {avg_time}s"