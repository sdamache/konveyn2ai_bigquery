"""
Unit tests for Svami Orchestrator main module.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
import httpx

# Import the Svami main app
import sys
import os

# Add the project root and the specific component directory to Python path
project_root = os.path.join(os.path.dirname(__file__), "../../..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))
sys.path.append(os.path.join(project_root, "src/svami-orchestrator"))
sys.path.append(os.path.join(project_root, "src/common"))

from main import app
from models import QueryRequest


class TestSvamiOrchestrator:
    """Test Svami Orchestrator service."""

    @pytest.fixture
    def client(self):
        """Test client for Svami service."""
        return TestClient(app)

    @pytest.fixture
    def mock_janapada_client(self):
        """Mock Janapada JSON-RPC client."""
        with patch("src.svami_orchestrator.main.janapada_client") as mock:
            mock.call = AsyncMock()
            yield mock

    @pytest.fixture
    def mock_amatya_client(self):
        """Mock Amatya JSON-RPC client."""
        with patch("src.svami_orchestrator.main.amatya_client") as mock:
            mock.call = AsyncMock()
            yield mock

    def test_health_endpoint(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "timestamp" in data
        assert "dependencies" in data

    def test_agent_manifest_endpoint(self, client):
        """Test agent manifest endpoint."""
        response = client.get("/.well-known/agent.json")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Svami Orchestrator"
        assert data["version"] == "1.0.0"
        assert "methods" in data
        assert len(data["methods"]) > 0

        # Check for answer method
        method_names = [method["name"] for method in data["methods"]]
        assert "answer" in method_names

    @pytest.mark.asyncio
    async def test_answer_endpoint_success(
        self, client, mock_janapada_client, mock_amatya_client, sample_snippets
    ):
        """Test successful answer endpoint workflow."""

        # Mock Janapada search response
        mock_janapada_client.call.return_value = {"snippets": sample_snippets}

        # Mock Amatya advice response
        mock_amatya_client.call.return_value = {
            "advice": "Here's how to implement authentication in FastAPI..."
        }

        # Make request
        response = client.post(
            "/answer",
            json={
                "question": "How do I implement authentication?",
                "role": "backend engineer",
            },
            headers={"Authorization": "Bearer test-token"},
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "request_id" in data
        assert len(data["sources"]) == len(sample_snippets)

        # Verify service calls were made
        mock_janapada_client.call.assert_called_once()
        mock_amatya_client.call.assert_called_once()

        # Verify call parameters
        janapada_call = mock_janapada_client.call.call_args
        assert janapada_call[1]["method"] == "search"
        assert "query" in janapada_call[1]["params"]

        amatya_call = mock_amatya_client.call.call_args
        assert amatya_call[1]["method"] == "advise"
        assert "role" in amatya_call[1]["params"]
        assert "chunks" in amatya_call[1]["params"]

    def test_answer_endpoint_missing_auth(self, client):
        """Test answer endpoint requires authentication."""
        response = client.post(
            "/answer",
            json={
                "question": "How do I implement authentication?",
                "role": "backend engineer",
            },
        )

        # Should require authentication
        assert response.status_code == 401

    def test_answer_endpoint_invalid_request(self, client):
        """Test answer endpoint with invalid request data."""
        response = client.post(
            "/answer",
            json={"invalid_field": "value"},
            headers={"Authorization": "Bearer test-token"},
        )

        # Should return validation error
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_answer_endpoint_janapada_failure(
        self, client, mock_janapada_client, mock_amatya_client
    ):
        """Test answer endpoint when Janapada service fails."""

        # Mock Janapada failure
        mock_janapada_client.call.side_effect = Exception("Service unavailable")

        # Mock Amatya success (graceful degradation)
        mock_amatya_client.call.return_value = {
            "advice": "I don't have specific code snippets, but here's general advice..."
        }

        response = client.post(
            "/answer",
            json={
                "question": "How do I implement authentication?",
                "role": "backend engineer",
            },
            headers={"Authorization": "Bearer test-token"},
        )

        # Should still return response with graceful degradation
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) == 0  # No sources due to Janapada failure

    @pytest.mark.asyncio
    async def test_answer_endpoint_amatya_failure(
        self, client, mock_janapada_client, mock_amatya_client, sample_snippets
    ):
        """Test answer endpoint when Amatya service fails."""

        # Mock Janapada success
        mock_janapada_client.call.return_value = {"snippets": sample_snippets}

        # Mock Amatya failure
        mock_amatya_client.call.side_effect = Exception("Service unavailable")

        response = client.post(
            "/answer",
            json={
                "question": "How do I implement authentication?",
                "role": "backend engineer",
            },
            headers={"Authorization": "Bearer test-token"},
        )

        # Should return error response
        assert response.status_code == 500
        data = response.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_health_endpoint_with_service_failures(
        self, client, mock_janapada_client, mock_amatya_client
    ):
        """Test health endpoint reports degraded status when services fail."""

        # Mock service failures
        mock_janapada_client.call.side_effect = httpx.ConnectTimeout(
            "Connection timeout"
        )
        mock_amatya_client.call.side_effect = httpx.ConnectTimeout("Connection timeout")

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert len(data["dependencies"]) == 2

        # Check that both services are marked as unhealthy
        dep_statuses = {dep["name"]: dep["status"] for dep in data["dependencies"]}
        assert dep_statuses["janapada"] == "unhealthy"
        assert dep_statuses["amatya"] == "unhealthy"

    def test_request_id_injection(self, client):
        """Test that request ID is properly injected into requests."""

        with patch(
            "src.svami_orchestrator.main.janapada_client"
        ) as mock_janapada, patch(
            "src.svami_orchestrator.main.amatya_client"
        ) as mock_amatya:
            mock_janapada.call = AsyncMock(return_value={"snippets": []})
            mock_amatya.call = AsyncMock(return_value={"advice": "test advice"})

            response = client.post(
                "/answer",
                json={"question": "test question", "role": "developer"},
                headers={"Authorization": "Bearer test-token"},
            )

            assert response.status_code == 200
            data = response.json()

            # Verify request_id is in response
            assert "request_id" in data
            assert len(data["request_id"]) > 0

            # Verify request_id was passed to service calls
            janapada_call = mock_janapada.call.call_args
            amatya_call = mock_amatya.call.call_args

            assert janapada_call[1]["id"] == data["request_id"]
            assert amatya_call[1]["id"] == data["request_id"]

    def test_cors_headers(self, client):
        """Test CORS headers are properly set."""
        response = client.options("/answer")

        # Should allow CORS
        assert response.status_code == 200

    def test_rate_limiting(self, client):
        """Test rate limiting (if implemented)."""
        # This test would verify rate limiting functionality
        # Implementation depends on the specific rate limiting strategy
        pass

    def test_request_validation(self, client):
        """Test comprehensive request validation."""

        test_cases = [
            # Missing question
            ({"role": "developer"}, 422),
            # Missing role
            ({"question": "test"}, 422),
            # Empty question
            ({"question": "", "role": "developer"}, 422),
            # Empty role
            ({"question": "test", "role": ""}, 422),
            # Invalid role type
            ({"question": "test", "role": 123}, 422),
            # Valid request
            ({"question": "test", "role": "developer"}, 401),  # 401 due to missing auth
        ]

        for request_data, expected_status in test_cases:
            response = client.post("/answer", json=request_data)
            assert response.status_code == expected_status


class TestRequestIDGeneration:
    """Test request ID generation and propagation."""

    def test_request_id_format(self):
        """Test request ID format and uniqueness."""
        from main import generate_request_id

        # Generate multiple IDs
        ids = [generate_request_id() for _ in range(100)]

        # All should be unique
        assert len(set(ids)) == 100

        # All should have expected format (UUID-like)
        for request_id in ids:
            assert len(request_id) > 10
            assert isinstance(request_id, str)


class TestErrorHandling:
    """Test error handling and user-friendly messages."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_json_rpc_error_translation(self, client):
        """Test JSON-RPC error translation to user-friendly messages."""

        with patch("src.svami_orchestrator.main.janapada_client") as mock_janapada:
            # Mock JSON-RPC error
            mock_janapada.call.side_effect = Exception(
                "JSON-RPC error: Invalid parameters"
            )

            response = client.post(
                "/answer",
                json={"question": "test", "role": "developer"},
                headers={"Authorization": "Bearer test-token"},
            )

            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            # Should contain user-friendly message, not technical JSON-RPC details
            assert "search service" in data["error"].lower()

    def test_timeout_handling(self, client):
        """Test timeout handling for service calls."""

        with patch("src.svami_orchestrator.main.janapada_client") as mock_janapada:
            # Mock timeout
            mock_janapada.call.side_effect = httpx.ReadTimeout("Request timeout")

            response = client.post(
                "/answer",
                json={"question": "test", "role": "developer"},
                headers={"Authorization": "Bearer test-token"},
            )

            assert response.status_code == 500
            data = response.json()
            assert "timeout" in data["error"].lower()
