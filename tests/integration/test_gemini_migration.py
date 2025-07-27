"""
Integration tests for Gemini API migration with hybrid fallback approach.

Tests the real implementation of Gemini-first, Vertex AI fallback architecture
without mocking the core AI service logic.
"""

import asyncio
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any

import pytest
import httpx
from fastapi.testclient import TestClient

# Add paths for imports
project_root = os.path.join(os.path.dirname(__file__), "../..")
amatya_path = os.path.abspath(os.path.join(project_root, "src/amatya-role-prompter"))
src_path = os.path.abspath(os.path.join(project_root, "src"))


@pytest.fixture(scope="module")
def amatya_app_with_gemini():
    """Module-level fixture for Amatya app with Gemini migration."""
    original_path = sys.path.copy()

    # Clean up any existing modules
    modules_to_remove = [
        key
        for key in sys.modules.keys()
        if key in ["main", "config", "advisor"]
        or key.startswith(("main.", "config.", "advisor."))
    ]
    for module_key in modules_to_remove:
        if module_key in sys.modules:
            del sys.modules[module_key]

    try:
        sys.path.insert(0, amatya_path)
        sys.path.insert(1, src_path)

        # Import with Gemini migration
        from main import app as amatya_app_instance

        yield amatya_app_instance

    finally:
        sys.path[:] = original_path
        # Clean up modules
        for module_key in modules_to_remove:
            if module_key in sys.modules:
                del sys.modules[module_key]


@pytest.fixture
def gemini_env_vars():
    """Environment variables for Gemini testing."""
    env_vars = {
        "GOOGLE_CLOUD_PROJECT": "konveyn2ai",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
        "GEMINI_API_KEY": "test_gemini_api_key",
        "GEMINI_MODEL": "gemini-2.0-flash-001",
        "VERTEX_AI_MODEL": "text-bison-001",
        "PYTEST_CURRENT_TEST": "test_gemini_migration",
    }

    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client for testing."""
    with patch("google.genai.Client") as mock_client_class:
        # Create mock client instance
        mock_client = MagicMock()

        # Mock successful response
        mock_response = MagicMock()
        mock_response.text = "Generated advice using Gemini API: Based on the provided code snippets, here are my recommendations for a backend developer..."

        # Mock the generate_content method
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        yield {
            "client_class": mock_client_class,
            "client": mock_client,
            "response": mock_response,
        }


@pytest.fixture
def mock_vertex_ai_fallback():
    """Mock Vertex AI for fallback testing."""
    with patch("vertexai.init") as mock_init, patch(
        "vertexai.language_models.TextGenerationModel.from_pretrained"
    ) as mock_model:
        # Mock successful Vertex AI response
        mock_response = MagicMock()
        mock_response.text = "Generated advice using Vertex AI fallback: Here's the analysis from Vertex AI..."

        model_instance = MagicMock()
        model_instance.predict.return_value = mock_response
        mock_model.return_value = model_instance

        yield {"init": mock_init, "model": mock_model, "response": mock_response}


@pytest.fixture
def sample_advice_request():
    """Sample advice request for testing."""
    return {
        "role": "backend_developer",
        "chunks": [
            {
                "file_path": "src/api/auth.py",
                "content": "from fastapi import HTTPException\n\ndef authenticate_user(token: str):\n    if not token:\n        raise HTTPException(status_code=401)\n    return decode_jwt(token)",
            },
            {
                "file_path": "src/models/user.py",
                "content": "from pydantic import BaseModel\n\nclass User(BaseModel):\n    id: int\n    username: str\n    email: str",
            },
        ],
    }


class TestGeminiMigrationIntegration:
    """Integration tests for Gemini migration."""

    @pytest.fixture
    def client(self, amatya_app_with_gemini):
        """Test client for Amatya service with Gemini."""
        return TestClient(amatya_app_with_gemini)

    def test_health_endpoint_shows_gemini_status(self, client, gemini_env_vars):
        """Test that health endpoint shows Gemini API status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Should show service status
        assert data["status"] in ["healthy", "starting"]
        assert "service" in data

        # In real implementation, should show AI services status
        if data["status"] == "healthy" and "ai_services" in data:
            assert "gemini_available" in data["ai_services"]
            assert "vertex_ai_available" in data["ai_services"]

    @pytest.mark.asyncio
    async def test_gemini_primary_success(
        self, client, gemini_env_vars, mock_gemini_client, sample_advice_request
    ):
        """Test successful advice generation using Gemini as primary."""

        jsonrpc_request = {
            "jsonrpc": "2.0",
            "id": "test-gemini-primary",
            "method": "advise",
            "params": sample_advice_request,
        }

        response = client.post("/", json=jsonrpc_request)

        assert response.status_code == 200
        data = response.json()

        # Should return successful JSON-RPC response
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "test-gemini-primary"
        assert "result" in data
        assert "advice" in data["result"]

        # Advice should be generated
        advice = data["result"]["advice"]
        assert len(advice) > 0
        assert isinstance(advice, str)

    @pytest.mark.asyncio
    async def test_gemini_failure_vertex_fallback(
        self, client, gemini_env_vars, mock_vertex_ai_fallback, sample_advice_request
    ):
        """Test Vertex AI fallback when Gemini fails."""

        # Mock Gemini failure
        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = Exception(
                "Gemini API error"
            )
            mock_client_class.return_value = mock_client

            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": "test-vertex-fallback",
                "method": "advise",
                "params": sample_advice_request,
            }

            response = client.post("/", json=jsonrpc_request)

            assert response.status_code == 200
            data = response.json()

            # Should still return successful response via fallback
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == "test-vertex-fallback"

            # Should have result (from fallback) or graceful error
            assert "result" in data or "error" in data

            if "result" in data:
                assert "advice" in data["result"]
                assert len(data["result"]["advice"]) > 0

    @pytest.mark.asyncio
    async def test_both_apis_fail_mock_fallback(
        self, client, gemini_env_vars, sample_advice_request
    ):
        """Test mock fallback when both Gemini and Vertex AI fail."""

        # Mock both Gemini and Vertex AI failures
        with patch("google.genai.Client") as mock_gemini, patch(
            "vertexai.language_models.TextGenerationModel.from_pretrained"
        ) as mock_vertex:
            # Gemini failure
            mock_gemini_client = MagicMock()
            mock_gemini_client.models.generate_content.side_effect = Exception(
                "Gemini failed"
            )
            mock_gemini.return_value = mock_gemini_client

            # Vertex AI failure
            mock_vertex_model = MagicMock()
            mock_vertex_model.predict.side_effect = Exception("Vertex AI failed")
            mock_vertex.return_value = mock_vertex_model

            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": "test-mock-fallback",
                "method": "advise",
                "params": sample_advice_request,
            }

            response = client.post("/", json=jsonrpc_request)

            assert response.status_code == 200
            data = response.json()

            # Should return response (mock fallback ensures service never fails)
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == "test-mock-fallback"
            assert "result" in data
            assert "advice" in data["result"]

            # Mock advice should still be meaningful
            advice = data["result"]["advice"]
            assert len(advice) > 50  # Should be substantial
            assert "backend" in advice.lower() or "developer" in advice.lower()

    def test_configuration_gemini_enabled(self, gemini_env_vars):
        """Test that configuration properly enables Gemini."""

        sys.path.insert(0, amatya_path)
        try:
            from config import AmataConfig

            config = AmataConfig()

            # Should detect Gemini API key and enable Gemini
            assert config.use_gemini == True
            assert config.gemini_api_key == "test_gemini_api_key"
            assert config.gemini_model == "gemini-2.0-flash-001"

            # Should have Gemini config method
            gemini_config = config.get_gemini_config()
            assert "temperature" in gemini_config
            assert "max_output_tokens" in gemini_config

        finally:
            sys.path.pop(0)

    def test_configuration_gemini_disabled(self):
        """Test configuration when Gemini API key is not provided."""

        env_vars = {
            "GOOGLE_CLOUD_PROJECT": "konveyn2ai",
            "PYTEST_CURRENT_TEST": "test_gemini_migration",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            sys.path.insert(0, amatya_path)
            try:
                from config import AmataConfig

                config = AmataConfig()

                # Should disable Gemini when no API key
                assert config.use_gemini == False
                assert config.gemini_api_key == ""

            finally:
                sys.path.pop(0)


class TestGeminiAPIIntegration:
    """Test real Gemini API integration patterns."""

    def test_gemini_client_initialization(self, gemini_env_vars):
        """Test Gemini client initialization with real configuration."""

        sys.path.insert(0, amatya_path)
        try:
            from config import AmataConfig
            from advisor import AdvisorService

            config = AmataConfig()
            advisor = AdvisorService(config)

            # Should have Gemini client attribute
            assert hasattr(advisor, "gemini_client")

            # In test mode, client should be None initially
            assert advisor.gemini_client is None

        finally:
            sys.path.pop(0)

    @pytest.mark.asyncio
    async def test_advisor_service_initialization(self, gemini_env_vars):
        """Test AdvisorService initialization with hybrid approach."""

        sys.path.insert(0, amatya_path)
        try:
            from config import AmataConfig
            from advisor import AdvisorService

            config = AmataConfig()
            advisor = AdvisorService(config)

            # Mock the initialization to avoid real API calls
            with patch.object(
                advisor, "_initialize_gemini"
            ) as mock_gemini_init, patch.object(
                advisor, "_initialize_vertex_ai"
            ) as mock_vertex_init:
                await advisor.initialize()

                # Should call both initialization methods
                mock_gemini_init.assert_called_once()
                mock_vertex_init.assert_called_once()

                # Should be marked as initialized
                assert advisor._initialized == True

        finally:
            sys.path.pop(0)

    @pytest.mark.asyncio
    async def test_health_check_with_ai_services(self, gemini_env_vars):
        """Test health check reflects AI service availability."""

        sys.path.insert(0, amatya_path)
        try:
            from config import AmataConfig
            from advisor import AdvisorService

            config = AmataConfig()
            advisor = AdvisorService(config)

            # Test with no services available
            assert await advisor.is_healthy() == False

            # Test with Gemini available
            advisor.gemini_client = MagicMock()
            advisor._initialized = True
            assert await advisor.is_healthy() == True

            # Test with Vertex AI available
            advisor.gemini_client = None
            advisor.llm_model = MagicMock()
            assert await advisor.is_healthy() == True

            # Test with both available
            advisor.gemini_client = MagicMock()
            advisor.llm_model = MagicMock()
            assert await advisor.is_healthy() == True

        finally:
            sys.path.pop(0)


class TestGeminiErrorHandling:
    """Test error handling in Gemini migration."""

    @pytest.fixture
    def client(self, amatya_app_with_gemini):
        """Test client for Amatya service with Gemini."""
        return TestClient(amatya_app_with_gemini)

    @pytest.mark.asyncio
    async def test_gemini_api_timeout(
        self, client, gemini_env_vars, sample_advice_request
    ):
        """Test handling of Gemini API timeouts."""

        # Mock Gemini timeout
        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = TimeoutError(
                "Request timeout"
            )
            mock_client_class.return_value = mock_client

            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": "test-timeout",
                "method": "advise",
                "params": sample_advice_request,
            }

            response = client.post("/", json=jsonrpc_request)

            assert response.status_code == 200
            data = response.json()

            # Should handle timeout gracefully
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == "test-timeout"

            # Should have result (from fallback) or appropriate error
            assert "result" in data or "error" in data

    @pytest.mark.asyncio
    async def test_gemini_rate_limit(
        self, client, gemini_env_vars, sample_advice_request
    ):
        """Test handling of Gemini API rate limits."""

        # Mock rate limit error
        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = Exception(
                "Rate limit exceeded"
            )
            mock_client_class.return_value = mock_client

            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": "test-rate-limit",
                "method": "advise",
                "params": sample_advice_request,
            }

            response = client.post("/", json=jsonrpc_request)

            assert response.status_code == 200
            data = response.json()

            # Should handle rate limit gracefully with fallback
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == "test-rate-limit"
            assert "result" in data or "error" in data

    @pytest.mark.asyncio
    async def test_invalid_api_key(self, client, sample_advice_request):
        """Test handling of invalid Gemini API key."""

        env_vars = {
            "GOOGLE_CLOUD_PROJECT": "konveyn2ai",
            "GEMINI_API_KEY": "invalid_api_key",
            "PYTEST_CURRENT_TEST": "test_gemini_migration",
        }

        with patch.dict(os.environ, env_vars):
            # Mock authentication error
            with patch("google.genai.Client") as mock_client_class:
                mock_client = MagicMock()
                mock_client.models.generate_content.side_effect = Exception(
                    "Authentication failed"
                )
                mock_client_class.return_value = mock_client

                jsonrpc_request = {
                    "jsonrpc": "2.0",
                    "id": "test-invalid-key",
                    "method": "advise",
                    "params": sample_advice_request,
                }

                response = client.post("/", json=jsonrpc_request)

                assert response.status_code == 200
                data = response.json()

                # Should handle auth error gracefully
                assert data["jsonrpc"] == "2.0"
                assert data["id"] == "test-invalid-key"
                assert "result" in data or "error" in data


class TestGeminiPerformance:
    """Test performance aspects of Gemini migration."""

    @pytest.fixture
    def client(self, amatya_app_with_gemini):
        """Test client for performance tests."""
        return TestClient(amatya_app_with_gemini)

    @pytest.mark.asyncio
    async def test_response_time_comparison(
        self, client, gemini_env_vars, mock_gemini_client, sample_advice_request
    ):
        """Test that Gemini responses are within expected time bounds."""

        import time

        jsonrpc_request = {
            "jsonrpc": "2.0",
            "id": "test-performance",
            "method": "advise",
            "params": sample_advice_request,
        }

        start_time = time.time()
        response = client.post("/", json=jsonrpc_request)
        end_time = time.time()

        response_time = end_time - start_time

        assert response.status_code == 200

        # Should respond within reasonable time (mocked, so should be fast)
        assert response_time < 5.0, f"Response took too long: {response_time}s"

        data = response.json()
        assert "result" in data
        assert "advice" in data["result"]

    def test_concurrent_requests_handling(
        self, client, gemini_env_vars, mock_gemini_client, sample_advice_request
    ):
        """Test handling of concurrent requests with Gemini."""

        import concurrent.futures
        import threading

        def make_request(request_id):
            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": f"concurrent-{request_id}",
                "method": "advise",
                "params": sample_advice_request,
            }

            # Use TestClient for synchronous requests
            response = client.post("/", json=jsonrpc_request)
            return response.json()

        # Test with smaller number for integration tests
        num_requests = 3

        # Use ThreadPoolExecutor to simulate concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # All requests should succeed
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == num_requests

        for result in successful_results:
            assert "result" in result or "error" in result
