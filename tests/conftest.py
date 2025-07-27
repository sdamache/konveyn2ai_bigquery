"""
Shared pytest fixtures and configuration for KonveyN2AI tests.
"""

import asyncio
import json
import os
from typing import AsyncGenerator, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
from fastapi.testclient import TestClient

# Test configuration
TEST_CONFIG = {
    "svami_url": "http://localhost:8080",
    "janapada_url": "http://localhost:8081",
    "amatya_url": "http://localhost:8082",
    "test_timeout": 10.0,
    "google_project": "test-project",
    "index_endpoint_id": "test-endpoint-123",
    "index_id": "test-index-456",
}


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TEST_CONFIG


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "GOOGLE_CLOUD_PROJECT": TEST_CONFIG["google_project"],
        "GOOGLE_API_KEY": "test-api-key",
        "INDEX_ENDPOINT_ID": TEST_CONFIG["index_endpoint_id"],
        "INDEX_ID": TEST_CONFIG["index_id"],
        "JANAPADA_URL": TEST_CONFIG["janapada_url"],
        "AMATYA_URL": TEST_CONFIG["amatya_url"],
    }

    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return {
        "question": "How do I implement authentication in FastAPI?",
        "role": "backend engineer",
    }


@pytest.fixture
def sample_snippets():
    """Sample code snippets for testing."""
    return [
        {
            "file_path": "src/auth/jwt.py",
            "content": "def create_jwt_token(data: dict) -> str:\n    return jwt.encode(data, SECRET_KEY, algorithm='HS256')",
        },
        {
            "file_path": "src/auth/middleware.py",
            "content": "async def authenticate_request(request: Request) -> User:\n    token = request.headers.get('Authorization')\n    return decode_token(token)",
        },
        {
            "file_path": "src/models/user.py",
            "content": "class User(BaseModel):\n    id: int\n    username: str\n    email: str",
        },
    ]


@pytest.fixture
def mock_google_credentials():
    """Mock Google Cloud credentials."""
    with patch(
        "google.oauth2.service_account.Credentials.from_service_account_file"
    ) as mock_creds:
        mock_creds.return_value = MagicMock()
        yield mock_creds


@pytest.fixture
def mock_vertex_ai():
    """Mock Vertex AI services."""
    with patch("vertexai.init") as mock_init, patch(
        "vertexai.language_models.TextEmbeddingModel.from_pretrained"
    ) as mock_model:
        # Mock embedding model
        embedding_mock = MagicMock()
        embedding_mock.values = [0.1, 0.2, 0.3] * 1024  # 3072 dimensions

        model_instance = MagicMock()
        model_instance.get_embeddings.return_value = [embedding_mock]
        mock_model.return_value = model_instance

        yield {"init": mock_init, "model": mock_model, "embedding": embedding_mock}


@pytest.fixture
def mock_matching_engine(sample_snippets):
    """Mock Google Cloud Matching Engine."""
    with patch("google.cloud.aiplatform.MatchingEngineIndexEndpoint") as mock_endpoint:
        # Create mock matches from sample snippets
        matches = []
        for snippet in sample_snippets:
            match_mock = MagicMock()
            match_mock.id = f"match-{len(matches)}"
            match_mock.distance = 0.1 + len(matches) * 0.05
            match_mock.metadata = json.dumps(snippet)
            matches.append(match_mock)

        # Mock index operations
        index_mock = MagicMock()
        index_mock.find_neighbors.return_value = [matches]

        endpoint_instance = MagicMock()
        endpoint_instance.get_index.return_value = index_mock
        mock_endpoint.return_value = endpoint_instance

        yield {"endpoint": mock_endpoint, "index": index_mock, "matches": matches}


@pytest.fixture
def mock_gemini_ai():
    """Mock Google Gemini AI (legacy - for backward compatibility)."""
    with patch("google.generativeai.configure") as mock_configure, patch(
        "google.generativeai.GenerativeModel"
    ) as mock_model:
        # Mock model response
        response_mock = MagicMock()
        response_mock.text = "Based on the code snippets provided, here's how to implement authentication in FastAPI:\n\n1. Create JWT tokens using the create_jwt_token function\n2. Use middleware to authenticate requests\n3. Define User models for user data"

        model_instance = MagicMock()
        model_instance.generate_content.return_value = response_mock
        mock_model.return_value = model_instance

        yield {
            "configure": mock_configure,
            "model": mock_model,
            "response": response_mock,
        }


@pytest.fixture
def mock_gemini_new_api():
    """Mock new Google Gemini API (google-genai SDK)."""
    with patch("google.genai.Client") as mock_client_class:
        # Mock client instance
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
def gemini_migration_env():
    """Environment variables for Gemini migration testing."""
    env_vars = {
        "GOOGLE_CLOUD_PROJECT": "konveyn2ai",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
        "GEMINI_API_KEY": "test_gemini_api_key_12345",
        "GEMINI_MODEL": "gemini-2.0-flash-001",
        "VERTEX_AI_MODEL": "text-bison-001",
        "PYTEST_CURRENT_TEST": "test_gemini_migration",
    }

    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
async def async_client():
    """Async HTTP client for integration tests."""
    client = httpx.AsyncClient(timeout=TEST_CONFIG["test_timeout"])
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def integration_client():
    """Synchronous HTTP client for integration tests."""
    with httpx.Client(timeout=TEST_CONFIG["test_timeout"]) as client:
        yield client


@pytest.fixture
def jsonrpc_request():
    """Sample JSON-RPC request."""
    return {
        "jsonrpc": "2.0",
        "id": "test-request-123",
        "method": "search",
        "params": {"query": "authentication implementation", "k": 3},
    }


@pytest.fixture
def mock_jsonrpc_client():
    """Mock JSON-RPC client for service communication."""
    client_mock = AsyncMock()

    # Mock successful search response
    client_mock.call.return_value = {
        "snippets": [
            {"file_path": "src/auth.py", "content": "def authenticate(): pass"},
            {"file_path": "src/models.py", "content": "class User: pass"},
        ]
    }

    return client_mock


# Health check fixtures
@pytest.fixture
def healthy_response():
    """Standard healthy service response."""
    return {
        "status": "healthy",
        "timestamp": "2025-07-26T12:00:00Z",
        "version": "1.0.0",
        "dependencies": [],
    }


@pytest.fixture
def degraded_response():
    """Degraded service response."""
    return {
        "status": "degraded",
        "timestamp": "2025-07-26T12:00:00Z",
        "version": "1.0.0",
        "dependencies": [
            {"name": "janapada", "status": "unhealthy", "error": "Connection timeout"}
        ],
    }


# Test utilities
class MockResponse:
    """Mock HTTP response for testing."""

    def __init__(self, json_data: Dict[str, Any], status_code: int = 200):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                message=f"HTTP {self.status_code}", request=MagicMock(), response=self
            )


@pytest.fixture
def mock_response_factory():
    """Factory for creating mock responses."""
    return MockResponse


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_env():
    """Cleanup environment after each test."""
    yield
    # Any cleanup code here


# Skip markers for conditional testing
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "docker: Tests requiring Docker")
    config.addinivalue_line("markers", "external: Tests requiring external services")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add unit marker to tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add integration marker to tests in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark tests that use docker fixtures
        if any(
            fixture in item.fixturenames
            for fixture in ["docker_compose", "docker_client"]
        ):
            item.add_marker(pytest.mark.docker)

        # Mark tests that use external service fixtures
        if any(
            fixture in item.fixturenames
            for fixture in ["mock_vertex_ai", "mock_gemini_ai"]
        ):
            item.add_marker(pytest.mark.external)
