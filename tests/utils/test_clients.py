"""
Unified TestClient factory for KonveyN2AI services.

This module provides a centralized way to create TestClient instances
with proper service initialization and mocking support.

Industry best practice: factory pattern for test client creation.
"""

from typing import Any
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from .service_imports import get_service_app, get_service_main, import_common_models


class ServiceTestClient:
    """
    Enhanced TestClient factory with service-specific initialization.

    Handles the complexity of TestClient + service mocking in one place.
    """

    def __init__(self, service_name: str, auto_init: bool = True):
        """
        Initialize a test client for a specific service.

        Args:
            service_name: One of 'svami', 'amatya', 'janapada'
            auto_init: Whether to automatically initialize service clients
        """
        self.service_name = service_name
        self.app = get_service_app(service_name)
        self.main_module = get_service_main(service_name)
        self.client = TestClient(self.app)
        self.common_models = import_common_models()

        if auto_init and service_name == "svami":
            self._initialize_svami_clients()

    def _initialize_svami_clients(self):
        """
        Initialize Svami orchestrator clients manually.

        Required because TestClient doesn't trigger FastAPI lifespan events.
        """
        JsonRpcClient = self.common_models["JsonRpcClient"]

        # Initialize the global clients manually since TestClient doesn't trigger lifespan
        self.main_module.janapada_client = JsonRpcClient("http://localhost:8001")
        self.main_module.amatya_client = JsonRpcClient("http://localhost:8002")

    def create_mocked_clients(self) -> dict[str, Any]:
        """
        Create mocked service clients for Svami orchestrator testing.

        Returns:
            Dictionary containing mock objects for janapada and amatya clients

        Example:
            >>> client = ServiceTestClient('svami')
            >>> mocks = client.create_mocked_clients()
            >>> mocks['janapada_call'].return_value = {"snippets": []}
        """
        if self.service_name != "svami":
            raise ValueError("Mocked clients are only available for Svami orchestrator")

        JsonRpcResponse = self.common_models["JsonRpcResponse"]

        # Create mock objects for the client calls
        mock_janapada_call = AsyncMock()
        mock_amatya_call = AsyncMock()

        # Set up default responses
        mock_janapada_call.return_value = JsonRpcResponse(
            id="test", result={"snippets": []}
        )
        mock_amatya_call.return_value = JsonRpcResponse(
            id="test", result={"advice": "Generated advice"}
        )

        # Apply the mocks to the actual client instances
        if hasattr(self.main_module, "janapada_client"):
            self.main_module.janapada_client.call = mock_janapada_call
        if hasattr(self.main_module, "amatya_client"):
            self.main_module.amatya_client.call = mock_amatya_call

        return {
            "janapada_call": mock_janapada_call,
            "amatya_call": mock_amatya_call,
            "JsonRpcResponse": JsonRpcResponse,
        }

    def get_client(self) -> TestClient:
        """Get the underlying TestClient instance."""
        return self.client

    def get_app(self) -> Any:
        """Get the underlying FastAPI app instance."""
        return self.app

    def get_main_module(self) -> Any:
        """Get the service main module for accessing global variables."""
        return self.main_module


def create_test_client(service_name: str, auto_init: bool = True) -> ServiceTestClient:
    """
    Factory function to create a ServiceTestClient.

    This is the main function tests should use for creating test clients.

    Args:
        service_name: One of 'svami', 'amatya', 'janapada'
        auto_init: Whether to automatically initialize service clients

    Returns:
        ServiceTestClient instance

    Example:
        >>> from tests.utils.test_clients import create_test_client
        >>> service_client = create_test_client('svami')
        >>> client = service_client.get_client()
        >>> mocks = service_client.create_mocked_clients()
    """
    return ServiceTestClient(service_name, auto_init)


def create_simple_client(service_name: str) -> TestClient:
    """
    Create a simple TestClient without additional setup.

    Use this for basic tests that don't need mocking or special initialization.

    Args:
        service_name: One of 'svami', 'amatya', 'janapada'

    Returns:
        FastAPI TestClient instance

    Example:
        >>> from tests.utils.test_clients import create_simple_client
        >>> client = create_simple_client('amatya')
        >>> response = client.get("/health")
    """
    app = get_service_app(service_name)
    return TestClient(app)


def create_mocked_svami_client() -> dict[str, Any]:
    """
    Convenience function to create a fully mocked Svami client.

    Returns:
        Dictionary with 'client' and mock objects

    Example:
        >>> from tests.utils.test_clients import create_mocked_svami_client
        >>> setup = create_mocked_svami_client()
        >>> client = setup['client']
        >>> setup['janapada_call'].return_value = custom_response
    """
    service_client = create_test_client("svami")
    mocks = service_client.create_mocked_clients()

    return {
        "client": service_client.get_client(),
        "service_client": service_client,
        **mocks,
    }


# Service configuration helpers
def get_service_config(service_name: str) -> dict[str, Any]:
    """
    Get service-specific configuration for testing.

    Args:
        service_name: One of 'svami', 'amatya', 'janapada'

    Returns:
        Configuration dictionary for the service
    """
    configs = {
        "svami": {
            "base_url": "http://localhost:8080",
            "requires_auth": True,
            "has_dependencies": True,
            "dependencies": ["janapada", "amatya"],
        },
        "amatya": {
            "base_url": "http://localhost:8081",
            "requires_auth": False,
            "has_dependencies": False,
            "dependencies": [],
        },
        "janapada": {
            "base_url": "http://localhost:8082",
            "requires_auth": False,
            "has_dependencies": False,
            "dependencies": [],
        },
    }

    if service_name not in configs:
        raise ValueError(f"Unknown service: {service_name}")

    return configs[service_name]


# Common test data factories
def create_sample_query_request() -> dict[str, Any]:
    """Create a sample query request for testing."""
    return {
        "question": "How do I implement authentication?",
        "role": "backend engineer",
    }


def create_sample_search_request() -> dict[str, Any]:
    """Create a sample search request for testing."""
    return {
        "jsonrpc": "2.0",
        "id": "test-123",
        "method": "search",
        "params": {"query": "authentication implementation", "k": 5},
    }


def create_sample_advice_request() -> dict[str, Any]:
    """Create a sample advice request for testing."""
    return {
        "jsonrpc": "2.0",
        "id": "test-456",
        "method": "advise",
        "params": {
            "role": "backend engineer",
            "chunks": [
                {
                    "content": "def authenticate_user():",
                    "file_path": "auth.py",
                    "line_number": 1,
                }
            ],
        },
    }


# Error simulation helpers
def simulate_service_failure(mock_call: AsyncMock, error_type: str = "timeout"):
    """
    Configure a mock to simulate service failures.

    Args:
        mock_call: The AsyncMock object to configure
        error_type: Type of error to simulate ('timeout', 'connection', 'server_error')
    """
    error_types = {
        "timeout": TimeoutError("Service request timed out"),
        "connection": ConnectionError("Failed to connect to service"),
        "server_error": Exception("Internal server error"),
    }

    if error_type not in error_types:
        raise ValueError(f"Unknown error type: {error_type}")

    mock_call.side_effect = error_types[error_type]
