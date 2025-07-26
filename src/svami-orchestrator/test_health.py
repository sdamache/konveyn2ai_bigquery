#!/usr/bin/env python3
"""
Test script for the Svami orchestrator health monitoring functionality.
Tests both basic and detailed health check endpoints.
"""

import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from main import app
from common.models import JsonRpcResponse, JsonRpcError


def test_basic_health_endpoint():
    """Test the basic /health endpoint provided by Guard-Fort."""
    print("Testing basic health endpoint...")

    client = TestClient(app)
    response = client.get("/health")

    print(f"Basic health status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Basic health check: {data.get('status', 'unknown')}")
        print(f"Service: {data.get('service', 'unknown')}")
        return True
    else:
        print(f"❌ Basic health check failed: {response.text}")
        return False


def test_detailed_health_uninitialized():
    """Test detailed health check when RPC clients are not initialized."""
    print("\nTesting detailed health with uninitialized clients...")

    # Ensure clients are None
    import main

    main.janapada_client = None
    main.amatya_client = None

    client = TestClient(app)
    response = client.get("/health/detailed")

    print(f"Detailed health status (uninitialized): {response.status_code}")

    if response.status_code == 503:
        data = response.json()
        print(f"✅ Correctly reports degraded status: {data.get('status')}")
        print(f"Error: {data.get('error')}")
        print(f"Dependencies: {data.get('dependencies', {})}")
        return True
    else:
        print(f"❌ Unexpected status code: {response.status_code}")
        return False


def test_detailed_health_with_healthy_services():
    """Test detailed health check with healthy mock services."""
    print("\nTesting detailed health with healthy services...")

    import main

    # Create mock clients that return healthy responses
    mock_janapada = AsyncMock()
    mock_janapada.call.return_value = JsonRpcResponse(
        id="health-check", result={"status": "healthy"}
    )

    mock_amatya = AsyncMock()
    mock_amatya.call.return_value = JsonRpcResponse(
        id="health-check", result={"status": "healthy"}
    )

    main.janapada_client = mock_janapada
    main.amatya_client = mock_amatya

    client = TestClient(app)
    response = client.get("/health/detailed")

    print(f"Detailed health status (healthy): {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ All services healthy: {data.get('status')}")
        print(
            f"Janapada: {data.get('dependencies', {}).get('janapada', {}).get('status')}"
        )
        print(f"Amatya: {data.get('dependencies', {}).get('amatya', {}).get('status')}")
        print(f"Health check duration: {data.get('health_check_duration_ms')}ms")

        # Verify calls were made
        assert mock_janapada.call.called, "Janapada health check should be called"
        assert mock_amatya.call.called, "Amatya health check should be called"

        return True
    else:
        print(f"❌ Unexpected status: {response.status_code} - {response.text}")
        return False


def test_detailed_health_with_partial_failure():
    """Test detailed health check with one service failing."""
    print("\nTesting detailed health with partial service failure...")

    import main

    # Janapada healthy, Amatya failing
    mock_janapada = AsyncMock()
    mock_janapada.call.return_value = JsonRpcResponse(
        id="health-check", result={"status": "healthy"}
    )

    mock_amatya = AsyncMock()
    mock_amatya.call.return_value = JsonRpcResponse(
        id="health-check",
        error=JsonRpcError(code=-32000, message="Service unavailable"),
    )

    main.janapada_client = mock_janapada
    main.amatya_client = mock_amatya

    client = TestClient(app)
    response = client.get("/health/detailed")

    print(f"Detailed health status (partial failure): {response.status_code}")

    if response.status_code == 200:  # Should still be 200 for partial functionality
        data = response.json()
        print(
            f"✅ Partial failure handled: {data.get('status')}"
        )  # Should be "degraded"
        print(f"Unhealthy services: {data.get('unhealthy_services', [])}")

        dependencies = data.get("dependencies", {})
        print(f"Janapada: {dependencies.get('janapada', {}).get('status')}")
        print(f"Amatya: {dependencies.get('amatya', {}).get('status')}")

        return True
    else:
        print(f"❌ Unexpected status: {response.status_code} - {response.text}")
        return False


def test_detailed_health_with_timeout():
    """Test detailed health check with service timeout."""
    print("\nTesting detailed health with service timeout...")

    import main

    # Create a mock that raises TimeoutError
    async def timeout_call(*args, **kwargs):
        raise asyncio.TimeoutError("Connection timed out")

    mock_janapada = AsyncMock()
    mock_janapada.call.side_effect = timeout_call

    mock_amatya = AsyncMock()
    mock_amatya.call.return_value = JsonRpcResponse(
        id="health-check", result={"status": "healthy"}
    )

    main.janapada_client = mock_janapada
    main.amatya_client = mock_amatya

    client = TestClient(app)
    response = client.get("/health/detailed")

    print(f"Detailed health status (timeout): {response.status_code}")

    if response.status_code == 200:  # Partial functionality
        data = response.json()
        print(f"✅ Timeout handled: {data.get('status')}")

        dependencies = data.get("dependencies", {})
        janapada_status = dependencies.get("janapada", {})
        print(f"Janapada timeout status: {janapada_status.get('status')}")
        print(f"Timeout error: {janapada_status.get('error', '')[:50]}...")

        return True
    else:
        print(f"❌ Unexpected status: {response.status_code}")
        return False


if __name__ == "__main__":
    print("Testing Svami Orchestrator health monitoring...")

    results = [
        test_basic_health_endpoint(),
        test_detailed_health_uninitialized(),
        test_detailed_health_with_healthy_services(),
        test_detailed_health_with_partial_failure(),
        test_detailed_health_with_timeout(),
    ]

    passed = sum(results)
    total = len(results)

    print(f"\n✅ Health monitoring tests completed: {passed}/{total} passed")

    if passed == total:
        print("🎉 All health monitoring tests passed!")
    else:
        print("⚠️  Some tests failed - check implementation")
