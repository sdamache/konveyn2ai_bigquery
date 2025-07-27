#!/usr/bin/env python3
"""
Test script for GuardFort exception handling and service integration functionality.

This script tests the enhanced GuardFort middleware with:
- Custom exception classes and handling
- Advanced exception categorization and logging
- Service integration utilities and health checks
- Service registry and status monitoring
- Request context creation for service-to-service communication
"""

import os
import sys

# Add the src directory to the path to import GuardFort
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from guard_fort.guard_fort import (  # noqa: E402
    AuthenticationException,
    AuthorizationException,
    ConfigurationException,
    ExternalServiceException,
    RateLimitException,
    ServiceUnavailableException,
    ValidationException,
    init_guard_fort,
)


def test_custom_exceptions():
    """Test custom exception classes and their behavior."""
    print("=== Testing Custom Exception Classes ===")

    # Create FastAPI app
    app = FastAPI(title="Exception Test Service")

    # Initialize GuardFort with debug mode enabled
    guard_fort = init_guard_fort(
        app=app,
        service_name="exception-test-service",
        enable_auth=False,
        debug_mode=True,
        log_format="json",
        add_metrics_endpoint=False,
        add_health_endpoint=False,
        add_service_status_endpoint=False,
    )

    # Add test endpoints that raise different exceptions
    @app.get("/test/auth-error")
    async def test_auth_error():
        raise AuthenticationException("Invalid token", reason="token_expired")

    @app.get("/test/authorization-error")
    async def test_authorization_error():
        raise AuthorizationException("Access denied", resource="/admin")

    @app.get("/test/validation-error")
    async def test_validation_error():
        raise ValidationException("Invalid input", field="email")

    @app.get("/test/rate-limit-error")
    async def test_rate_limit_error():
        raise RateLimitException("Too many requests", retry_after=60)

    @app.get("/test/service-unavailable")
    async def test_service_unavailable():
        raise ServiceUnavailableException("Service down", service_name="database")

    @app.get("/test/external-service-error")
    async def test_external_service_error():
        raise ExternalServiceException(
            "API call failed", service_name="payment-api", upstream_status=502
        )

    @app.get("/test/config-error")
    async def test_config_error():
        raise ConfigurationException("Missing config", config_key="API_KEY")

    @app.get("/test/http-exception")
    async def test_http_exception():
        raise HTTPException(status_code=418, detail="I'm a teapot")

    @app.get("/test/generic-error")
    async def test_generic_error():
        raise ValueError("This is a generic error")

    @app.get("/test/key-error")
    async def test_key_error():
        data = {"name": "test"}
        return data["missing_key"]  # This will raise KeyError

    # Test with client
    client = TestClient(app)

    test_cases = [
        ("/test/auth-error", 401, "authentication_error"),
        ("/test/authorization-error", 403, "authorization_error"),
        ("/test/validation-error", 422, "validation_error"),
        ("/test/rate-limit-error", 429, "rate_limit_error"),
        ("/test/service-unavailable", 503, "service_unavailable"),
        ("/test/external-service-error", 502, "external_service_error"),
        ("/test/config-error", 500, "configuration_error"),
        ("/test/http-exception", 418, "http_418"),
        ("/test/generic-error", 400, "value_error"),
        ("/test/key-error", 400, "key_error"),
    ]

    print("\nTesting different exception types:")
    for endpoint, expected_status, expected_error in test_cases:
        response = client.get(endpoint)
        data = response.json()

        print(
            f"   {endpoint}: Status {response.status_code}, Error: {data.get('error')}"
        )

        assert (
            response.status_code == expected_status
        ), f"Expected {expected_status}, got {response.status_code}"
        assert (
            data.get("error") == expected_error
        ), f"Expected {expected_error}, got {data.get('error')}"
        assert "request_id" in data, "Response should include request_id"
        assert "timestamp" in data, "Response should include timestamp"

        # Check for exception-specific details in debug mode
        if expected_error in [
            "rate_limit_error",
            "external_service_error",
            "validation_error",
        ]:
            assert "details" in data, f"Expected details for {expected_error}"

    print("✅ Custom exceptions test passed!")
    return guard_fort


def test_service_integration():
    """Test service integration utilities."""
    print("\n=== Testing Service Integration Utilities ===")

    app = FastAPI(title="Integration Test Service")

    guard_fort = init_guard_fort(
        app=app,
        service_name="integration-test",
        enable_auth=False,
        add_metrics_endpoint=False,
        add_health_endpoint=False,
    )

    # Test service registration
    print("\n1. Testing service registration...")
    guard_fort.register_external_service(
        service_name="user-service",
        base_url="http://localhost:8001",
        health_endpoint="/health",
    )

    guard_fort.register_external_service(
        service_name="payment-service",
        base_url="http://localhost:8002",
        health_endpoint="/status",
    )

    # Get service registry
    registry = guard_fort.get_service_registry()
    print(f"   Registered services: {list(registry.keys())}")
    assert "user-service" in registry
    assert "payment-service" in registry
    assert registry["user-service"]["base_url"] == "http://localhost:8001"
    assert registry["payment-service"]["health_endpoint"] == "/status"

    # Test service headers creation
    print("\n2. Testing service request headers...")
    headers = guard_fort.create_service_headers("test-request-123", "integration-test")
    print(f"   Generated headers: {list(headers.keys())}")
    assert "X-Request-ID" in headers
    assert "X-Source-Service" in headers
    assert "X-Correlation-ID" in headers
    assert "User-Agent" in headers
    assert headers["X-Request-ID"] == "test-request-123"
    assert headers["X-Source-Service"] == "integration-test"

    # Test services status endpoint
    print("\n3. Testing services status endpoint...")
    client = TestClient(app)
    response = client.get("/services")

    print(f"   Services endpoint: Status {response.status_code}")
    assert response.status_code == 200

    data = response.json()
    assert "registered_services" in data
    assert "summary" in data
    assert data["summary"]["total_services"] == 2
    print(f"   Total registered services: {data['summary']['total_services']}")

    print("✅ Service integration test passed!")
    return guard_fort


def test_exception_logging_integration():
    """Test exception logging integration with structured logger."""
    print("\n=== Testing Exception Logging Integration ===")

    app = FastAPI(title="Logging Test Service")

    init_guard_fort(
        app=app,
        service_name="logging-test",
        enable_auth=False,
        log_format="json",
        include_trace_logs=True,
        debug_mode=True,
        add_metrics_endpoint=False,
        add_health_endpoint=False,
        add_service_status_endpoint=False,
    )

    @app.get("/test/logged-error")
    async def test_logged_error():
        raise ValidationException(
            "This error should be logged with full context", field="test_field"
        )

    @app.get("/test/generic-logged-error")
    async def test_generic_logged_error():
        raise ConnectionError("Connection failed to external service")

    client = TestClient(app)

    print("\n1. Testing exception logging with custom exception...")
    response = client.get("/test/logged-error")
    data = response.json()

    print(
        f"   Custom exception: Status {response.status_code}, Error: {data.get('error')}"
    )
    assert response.status_code == 422
    assert data.get("error") == "validation_error"
    assert "details" in data  # Debug mode should include details
    assert data["details"]["field"] == "test_field"

    print("\n2. Testing exception logging with generic exception...")
    response = client.get("/test/generic-logged-error")
    data = response.json()

    print(
        f"   Generic exception: Status {response.status_code}, Error: {data.get('error')}"
    )
    assert response.status_code == 502
    assert data.get("error") == "connection_error"

    print("✅ Exception logging integration test passed!")


def test_service_health_monitoring():
    """Test service health monitoring capabilities."""
    print("\n=== Testing Service Health Monitoring ===")

    app = FastAPI(title="Health Monitor Test Service")

    guard_fort = init_guard_fort(
        app=app,
        service_name="health-monitor-test",
        enable_auth=False,
        add_metrics_endpoint=False,
        add_health_endpoint=False,
    )

    # Register some test services
    guard_fort.register_external_service(
        service_name="api-service", base_url="http://localhost:9001"
    )

    guard_fort.register_external_service(
        service_name="db-service",
        base_url="http://localhost:9002",
        health_endpoint="/ping",
    )

    @app.get("/test/check-health/{service_name}")
    async def test_check_health(service_name: str):
        try:
            health_status = await guard_fort.check_service_health(service_name)
            return {"status": "success", "health": health_status}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    client = TestClient(app)

    print("\n1. Testing individual service health checks...")
    # Test health check for registered service (will use mock implementation)
    response = client.get("/test/check-health/api-service")
    data = response.json()

    print(f"   API service health check: {data.get('status')}")
    assert data.get("status") == "success"
    assert "health" in data

    # Test health check for unregistered service
    response = client.get("/test/check-health/unknown-service")
    data = response.json()

    print(f"   Unknown service health check: {data.get('status')}")
    assert data.get("status") == "error"
    assert "not registered" in data.get("message", "")

    print("\n2. Testing services status endpoint with health info...")
    response = client.get("/services")
    data = response.json()

    print(f"   Services status endpoint: Status {response.status_code}")
    assert response.status_code == 200
    assert "registered_services" in data

    # Check that health information is included
    for service_name, service_info in data["registered_services"].items():
        print(
            f"   Service {service_name}: Status {service_info.get('status', 'unknown')}"
        )
        assert "base_url" in service_info
        assert "health_endpoint" in service_info

    print("✅ Service health monitoring test passed!")


def test_debug_vs_production_mode():
    """Test differences between debug and production mode."""
    print("\n=== Testing Debug vs Production Mode ===")

    # Test debug mode
    print("\n1. Testing debug mode (detailed errors)...")
    app_debug = FastAPI(title="Debug Mode Test")

    init_guard_fort(
        app=app_debug,
        service_name="debug-test",
        enable_auth=False,
        debug_mode=True,
        log_format="json",
        add_metrics_endpoint=False,
        add_health_endpoint=False,
        add_service_status_endpoint=False,
    )

    @app_debug.get("/test/debug-error")
    async def test_debug_error():
        raise ValueError("This is a detailed error message for debugging")

    client_debug = TestClient(app_debug)
    response = client_debug.get("/test/debug-error")
    data = response.json()

    print(f"   Debug mode error: {data.get('message')}")
    assert "detailed error message" in data.get("message", "")

    # Test production mode
    print("\n2. Testing production mode (sanitized errors)...")
    app_prod = FastAPI(title="Production Mode Test")

    init_guard_fort(
        app=app_prod,
        service_name="production-test",
        enable_auth=False,
        debug_mode=False,
        log_format="json",
        add_metrics_endpoint=False,
        add_health_endpoint=False,
        add_service_status_endpoint=False,
    )

    @app_prod.get("/test/prod-error")
    async def test_prod_error():
        raise ValueError("This detailed error should be sanitized")

    client_prod = TestClient(app_prod)
    response = client_prod.get("/test/prod-error")
    data = response.json()

    print(f"   Production mode error: {data.get('message')}")
    assert "Invalid value provided" in data.get("message", "")
    assert "detailed error should be sanitized" not in data.get("message", "")

    print("✅ Debug vs production mode test passed!")


def main():
    """Run all tests."""
    print("🚀 Starting GuardFort Exception Handling and Integration Tests")
    print("=" * 70)

    try:
        # Run all tests
        test_custom_exceptions()
        test_service_integration()
        test_exception_logging_integration()
        test_service_health_monitoring()
        test_debug_vs_production_mode()

        print("\n" + "=" * 70)
        print("🎉 All tests passed successfully!")
        print("\nGuardFort middleware now includes:")
        print("✅ Custom exception classes with proper categorization")
        print("✅ Advanced exception handler with comprehensive logging")
        print("✅ Service integration utilities and registry")
        print("✅ Service health monitoring and status endpoints")
        print("✅ Request context creation for service-to-service communication")
        print("✅ Debug vs production mode error handling")
        print("✅ Integration with structured logging and metrics")
        print("✅ Sanitized error responses for security")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
