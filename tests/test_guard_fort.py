"""
Unit tests for GuardFort middleware core functionality.

Tests request ID generation, propagation, timing, and basic middleware behavior.
"""

import pytest
import json
import uuid
from unittest.mock import patch, MagicMock
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from guard_fort import GuardFort, init_guard_fort


class TestGuardFortCore:
    """Test core GuardFort middleware functionality."""
    
    def setup_method(self):
        """Set up test FastAPI app with GuardFort middleware."""
        self.app = FastAPI()
        
        # Add a simple test endpoint
        @self.app.get("/test")
        async def test_endpoint(request: Request):
            return {"message": "test", "request_id": request.state.request_id}
        
        @self.app.get("/health")
        async def health_endpoint():
            return {"status": "healthy"}
        
        @self.app.get("/error")
        async def error_endpoint():
            raise ValueError("Test exception")
        
        # Initialize GuardFort middleware
        self.guard_fort = GuardFort(
            app=self.app,
            service_name="test-service",
            enable_auth=True,
            log_level="DEBUG"
        )
        
        self.client = TestClient(self.app)
    
    def test_request_id_generation(self):
        """Test that request ID is generated when not provided."""
        response = self.client.get("/test")
        
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        
        # Verify it's a valid UUID format
        request_id = response.headers["X-Request-ID"]
        try:
            uuid.UUID(request_id)
            valid_uuid = True
        except ValueError:
            valid_uuid = False
        
        assert valid_uuid, f"Request ID {request_id} is not a valid UUID"
        
        # Verify request ID is in response body (from endpoint)
        data = response.json()
        assert data["request_id"] == request_id
    
    def test_request_id_propagation(self):
        """Test that provided request ID is propagated correctly."""
        test_request_id = str(uuid.uuid4())
        
        response = self.client.get(
            "/test",
            headers={"X-Request-ID": test_request_id}
        )
        
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == test_request_id
        
        # Verify request ID is accessible in endpoint
        data = response.json()
        assert data["request_id"] == test_request_id
    
    def test_service_headers(self):
        """Test that GuardFort adds service identification headers."""
        response = self.client.get("/test")
        
        assert response.status_code == 200
        assert response.headers["X-Service"] == "test-service"
        assert response.headers["X-GuardFort-Version"] == "1.0.0"
    
    def test_health_endpoint_auth_bypass(self):
        """Test that health endpoints bypass authentication."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_demo_authentication(self):
        """Test the demo authentication mechanism."""
        # Test with Bearer token (should pass)
        response = self.client.get(
            "/test",
            headers={"Authorization": "Bearer demo-token"}
        )
        assert response.status_code == 200
        
        # Test without auth header (should pass in demo mode)
        response = self.client.get("/test")
        assert response.status_code == 200
        
        # Test with invalid auth format (should fail)
        response = self.client.get(
            "/test", 
            headers={"Authorization": "Invalid format"}
        )
        assert response.status_code == 401
        
        data = response.json()
        assert data["error"] == "authentication_required"
        assert "request_id" in data
    
    def test_exception_handling(self):
        """Test that exceptions are caught and handled properly."""
        response = self.client.get("/error")
        
        assert response.status_code == 500
        assert "X-Request-ID" in response.headers
        
        data = response.json()
        assert data["error"] == "internal_server_error"
        assert "request_id" in data
        assert "An internal error occurred" in data["message"]
        
        # Verify exception details are not exposed
        assert "ValueError" not in data["message"]
        assert "Test exception" not in data["message"]
    
    @patch('logging.Logger.info')
    def test_request_logging(self, mock_log_info):
        """Test that requests are logged with proper structure."""
        response = self.client.get("/test?param=value")
        
        assert response.status_code == 200
        
        # Verify logging was called (at least twice: init + request log)
        assert mock_log_info.call_count >= 2
        
        # Get the request log (should be the last call)
        log_calls = mock_log_info.call_args_list
        request_log_call = None
        
        # Find the request log (not the initialization log)
        for call in log_calls:
            log_data = call[0][0]
            if isinstance(log_data, dict):  # Request logs are dict objects
                request_log_call = log_data
                break
            elif isinstance(log_data, str) and '"method"' in log_data:
                # If it's a JSON string, parse it
                request_log_call = json.loads(log_data)
                break
        
        assert request_log_call is not None, "Request log not found"
        
        # Verify log structure
        assert "request_id" in request_log_call
        assert request_log_call["method"] == "GET"
        assert request_log_call["path"] == "/test"
        assert request_log_call["status_code"] == 200
        assert "duration_ms" in request_log_call
        assert request_log_call["duration_ms"] > 0
        assert "param=value" in request_log_call["query_params"]
    
    @patch('logging.Logger.error')
    def test_exception_logging(self, mock_log_error):
        """Test that exceptions are logged with proper context."""
        response = self.client.get("/error")
        
        assert response.status_code == 500
        
        # Verify exception logging was called
        assert mock_log_error.called
        
        # Get the logged exception data
        log_call = mock_log_error.call_args[0][0]
        
        # Handle both dict and JSON string formats
        if isinstance(log_call, dict):
            log_data = log_call
        else:
            log_data = json.loads(log_call)
        
        # Verify exception log structure
        assert "request_id" in log_data
        assert log_data["method"] == "GET"
        assert log_data["path"] == "/error"
        assert log_data["exception_type"] == "ValueError"
        assert log_data["exception_message"] == "Test exception"
        assert log_data["status"] == "error"
        assert "duration_ms" in log_data
    
    def test_timing_functionality(self):
        """Test that request timing is captured and logged."""
        with patch('logging.Logger.info') as mock_log:
            response = self.client.get("/test")
            
            assert response.status_code == 200
            
            # Find the request log from the mock calls
            request_log_data = None
            for call in mock_log.call_args_list:
                log_data = call[0][0]
                if isinstance(log_data, dict) and "duration_ms" in log_data:
                    request_log_data = log_data
                    break
                elif isinstance(log_data, str) and '"duration_ms"' in log_data:
                    request_log_data = json.loads(log_data)
                    break
            
            assert request_log_data is not None, "Request log with timing not found"
            assert "duration_ms" in request_log_data
            assert isinstance(request_log_data["duration_ms"], (int, float))
            assert request_log_data["duration_ms"] >= 0


class TestGuardFortUtilities:
    """Test GuardFort utility functions."""
    
    def test_init_guard_fort_utility(self):
        """Test the init_guard_fort utility function."""
        app = FastAPI()
        
        guard_fort = init_guard_fort(
            app=app,
            service_name="utility-test",
            enable_auth=False,
            log_level="ERROR"
        )
        
        assert isinstance(guard_fort, GuardFort)
        assert guard_fort.service_name == "utility-test"
        assert guard_fort.enable_auth is False
        
        # Verify middleware was registered by checking app middleware stack
        assert len(app.user_middleware) > 0
    
    def test_guard_fort_disabled_auth(self):
        """Test GuardFort behavior with authentication disabled."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        guard_fort = init_guard_fort(
            app=app,
            service_name="no-auth-test",
            enable_auth=False
        )
        
        client = TestClient(app)
        
        # Test with invalid auth - should still pass since auth is disabled
        response = client.get(
            "/test",
            headers={"Authorization": "Invalid format"}
        )
        
        assert response.status_code == 200
        assert response.headers["X-Service"] == "no-auth-test"


class TestGuardFortIntegration:
    """Integration tests for GuardFort middleware."""
    
    def test_multiple_requests_with_different_ids(self):
        """Test handling multiple requests with different request IDs."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint(request: Request):
            return {"request_id": request.state.request_id}
        
        GuardFort(app, service_name="multi-test")
        client = TestClient(app)
        
        # Make multiple requests with different IDs
        id1 = str(uuid.uuid4())
        id2 = str(uuid.uuid4())
        
        response1 = client.get("/test", headers={"X-Request-ID": id1})
        response2 = client.get("/test", headers={"X-Request-ID": id2})
        
        assert response1.json()["request_id"] == id1
        assert response2.json()["request_id"] == id2
        assert response1.headers["X-Request-ID"] == id1
        assert response2.headers["X-Request-ID"] == id2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])