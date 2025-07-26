"""
Unit tests for JSON-RPC models and shared data models.

Tests validation, serialization/deserialization, and error handling
for all Pydantic models in the common module.
"""

import json
import os
import sys

import pytest
from pydantic import ValidationError

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from common.models import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcError,
    JsonRpcErrorCode,
    Snippet,
    SearchRequest,
    AdviceRequest,
    QueryRequest,
    AnswerResponse
)


class TestJsonRpcError:
    """Test JSON-RPC error model."""
    
    def test_valid_error_creation(self):
        """Test creating valid JSON-RPC errors."""
        error = JsonRpcError(
            code=JsonRpcErrorCode.INVALID_PARAMS,
            message="Invalid parameters",
            data={"field": "query", "reason": "missing"}
        )
        
        assert error.code == -32602
        assert error.message == "Invalid parameters"
        assert error.data["field"] == "query"
    
    def test_error_without_data(self):
        """Test error creation without optional data field."""
        error = JsonRpcError(
            code=JsonRpcErrorCode.METHOD_NOT_FOUND,
            message="Method not found"
        )
        
        assert error.code == -32601
        assert error.message == "Method not found"
        assert error.data is None
    
    def test_custom_error_codes(self):
        """Test custom application error codes."""
        # Custom application error
        error = JsonRpcError(code=-32001, message="Authentication failed")
        assert error.code == -32001
        
        # Server error range
        error = JsonRpcError(code=-32050, message="Server overloaded")
        assert error.code == -32050
    
    def test_invalid_error_codes(self):
        """Test validation of invalid error codes."""
        with pytest.raises(ValidationError):
            JsonRpcError(code=-33000, message="Invalid code")  # Outside valid range


class TestJsonRpcRequest:
    """Test JSON-RPC request model."""
    
    def test_valid_request_creation(self):
        """Test creating valid JSON-RPC requests."""
        request = JsonRpcRequest(
            id="req-123",
            method="search",
            params={"query": "test", "k": 5}
        )
        
        assert request.jsonrpc == "2.0"
        assert request.id == "req-123"
        assert request.method == "search"
        assert request.params["query"] == "test"
    
    def test_request_without_params(self):
        """Test request creation without parameters."""
        request = JsonRpcRequest(id="req-456", method="health_check")
        
        assert request.params == {}
    
    def test_request_serialization(self):
        """Test JSON serialization of requests."""
        request = JsonRpcRequest(
            id="req-789",
            method="advise",
            params={"role": "developer", "chunks": []}
        )
        
        json_str = request.to_json()
        data = json.loads(json_str)
        
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "req-789"
        assert data["method"] == "advise"
        assert "params" in data
    
    def test_invalid_jsonrpc_version(self):
        """Test validation of JSON-RPC version."""
        with pytest.raises(ValidationError):
            JsonRpcRequest(
                jsonrpc="1.0",
                id="req-123",
                method="test"
            )
    
    def test_invalid_method_names(self):
        """Test validation of method names."""
        # Empty method name
        with pytest.raises(ValidationError):
            JsonRpcRequest(id="req-123", method="")
        
        # Reserved method name
        with pytest.raises(ValidationError):
            JsonRpcRequest(id="req-123", method="rpc.test")


class TestJsonRpcResponse:
    """Test JSON-RPC response model."""
    
    def test_successful_response(self):
        """Test creating successful responses."""
        response = JsonRpcResponse.create_success(
            request_id="req-123",
            result={"data": "test result"}
        )
        
        assert response.jsonrpc == "2.0"
        assert response.id == "req-123"
        assert response.result["data"] == "test result"
        assert response.error is None
    
    def test_error_response(self):
        """Test creating error responses."""
        error = JsonRpcError(
            code=JsonRpcErrorCode.INTERNAL_ERROR,
            message="Internal server error"
        )
        response = JsonRpcResponse.create_error(request_id="req-456", error=error)
        
        assert response.jsonrpc == "2.0"
        assert response.id == "req-456"
        assert response.result is None
        assert response.error.code == -32603
    
    def test_response_serialization(self):
        """Test JSON serialization of responses."""
        response = JsonRpcResponse.create_success(
            request_id="req-789",
            result={"snippets": [{"file": "test.py", "content": "code"}]}
        )
        
        json_str = response.to_json()
        data = json.loads(json_str)
        
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "req-789"
        assert "result" in data
        assert "error" not in data  # Should be excluded when None
    
    def test_invalid_response_validation(self):
        """Test validation that prevents both result and error."""
        error = JsonRpcError(code=-32603, message="Error")
        
        with pytest.raises(ValueError):
            JsonRpcResponse(
                id="req-123",
                result={"data": "test"},
                error=error
            )
    
    def test_missing_result_and_error(self):
        """Test validation that requires either result or error."""
        with pytest.raises(ValueError):
            JsonRpcResponse(id="req-123")


class TestSharedDataModels:
    """Test shared data models for service communication."""
    
    def test_snippet_model(self):
        """Test Snippet model validation."""
        snippet = Snippet(
            file_path="src/test.py",
            content="def test(): pass"
        )
        
        assert snippet.file_path == "src/test.py"
        assert snippet.content == "def test(): pass"
    
    def test_search_request_model(self):
        """Test SearchRequest model validation."""
        # Valid request
        request = SearchRequest(query="authentication", k=10)
        assert request.query == "authentication"
        assert request.k == 10
        
        # Default k value
        request = SearchRequest(query="test")
        assert request.k == 5
        
        # Invalid k values
        with pytest.raises(ValidationError):
            SearchRequest(query="test", k=0)  # Too small
        
        with pytest.raises(ValidationError):
            SearchRequest(query="test", k=25)  # Too large
    
    def test_advice_request_model(self):
        """Test AdviceRequest model validation."""
        snippets = [
            Snippet(file_path="auth.py", content="def auth(): pass"),
            Snippet(file_path="middleware.py", content="class Middleware: pass")
        ]
        
        request = AdviceRequest(role="backend_developer", chunks=snippets)
        
        assert request.role == "backend_developer"
        assert len(request.chunks) == 2
        assert request.chunks[0].file_path == "auth.py"
    
    def test_query_request_model(self):
        """Test QueryRequest model validation."""
        # With explicit role
        request = QueryRequest(
            question="How to implement auth?",
            role="security_engineer"
        )
        assert request.question == "How to implement auth?"
        assert request.role == "security_engineer"
        
        # With default role
        request = QueryRequest(question="Test question")
        assert request.role == "developer"
    
    def test_answer_response_model(self):
        """Test AnswerResponse model validation."""
        response = AnswerResponse(
            answer="Here's how to implement authentication...",
            sources=["auth.py", "middleware.py"],
            request_id="req-123"
        )
        
        assert "authentication" in response.answer
        assert len(response.sources) == 2
        assert response.request_id == "req-123"
        
        # With empty sources (default)
        response = AnswerResponse(
            answer="Simple answer",
            request_id="req-456"
        )
        assert response.sources == []


class TestModelSerialization:
    """Test serialization and deserialization of all models."""
    
    def test_round_trip_serialization(self):
        """Test that models can be serialized and deserialized correctly."""
        # Create a complex request
        original_request = JsonRpcRequest(
            id="req-complex",
            method="advise",
            params={
                "role": "backend_developer",
                "chunks": [
                    {"file_path": "test.py", "content": "code"},
                    {"file_path": "auth.py", "content": "auth code"}
                ]
            }
        )
        
        # Serialize to JSON
        json_str = original_request.to_json()
        
        # Deserialize back
        data = json.loads(json_str)
        reconstructed_request = JsonRpcRequest(**data)
        
        # Verify they match
        assert reconstructed_request.id == original_request.id
        assert reconstructed_request.method == original_request.method
        assert reconstructed_request.params == original_request.params
    
    def test_model_config_extra_forbid(self):
        """Test that models reject extra fields."""
        with pytest.raises(ValidationError):
            JsonRpcRequest(
                id="req-123",
                method="test",
                extra_field="not allowed"
            )
        
        with pytest.raises(ValidationError):
            Snippet(
                file_path="test.py",
                content="code",
                extra_field="not allowed"
            )
