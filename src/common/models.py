"""
Shared data models for KonveyN2AI multi-agent system.

This module contains:
1. JSON-RPC 2.0 protocol models (JsonRpcRequest, JsonRpcResponse, JsonRpcError)
2. Service communication models (SearchRequest, AdviceRequest, etc.)
3. Common data structures (Snippet, etc.)

All models use Pydantic for validation and serialization.
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Dict, Any, Optional, List, Union
from enum import IntEnum
import json


class JsonRpcErrorCode(IntEnum):
    """Standard JSON-RPC 2.0 error codes."""
    
    # Standard errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # Implementation-defined server errors
    SERVER_ERROR_START = -32099
    SERVER_ERROR_END = -32000
    
    # Custom application errors
    AUTHENTICATION_ERROR = -32001
    AUTHORIZATION_ERROR = -32002
    VALIDATION_ERROR = -32003
    EXTERNAL_SERVICE_ERROR = -32004
    TIMEOUT_ERROR = -32005


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error object."""
    
    code: int = Field(..., description="Error code indicating the error type")
    message: str = Field(..., description="Short description of the error")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional error information")
    
    @field_validator('code')
    @classmethod
    def validate_error_code(cls, v):
        """Validate that error code follows JSON-RPC 2.0 specification."""
        # Allow standard error codes and implementation-defined server errors
        if v in JsonRpcErrorCode.__members__.values():
            return v
        if -32099 <= v <= -32000:  # Server error range
            return v
        if v >= -32000:  # Application error range
            return v
        raise ValueError(f"Invalid JSON-RPC error code: {v}")
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "example": {
                "code": -32602,
                "message": "Invalid params",
                "data": {"parameter": "query", "reason": "missing required field"}
            }
        }
    )


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request object."""
    
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: str = Field(..., description="Request identifier for correlation")
    method: str = Field(..., description="Method name to invoke")
    params: Dict[str, Any] = Field(default_factory=dict, description="Method parameters")
    
    @field_validator('jsonrpc')
    @classmethod
    def validate_jsonrpc_version(cls, v):
        """Ensure JSON-RPC version is 2.0."""
        if v != "2.0":
            raise ValueError("JSON-RPC version must be '2.0'")
        return v

    @field_validator('method')
    @classmethod
    def validate_method_name(cls, v):
        """Validate method name format."""
        if not v or not isinstance(v, str):
            raise ValueError("Method name must be a non-empty string")
        if v.startswith('rpc.'):
            raise ValueError("Method names starting with 'rpc.' are reserved")
        return v
    
    def to_json(self) -> str:
        """Serialize request to JSON string."""
        return self.model_dump_json(exclude_none=True)
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "example": {
                "jsonrpc": "2.0",
                "id": "req-123",
                "method": "search",
                "params": {"query": "authentication middleware", "k": 5}
            }
        }
    )


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 response object."""

    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: str = Field(..., description="Request identifier for correlation")
    result: Optional[Dict[str, Any]] = Field(None, description="Method result (if successful)")
    error: Optional[JsonRpcError] = Field(None, description="Error object (if failed)")

    @field_validator('jsonrpc')
    @classmethod
    def validate_jsonrpc_version(cls, v):
        """Ensure JSON-RPC version is 2.0."""
        if v != "2.0":
            raise ValueError("JSON-RPC version must be '2.0'")
        return v

    def __init__(self, **data):
        """Custom init to handle result/error validation."""
        # If only result is provided, ensure error is not set
        if 'result' in data and 'error' not in data:
            data['error'] = None
        # If only error is provided, ensure result is not set
        elif 'error' in data and 'result' not in data:
            data['result'] = None
        # If both or neither are provided, let validation handle it

        super().__init__(**data)

        # Validate exactly one of result or error
        has_result = self.result is not None
        has_error = self.error is not None

        if has_result and has_error:
            raise ValueError("Response cannot have both result and error")
        if not has_result and not has_error:
            raise ValueError("Response must have either result or error")
    
    def to_json(self) -> str:
        """Serialize response to JSON string."""
        return self.model_dump_json(exclude_none=True)
    
    @classmethod
    def create_success(cls, request_id: str, result: Dict[str, Any]) -> 'JsonRpcResponse':
        """Create a successful response."""
        return cls(id=request_id, result=result)

    @classmethod
    def create_error(cls, request_id: str, error: JsonRpcError) -> 'JsonRpcResponse':
        """Create an error response."""
        return cls(id=request_id, error=error)
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "example": {
                "jsonrpc": "2.0",
                "id": "req-123",
                "result": {"snippets": [{"file_path": "auth.py", "content": "..."}]}
            }
        }
    )


# Shared data models for service communication

class Snippet(BaseModel):
    """Code snippet with file path and content."""
    
    file_path: str = Field(..., description="Path to the source file")
    content: str = Field(..., description="Code snippet content")
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "example": {
                "file_path": "src/auth/middleware.py",
                "content": "def authenticate_request(token: str) -> bool:\n    return validate_token(token)"
            }
        }
    )


class SearchRequest(BaseModel):
    """Request for searching code snippets."""
    
    query: str = Field(..., description="Search query text")
    k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "example": {
                "query": "authentication middleware implementation",
                "k": 5
            }
        }
    )


class AdviceRequest(BaseModel):
    """Request for generating role-specific advice."""
    
    role: str = Field(..., description="User role (e.g., 'backend_developer', 'security_engineer')")
    chunks: List[Snippet] = Field(..., description="Code snippets for context")
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "example": {
                "role": "backend_developer",
                "chunks": [
                    {
                        "file_path": "src/auth.py",
                        "content": "def authenticate(token): ..."
                    }
                ]
            }
        }
    )


class QueryRequest(BaseModel):
    """Request for querying the orchestrator."""
    
    question: str = Field(..., description="User question or query")
    role: str = Field(default="developer", description="User role for context")
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "example": {
                "question": "How do I implement authentication middleware?",
                "role": "backend_developer"
            }
        }
    )


class AnswerResponse(BaseModel):
    """Response containing generated answer with sources."""
    
    answer: str = Field(..., description="Generated answer text")
    sources: List[str] = Field(default_factory=list, description="Source file references")
    request_id: str = Field(..., description="Request ID for tracing")
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "example": {
                "answer": "To implement authentication middleware, create a FastAPI middleware...",
                "sources": ["src/auth/middleware.py", "src/auth/tokens.py"],
                "request_id": "req-123"
            }
        }
    )
