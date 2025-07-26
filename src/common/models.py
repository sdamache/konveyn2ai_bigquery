"""
Shared data models for KonveyN2AI multi-agent system.

This module contains:
1. JSON-RPC 2.0 protocol models (JsonRpcRequest, JsonRpcResponse, JsonRpcError)
2. Service communication models (SearchRequest, AdviceRequest, etc.)
3. Common data structures (Snippet, etc.)

All models use Pydantic for validation and serialization.
"""

from enum import IntEnum, Enum
from typing import Any, Optional
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    data: Optional[dict[str, Any]] = Field(
        None, description="Additional error information"
    )

    @field_validator("code")
    @classmethod
    def validate_error_code(cls, v: int) -> int:
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
        extra="forbid",
        json_schema_extra={
            "example": {
                "code": -32602,
                "message": "Invalid params",
                "data": {"parameter": "query", "reason": "missing required field"},
            }
        },
    )


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request object."""

    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: Optional[str] = Field(
        None,
        description="Request identifier for correlation (optional for notifications)",
    )
    method: str = Field(..., description="Method name to invoke")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Method parameters"
    )

    @field_validator("jsonrpc")
    @classmethod
    def validate_jsonrpc_version(cls, v: str) -> str:
        """Ensure JSON-RPC version is 2.0."""
        if v != "2.0":
            raise ValueError("JSON-RPC version must be '2.0'")
        return v

    @field_validator("method")
    @classmethod
    def validate_method_name(cls, v: str) -> str:
        """Validate method name format."""
        if not v or not isinstance(v, str):
            raise ValueError("Method name must be a non-empty string")
        if v.startswith("rpc."):
            raise ValueError("Method names starting with 'rpc.' are reserved")
        return v

    def to_json(self) -> str:
        """Serialize request to JSON string."""
        return self.model_dump_json(exclude_none=True)

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "jsonrpc": "2.0",
                "id": "req-123",
                "method": "search",
                "params": {"query": "authentication middleware", "k": 5},
            }
        },
    )


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 response object."""

    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: str = Field(..., description="Request identifier for correlation")
    result: Optional[dict[str, Any]] = Field(
        None, description="Method result (if successful)"
    )
    error: Optional[JsonRpcError] = Field(None, description="Error object (if failed)")

    @field_validator("jsonrpc")
    @classmethod
    def validate_jsonrpc_version(cls, v: str) -> str:
        """Ensure JSON-RPC version is 2.0."""
        if v != "2.0":
            raise ValueError("JSON-RPC version must be '2.0'")
        return v

    def __init__(self, **data: Any) -> None:
        """Custom init to handle result/error validation."""
        # If only result is provided, ensure error is not set
        if "result" in data and "error" not in data:
            data["error"] = None
        # If only error is provided, ensure result is not set
        elif "error" in data and "result" not in data:
            data["result"] = None
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
    def create_success(
        cls, request_id: str, result: dict[str, Any]
    ) -> "JsonRpcResponse":
        """Create a successful response."""
        return cls(id=request_id, result=result)

    @classmethod
    def create_error(cls, request_id: str, error: JsonRpcError) -> "JsonRpcResponse":
        """Create an error response."""
        return cls(id=request_id, error=error)

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "jsonrpc": "2.0",
                "id": "req-123",
                "result": {"snippets": [{"file_path": "auth.py", "content": "..."}]},
            }
        },
    )


# Shared data models for service communication

# Enhanced vector search models (adapted from konveyor integration)


class SearchType(str, Enum):
    """Search type enumeration for vector search operations (adapted from konveyor)."""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"


class SearchQuery(BaseModel):
    """Enhanced search query with embedding support (adapted from konveyor/core/search/interfaces.py)."""

    text: str = Field(..., description="Search query text")
    search_type: SearchType = Field(
        default=SearchType.HYBRID, description="Type of search to perform"
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of results to return"
    )
    filters: Optional[dict[str, Any]] = Field(
        None, description="Additional search filters"
    )
    min_score: Optional[float] = Field(
        None, description="Minimum relevance score threshold"
    )
    embedding: Optional[list[float]] = Field(
        None, description="Query embedding vector (3072 dimensions)"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "text": "authentication middleware implementation",
                "embedding": None,
                "search_type": "hybrid",
                "top_k": 5,
                "filters": {"file_type": "python"},
            }
        },
    )


class SearchResult(BaseModel):
    """Search result with relevance score (adapted from konveyor/core/search/interfaces.py)."""

    id: str = Field(..., description="Unique result identifier")
    content: str = Field(..., description="Result content")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Relevance score (0.0 to 1.0)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Result metadata"
    )
    document_id: Optional[str] = Field(None, description="Parent document identifier")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")
    chunk_index: Optional[int] = Field(None, description="Chunk position in document")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "id": "result-abc123",
                "content": "def authenticate_request(token: str) -> bool:\n    return validate_token(token)",
                "score": 0.85,
                "metadata": {
                    "language": "python",
                    "lines": "15-20",
                    "file_path": "src/auth/middleware.py",
                },
                "document_id": "doc-456",
                "chunk_id": "chunk-123",
                "chunk_index": 0,
            }
        },
    )


class DocumentChunk(BaseModel):
    """Document chunk for processing and indexing (adapted from konveyor/core/search/interfaces.py)."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique chunk identifier"
    )
    content: str = Field(..., description="Chunk content")
    document_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(
        ..., ge=0, description="Chunk position in document (0-based)"
    )
    embedding: Optional[list[float]] = Field(
        None, description="Chunk embedding vector (3072 dimensions)"
    )
    metadata: Optional[dict[str, Any]] = Field(None, description="Chunk metadata")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "id": "chunk-abc123",
                "content": "class AuthenticationMiddleware:\n    def __init__(self):\n        pass",
                "document_id": "doc-xyz789",
                "chunk_index": 0,
                "embedding": None,
                "metadata": {"start_line": 1, "end_line": 3, "language": "python"},
            }
        },
    )


class Document(BaseModel):
    """Document metadata model (adapted from konveyor/apps/documents/models.py)."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique document identifier",
    )
    title: str = Field(..., description="Document title")
    filename: Optional[str] = Field(None, description="Original filename")
    file_path: str = Field(..., description="Full file path in repository")
    status: str = Field(default="pending", description="Processing status")
    content_type: Optional[str] = Field(None, description="MIME type of the document")
    size_bytes: Optional[int] = Field(None, ge=0, description="File size in bytes")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Creation timestamp",
    )
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    metadata: Optional[dict[str, Any]] = Field(
        None, description="Additional document metadata"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "id": "doc-abc123",
                "title": "Authentication Middleware",
                "filename": "auth_middleware.py",
                "file_path": "src/auth/middleware.py",
                "status": "processed",
                "content_type": "text/x-python",
                "size_bytes": 2048,
                "created_at": "2025-01-26T10:30:00Z",
                "metadata": {"language": "python", "lines": 85},
            }
        },
    )


# Original service communication models


class Snippet(BaseModel):
    """Code snippet with file path and content."""

    file_path: str = Field(..., description="Path to the source file")
    content: str = Field(..., description="Code snippet content")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "file_path": "src/auth/middleware.py",
                "content": "def authenticate_request(token: str) -> bool:\n    return validate_token(token)",
            }
        },
    )


class SearchRequest(BaseModel):
    """Request for searching code snippets."""

    query: str = Field(..., description="Search query text")
    k: int = Field(default=5, ge=1, le=20, description="Number of results to return")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {"query": "authentication middleware implementation", "k": 5}
        },
    )


class AdviceRequest(BaseModel):
    """Request for generating role-specific advice."""

    role: str = Field(
        ..., description="User role (e.g., 'backend_developer', 'security_engineer')"
    )
    chunks: list[Snippet] = Field(..., description="Code snippets for context")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "role": "backend_developer",
                "chunks": [
                    {
                        "file_path": "src/auth.py",
                        "content": "def authenticate(token): ...",
                    }
                ],
            }
        },
    )


class QueryRequest(BaseModel):
    """Request for querying the orchestrator."""

    question: str = Field(..., description="User question or query")
    role: str = Field(default="developer", description="User role for context")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "question": "How do I implement authentication middleware?",
                "role": "backend_developer",
            }
        },
    )


class AnswerResponse(BaseModel):
    """Response containing generated answer with sources."""

    answer: str = Field(..., description="Generated answer text")
    sources: list[str] = Field(
        default_factory=list, description="Source file references"
    )
    request_id: str = Field(..., description="Request ID for tracing")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "answer": "To implement authentication middleware, create a FastAPI middleware...",
                "sources": ["src/auth/middleware.py", "src/auth/tokens.py"],
                "request_id": "req-123",
            }
        },
    )
