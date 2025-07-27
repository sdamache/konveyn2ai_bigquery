"""
Unit tests for JSON-RPC models and shared data models.

Tests validation, serialization/deserialization, and error handling
for all Pydantic models in the common module.
"""

import json

# Clean import pattern using centralized utilities - no sys.path needed with PYTHONPATH=src
import pytest  # noqa: E402
from common.models import (  # noqa: E402
    AdviceRequest,
    AnswerResponse,
    Document,
    DocumentChunk,
    JsonRpcError,
    JsonRpcErrorCode,
    JsonRpcRequest,
    JsonRpcResponse,
    QueryRequest,
    SearchQuery,
    SearchRequest,
    SearchResult,
    SearchType,
    Snippet,
)
from pydantic import ValidationError  # noqa: E402


class TestJsonRpcError:
    """Test JSON-RPC error model."""

    def test_valid_error_creation(self):
        """Test creating valid JSON-RPC errors."""
        error = JsonRpcError(
            code=JsonRpcErrorCode.INVALID_PARAMS,
            message="Invalid parameters",
            data={"field": "query", "reason": "missing"},
        )

        assert error.code == -32602
        assert error.message == "Invalid parameters"
        assert error.data["field"] == "query"

    def test_error_without_data(self):
        """Test error creation without optional data field."""
        error = JsonRpcError(
            code=JsonRpcErrorCode.METHOD_NOT_FOUND, message="Method not found"
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
            id="req-123", method="search", params={"query": "test", "k": 5}
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
            id="req-789", method="advise", params={"role": "developer", "chunks": []}
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
            JsonRpcRequest(jsonrpc="1.0", id="req-123", method="test")

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
            request_id="req-123", result={"data": "test result"}
        )

        assert response.jsonrpc == "2.0"
        assert response.id == "req-123"
        assert response.result["data"] == "test result"
        assert response.error is None

    def test_error_response(self):
        """Test creating error responses."""
        error = JsonRpcError(
            code=JsonRpcErrorCode.INTERNAL_ERROR, message="Internal server error"
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
            result={"snippets": [{"file": "test.py", "content": "code"}]},
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
            JsonRpcResponse(id="req-123", result={"data": "test"}, error=error)

    def test_missing_result_and_error(self):
        """Test validation that requires either result or error."""
        with pytest.raises(ValueError):
            JsonRpcResponse(id="req-123")


class TestSharedDataModels:
    """Test shared data models for service communication."""

    def test_snippet_model(self):
        """Test Snippet model validation."""
        snippet = Snippet(file_path="src/test.py", content="def test(): pass")

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
            Snippet(file_path="middleware.py", content="class Middleware: pass"),
        ]

        request = AdviceRequest(role="backend_developer", chunks=snippets)

        assert request.role == "backend_developer"
        assert len(request.chunks) == 2
        assert request.chunks[0].file_path == "auth.py"

    def test_query_request_model(self):
        """Test QueryRequest model validation."""
        # With explicit role
        request = QueryRequest(
            question="How to implement auth?", role="security_engineer"
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
            request_id="req-123",
        )

        assert "authentication" in response.answer
        assert len(response.sources) == 2
        assert response.request_id == "req-123"

        # With empty sources (default)
        response = AnswerResponse(answer="Simple answer", request_id="req-456")
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
                    {"file_path": "auth.py", "content": "auth code"},
                ],
            },
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
            JsonRpcRequest(id="req-123", method="test", extra_field="not allowed")

        with pytest.raises(ValidationError):
            Snippet(file_path="test.py", content="code", extra_field="not allowed")


class TestEnhancedVectorSearchModels:
    """Test enhanced vector search models adapted from konveyor integration."""

    def test_search_type_enum(self):
        """Test SearchType enumeration."""
        assert SearchType.VECTOR == "vector"
        assert SearchType.KEYWORD == "keyword"
        assert SearchType.HYBRID == "hybrid"
        assert SearchType.SEMANTIC == "semantic"

    def test_search_query_model(self):
        """Test SearchQuery model with embedding support."""
        # Basic text search
        query = SearchQuery(text="authentication middleware")
        assert query.text == "authentication middleware"
        assert query.embedding is None
        assert query.search_type == SearchType.HYBRID
        assert query.top_k == 5
        assert query.filters is None

        # Vector search with embedding
        embedding = [0.1] * 3072  # 3072-dimensional vector
        query = SearchQuery(
            text="test query",
            embedding=embedding,
            search_type=SearchType.VECTOR,
            top_k=10,
            filters={"file_type": "python"},
        )
        assert query.embedding == embedding
        assert query.search_type == SearchType.VECTOR
        assert query.top_k == 10
        assert query.filters == {"file_type": "python"}

        # Test validation
        with pytest.raises(ValidationError):
            SearchQuery(text="test", top_k=0)  # Invalid top_k

        with pytest.raises(ValidationError):
            SearchQuery(text="test", top_k=25)  # Invalid top_k

    def test_search_result_model(self):
        """Test SearchResult model."""
        result = SearchResult(
            id="result-123",
            content="def authenticate(): pass",
            score=0.85,
            metadata={"language": "python", "file_path": "src/auth.py"},
            chunk_id="chunk-123",
            document_id="doc-456",
            chunk_index=0,
        )

        assert result.id == "result-123"
        assert result.content == "def authenticate(): pass"
        assert result.score == 0.85
        assert result.chunk_id == "chunk-123"
        assert result.document_id == "doc-456"
        assert result.chunk_index == 0
        assert result.metadata == {"language": "python", "file_path": "src/auth.py"}

        # Test score validation
        with pytest.raises(ValidationError):
            SearchResult(id="test", content="test", score=-0.1)  # Invalid score

        with pytest.raises(ValidationError):
            SearchResult(id="test", content="test", score=1.1)  # Invalid score

    def test_document_chunk_model(self):
        """Test DocumentChunk model."""
        chunk = DocumentChunk(
            content="class TestClass: pass",
            document_id="doc-123",
            chunk_index=0,
            metadata={"start_line": 1, "end_line": 1},
        )

        assert chunk.content == "class TestClass: pass"
        assert chunk.document_id == "doc-123"
        assert chunk.chunk_index == 0
        assert chunk.metadata == {"start_line": 1, "end_line": 1}
        assert chunk.id is not None  # UUID should be generated
        assert chunk.embedding is None

        # Test with embedding
        embedding = [0.2] * 3072
        chunk_with_embedding = DocumentChunk(
            content="test content",
            document_id="doc-456",
            chunk_index=1,
            embedding=embedding,
        )
        assert chunk_with_embedding.embedding == embedding

        # Test validation
        with pytest.raises(ValidationError):
            DocumentChunk(
                content="test", document_id="doc", chunk_index=-1
            )  # Invalid index

    def test_document_model(self):
        """Test Document model."""
        doc = Document(
            title="Test Document",
            file_path="src/test.py",
            filename="test.py",
            status="processed",
            content_type="text/x-python",
            size_bytes=1024,
            metadata={"language": "python", "lines": 50},
        )

        assert doc.title == "Test Document"
        assert doc.file_path == "src/test.py"
        assert doc.filename == "test.py"
        assert doc.status == "processed"
        assert doc.content_type == "text/x-python"
        assert doc.size_bytes == 1024
        assert doc.metadata == {"language": "python", "lines": 50}
        assert doc.id is not None  # UUID should be generated
        assert doc.created_at is not None  # Timestamp should be generated

        # Test with minimal fields
        minimal_doc = Document(title="Minimal", file_path="minimal.py")
        assert minimal_doc.status == "pending"  # Default status
        assert minimal_doc.filename is None
        assert minimal_doc.updated_at is None

        # Test validation
        with pytest.raises(ValidationError):
            Document(title="test", file_path="test.py", size_bytes=-1)  # Invalid size


class TestEnhancedModelIntegration:
    """Test integration between enhanced models and existing models."""

    def test_search_query_with_json_rpc(self):
        """Test SearchQuery integration with JSON-RPC protocol."""
        search_params = {
            "text": "authentication middleware",
            "search_type": "hybrid",
            "top_k": 5,
        }

        request = JsonRpcRequest(
            id="req-search-123", method="search", params=search_params
        )

        # Verify the request can be created and serialized
        json_str = request.to_json()
        assert "authentication middleware" in json_str
        assert "hybrid" in json_str

    def test_search_result_serialization(self):
        """Test SearchResult serialization for JSON-RPC responses."""
        results = [
            SearchResult(
                id="result-1",
                content="def auth(): pass",
                score=0.9,
                metadata={"file_path": "auth.py"},
            ),
            SearchResult(
                id="result-2",
                content="class Middleware: pass",
                score=0.8,
                metadata={"file_path": "middleware.py"},
            ),
        ]

        response = JsonRpcResponse.create_success(
            request_id="req-123", result={"results": [r.model_dump() for r in results]}
        )

        # Verify serialization works
        json_str = response.to_json()
        assert "def auth(): pass" in json_str
        assert "class Middleware: pass" in json_str
