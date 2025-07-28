"""
Tests for JSON-RPC 2.0 Server implementation.
"""

import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import Request

from src.common.models import JsonRpcErrorCode, JsonRpcRequest, JsonRpcResponse
from src.common.rpc_server import (
    JsonRpcMethodRegistry,
    JsonRpcServer,
    create_error_response,
    create_success_response,
    get_global_server,
    logging_middleware,
    rpc_method,
)


class TestJsonRpcMethodRegistry:
    """Test the method registry functionality."""

    def test_method_registration(self):
        registry = JsonRpcMethodRegistry()

        def test_handler():
            return "test"

        registry.register("test_method", test_handler, "Test method")

        assert registry.get_handler("test_method") == test_handler
        methods = registry.get_methods()
        assert "test_method" in methods
        assert methods["test_method"]["description"] == "Test method"

    def test_duplicate_method_registration(self):
        registry = JsonRpcMethodRegistry()

        def handler1():
            pass

        def handler2():
            pass

        registry.register("test", handler1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", handler2)

    def test_reserved_method_names(self):
        registry = JsonRpcMethodRegistry()

        def handler():
            pass

        with pytest.raises(ValueError, match="reserved"):
            registry.register("rpc.test", handler)

    def test_method_schema_extraction(self):
        registry = JsonRpcMethodRegistry()

        def handler(param1: str, param2: int = 10, param3=None):
            """Test handler docstring."""
            pass

        registry.register("test", handler, "Custom description")
        methods = registry.get_methods()

        assert methods["test"]["description"] == "Custom description"
        params = methods["test"]["parameters"]
        assert params["param1"]["required"] is True
        assert params["param2"]["required"] is False
        assert "param3" in params


class TestJsonRpcServer:
    """Test the JSON-RPC server functionality."""

    @pytest.fixture
    def server(self):
        return JsonRpcServer("Test Server", "1.0.0")

    @pytest.fixture
    def mock_request(self):
        request = Mock(spec=Request)
        request.body = AsyncMock()
        return request

    def test_server_initialization(self, server):
        assert server.title == "Test Server"
        assert server.version == "1.0.0"
        assert isinstance(server.registry, JsonRpcMethodRegistry)

    def test_method_decorator(self, server):
        @server.method("test_method", "Test description")
        def test_handler():
            return {"result": "success"}

        handler = server.registry.get_handler("test_method")
        assert handler == test_handler

        methods = server.registry.get_methods()
        assert "test_method" in methods

    @pytest.mark.asyncio
    async def test_valid_request_handling(self, server, mock_request):
        @server.method("add")
        def add_handler(a: int, b: int):
            return {"sum": a + b}

        request_data = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "add",
            "params": {"a": 5, "b": 3},
        }

        mock_request.body.return_value = json.dumps(request_data).encode()

        response = await server.handle_request(mock_request)
        response_data = json.loads(response.body)

        assert response_data["jsonrpc"] == "2.0"
        assert response_data["id"] == "1"
        assert response_data["result"]["sum"] == 8
        assert response_data.get("error") is None

    @pytest.mark.asyncio
    async def test_method_not_found(self, server, mock_request):
        request_data = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "nonexistent",
            "params": {},
        }

        mock_request.body.return_value = json.dumps(request_data).encode()

        response = await server.handle_request(mock_request)
        response_data = json.loads(response.body)

        assert response_data["error"]["code"] == JsonRpcErrorCode.METHOD_NOT_FOUND
        assert "nonexistent" in response_data["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_params(self, server, mock_request):
        @server.method("strict_method")
        def strict_handler(required_param: str):
            return {"param": required_param}

        request_data = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "strict_method",
            "params": {"wrong_param": "value"},
        }

        mock_request.body.return_value = json.dumps(request_data).encode()

        response = await server.handle_request(mock_request)
        response_data = json.loads(response.body)

        assert response_data["error"]["code"] == JsonRpcErrorCode.INVALID_PARAMS

    @pytest.mark.asyncio
    async def test_parse_error(self, server, mock_request):
        mock_request.body.return_value = b"invalid json"

        response = await server.handle_request(mock_request)
        response_data = json.loads(response.body)

        assert response_data["error"]["code"] == JsonRpcErrorCode.PARSE_ERROR

    @pytest.mark.asyncio
    async def test_batch_requests(self, server, mock_request):
        @server.method("multiply")
        def multiply_handler(a: int, b: int):
            return {"product": a * b}

        batch_data = [
            {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "multiply",
                "params": {"a": 2, "b": 3},
            },
            {
                "jsonrpc": "2.0",
                "id": "2",
                "method": "multiply",
                "params": {"a": 4, "b": 5},
            },
        ]

        mock_request.body.return_value = json.dumps(batch_data).encode()

        response = await server.handle_request(mock_request)
        response_data = json.loads(response.body)

        assert isinstance(response_data, list)
        assert len(response_data) == 2
        assert response_data[0]["result"]["product"] == 6
        assert response_data[1]["result"]["product"] == 20

    @pytest.mark.asyncio
    async def test_empty_batch_request(self, server, mock_request):
        mock_request.body.return_value = json.dumps([]).encode()

        response = await server.handle_request(mock_request)
        response_data = json.loads(response.body)

        assert response_data["error"]["code"] == JsonRpcErrorCode.INVALID_REQUEST
        assert "Empty batch" in response_data["error"]["message"]

    @pytest.mark.asyncio
    async def test_notification_request(self, server, mock_request):
        @server.method("notify")
        def notify_handler(message: str):
            return {"received": message}

        request_data = {
            "jsonrpc": "2.0",
            "method": "notify",
            "params": {"message": "hello"},
            # No 'id' field = notification
        }

        mock_request.body.return_value = json.dumps(request_data).encode()

        response = await server.handle_request(mock_request)

        # Notifications should return 204 No Content
        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_handler_exception(self, server, mock_request):
        @server.method("error_method")
        def error_handler():
            raise ValueError("Something went wrong")

        request_data = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "error_method",
            "params": {},
        }

        mock_request.body.return_value = json.dumps(request_data).encode()

        response = await server.handle_request(mock_request)
        response_data = json.loads(response.body)

        assert response_data["error"]["code"] == JsonRpcErrorCode.INTERNAL_ERROR
        assert "Something went wrong" in response_data["error"]["message"]

    @pytest.mark.asyncio
    async def test_context_injection(self, server, mock_request):
        @server.method("context_method")
        def context_handler(request_id: str, context: dict):
            return {"request_id": request_id, "has_context": "http_request" in context}

        request_data = {
            "jsonrpc": "2.0",
            "id": "test-123",
            "method": "context_method",
            "params": {},
        }

        mock_request.body.return_value = json.dumps(request_data).encode()

        response = await server.handle_request(mock_request)
        response_data = json.loads(response.body)

        assert response_data["result"]["request_id"] == "test-123"
        assert response_data["result"]["has_context"] is True

    @pytest.mark.asyncio
    async def test_middleware(self, server, mock_request):
        middleware_called = False

        async def test_middleware(rpc_request, http_request):
            nonlocal middleware_called
            middleware_called = True
            assert isinstance(rpc_request, JsonRpcRequest)

        server.add_middleware(test_middleware)

        @server.method("test")
        def test_handler():
            return {"status": "ok"}

        request_data = {"jsonrpc": "2.0", "id": "1", "method": "test", "params": {}}

        mock_request.body.return_value = json.dumps(request_data).encode()

        await server.handle_request(mock_request)
        assert middleware_called is True

    def test_manifest_generation(self, server):
        @server.method("test_method", "A test method")
        def test_handler(param1: str, param2: int = 42):
            return {}

        manifest = server.get_manifest()

        assert manifest["name"] == "Test Server"
        assert manifest["version"] == "1.0.0"
        assert manifest["protocol"] == "json-rpc-2.0"
        assert "test_method" in manifest["methods"]

        # Check capabilities in the new enhanced format
        capability_names = [cap["name"] for cap in manifest["capabilities"]]
        assert "single-requests" in capability_names
        assert "batch-requests" in capability_names


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_success_response(self):
        response = create_success_response("123", {"data": "test"})

        assert isinstance(response, JsonRpcResponse)
        assert response.id == "123"
        assert response.result == {"data": "test"}
        assert response.error is None

    def test_create_error_response(self):
        response = create_error_response(
            "123", -32000, "Test error", {"detail": "info"}
        )

        assert isinstance(response, JsonRpcResponse)
        assert response.id == "123"
        assert response.result is None
        assert response.error.code == -32000
        assert response.error.message == "Test error"
        assert response.error.data == {"detail": "info"}

    @pytest.mark.asyncio
    async def test_logging_middleware(self):
        request = JsonRpcRequest(id="test", method="test_method", params={})
        http_request = Mock(spec=Request)

        # Should not raise any exceptions
        await logging_middleware(request, http_request)


class TestGlobalServerDecorator:
    """Test global server decorator functionality."""

    def test_rpc_method_decorator(self):
        @rpc_method("global_test", "Global test method")
        def global_handler():
            return {"global": True}

        server = get_global_server()
        handler = server.registry.get_handler("global_test")

        assert handler == global_handler

    def test_get_global_server(self):
        server1 = get_global_server()
        server2 = get_global_server()

        # Should return the same instance
        assert server1 is server2
