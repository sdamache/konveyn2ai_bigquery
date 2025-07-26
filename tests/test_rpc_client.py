from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from src.common.models import JsonRpcError, JsonRpcResponse
from src.common.rpc_client import JsonRpcClient


@pytest.mark.asyncio
async def test_json_rpc_client_call_success():
    mock_response = {"jsonrpc": "2.0", "id": "1", "result": {"message": "success"}}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_response_obj = Mock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status.return_value = None
        mock_post.return_value = mock_response_obj

        client = JsonRpcClient(url="http://test.com")
        response = await client.call(method="test_method", params={}, id="1")

        assert isinstance(response, JsonRpcResponse)
        assert response.id == "1"
        assert response.result == {"message": "success"}
        assert response.error is None


@pytest.mark.asyncio
async def test_json_rpc_client_call_http_error():
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.side_effect = httpx.HTTPStatusError(
            "Error", request=Mock(), response=mock_response
        )

        client = JsonRpcClient(url="http://test.com")
        response = await client.call(method="test_method", params={}, id="1")

        assert isinstance(response, JsonRpcResponse)
        assert response.id == "1"
        assert response.result is None
        assert isinstance(response.error, JsonRpcError)
        assert response.error.code == -32000


@pytest.mark.asyncio
async def test_json_rpc_client_call_request_error():
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.RequestError("Connection failed")

        client = JsonRpcClient(url="http://test.com")
        response = await client.call(method="test_method", params={}, id="1")

        assert isinstance(response, JsonRpcResponse)
        assert response.id == "1"
        assert response.result is None
        assert isinstance(response.error, JsonRpcError)
        assert response.error.code == -32000


@pytest.mark.asyncio
async def test_json_rpc_client_auto_id_generation():
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # We need to capture the generated ID from the request
        mock_response_obj = Mock()
        mock_response_obj.status_code = 200
        mock_response_obj.raise_for_status.return_value = None

        def mock_json():
            # Extract ID from the request that was made
            call_args = mock_post.call_args
            request_data = call_args.kwargs["json"]
            return {
                "jsonrpc": "2.0",
                "id": request_data["id"],  # Use the same ID from request
                "result": {"status": "ok"},
            }

        mock_response_obj.json = mock_json
        mock_post.return_value = mock_response_obj

        client = JsonRpcClient(url="http://test.com")
        response = await client.call(method="test_method", params={})  # No ID provided

        assert isinstance(response, JsonRpcResponse)
        assert response.result == {"status": "ok"}
        assert response.id is not None  # Should have auto-generated ID


@pytest.mark.asyncio
async def test_json_rpc_client_custom_timeout():
    client = JsonRpcClient(url="http://test.com", timeout=60, max_retries=5)
    assert client.timeout == 60
    assert client.max_retries == 5
