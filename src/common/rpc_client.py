import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional, Union

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import JsonRpcError, JsonRpcRequest, JsonRpcResponse

# Connection pool configuration constants
DEFAULT_MAX_CONNECTIONS = 100
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 20
DEFAULT_KEEPALIVE_EXPIRY = 5.0


class JsonRpcClient:
    def __init__(
        self,
        url: str,
        timeout: int = 30,
        max_retries: int = 3,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_keepalive_connections: int = DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry: float = DEFAULT_KEEPALIVE_EXPIRY,
    ):
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries

        # Connection pool configuration
        self._limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry,
        )

        # Persistent HTTP client with connection pooling
        self._client: Optional[httpx.AsyncClient] = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure the HTTP client is initialized with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                limits=self._limits,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self):
        """Close the HTTP client and cleanup connections."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @asynccontextmanager
    async def session(self) -> Any:
        """Context manager for HTTP client session with connection pooling."""
        client = await self._ensure_client()
        yield client

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def call(
        self, method: str, params: dict[str, Any], id: Optional[Union[str, int]] = None
    ) -> JsonRpcResponse:
        if id is None:
            id = str(uuid.uuid4())

        request = JsonRpcRequest(
            id=str(id) if id is not None else None, method=method, params=params
        )

        async with self.session() as client:
            try:
                response = await client.post(
                    self.url,
                    json=request.model_dump(by_alias=True),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return JsonRpcResponse(**response.json())
            except httpx.HTTPStatusError as e:
                error = JsonRpcError(
                    code=-32000,
                    message=f"HTTP Error: {e.response.status_code}",
                    data={"reason": str(e)},
                )
                return JsonRpcResponse(id=id, error=error)
            except httpx.RequestError as e:
                error = JsonRpcError(
                    code=-32000, message=f"Request Error: {e}", data={"reason": str(e)}
                )
                return JsonRpcResponse(id=id, error=error)
