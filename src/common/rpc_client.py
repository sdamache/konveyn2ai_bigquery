import httpx
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import JsonRpcRequest, JsonRpcResponse, JsonRpcError

class JsonRpcClient:
    def __init__(self, url: str, timeout: int = 30, max_retries: int = 3):
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries

    @asynccontextmanager
    async def session(self) -> Any:
        async with httpx.AsyncClient() as client:
            yield client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def call(self, method: str, params: Dict[str, Any], id: Optional[Union[str, int]] = None) -> JsonRpcResponse:
        if id is None:
            id = str(uuid.uuid4())

        request = JsonRpcRequest(id=str(id) if id is not None else None, method=method, params=params)

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
                error = JsonRpcError(code=-32000, message=f"HTTP Error: {e.response.status_code}", data={"reason": str(e)})
                return JsonRpcResponse(id=id, error=error)
            except httpx.RequestError as e:
                error = JsonRpcError(code=-32000, message=f"Request Error: {e}", data={"reason": str(e)})
                return JsonRpcResponse(id=id, error=error)
