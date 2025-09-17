"""Shared helpers for shaping API responses."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi.responses import JSONResponse


def utc_now_iso() -> str:
    """Return current UTC timestamp with Z suffix."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def error_response(
    *,
    status_code: int,
    error: str,
    message: str,
    details: Optional[dict[str, Any]] = None,
) -> JSONResponse:
    """Construct a JSON error response that matches the API contract."""

    payload: dict[str, Any] = {
        "error": error,
        "message": message,
        "timestamp": utc_now_iso(),
    }
    if details:
        payload["details"] = details
    return JSONResponse(status_code=status_code, content=payload)
