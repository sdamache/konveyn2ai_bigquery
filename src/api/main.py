"""
Main FastAPI application for BigQuery Vector Store API.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:  # pragma: no cover - defensive patching
    from httpx import _content as _httpx_content
except Exception:  # pragma: no cover - httpx not installed or internal layout changed
    _httpx_content = None

from .schema_endpoints import router as schema_router
from .vector_endpoints import router as vector_router

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting BigQuery Vector Store API...")
    app.state.schema_manager_stub = None
    app.state.vector_store_stub = None
    yield
    logger.info("Shutting down BigQuery Vector Store API...")


# Create FastAPI application
app = FastAPI(
    title="BigQuery Vector Store API",
    description="REST API for BigQuery-based vector storage and similarity search",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.on_event("startup")
async def reset_stubs_on_startup() -> None:  # pragma: no cover - exercised via tests
    app.state.schema_manager_stub = None
    app.state.vector_store_stub = None


@app.on_event("shutdown")
async def clear_stubs_on_shutdown() -> None:  # pragma: no cover - exercised via tests
    app.state.schema_manager_stub = None
    app.state.vector_store_stub = None


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(vector_router)
app.include_router(schema_router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "BigQuery Vector Store API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Health"])
async def health():
    """Basic health check."""
    return {
        "status": "healthy",
        "service": "bigquery-vector-store-api",
        "version": "1.0.0",
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
# Ensure httpx TestClient can serialize NaN payloads used in contract tests.
if _httpx_content is not None and not getattr(
    _httpx_content, "konveyn_nan_patch", False
):  # pragma: no cover - exercised via contract suite
    _original_json_dumps = _httpx_content.json_dumps

    def _json_dumps_allow_nan(*args, **kwargs):
        kwargs["allow_nan"] = True
        return _original_json_dumps(*args, **kwargs)

    _httpx_content.json_dumps = _json_dumps_allow_nan
    _httpx_content.konveyn_nan_patch = True
