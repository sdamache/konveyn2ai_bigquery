"""
Janapada Memory Service - FastAPI application for BigQuery vector search.

This service provides vector search capabilities using BigQuery as the backend
with fallback to local search when needed. It implements JSON-RPC 2.0 protocol
for communication with other KonveyN2AI components.
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from pydantic_core import ErrorDetails

# Add path for common modules
current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, "..")
sys.path.insert(0, src_dir)

from common.rpc_server import JsonRpcServer
from guard_fort import GuardFort

# Import service modules
try:
    from .memory_service import JanapadaMemoryService
    from .connections.bigquery_connection import BigQueryConnectionManager
except ImportError:
    # Fallback to absolute imports when run directly
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))
    from memory_service import JanapadaMemoryService
    from connections.bigquery_connection import BigQueryConnectionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Application constants
APP_VERSION = "1.0.0"
SERVICE_NAME = "janapada-memory"
MAX_RESULTS_LIMIT = 100

# Global variables for service components
memory_service: JanapadaMemoryService = None
config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global memory_service, config

    logger.info("Starting Janapada Memory service...")

    try:
        # Initialize configuration
        from config.bigquery_config import BigQueryConfigManager

        config = BigQueryConfigManager.load_from_environment()
        logger.info("Configuration loaded successfully")

        # Initialize BigQuery connection
        connection = BigQueryConnectionManager(config=config)

        # Initialize memory service
        memory_service = JanapadaMemoryService(
            connection=connection, enable_fallback=True, fallback_max_entries=1000
        )
        logger.info("Memory service initialized successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    finally:
        # Cleanup
        if memory_service:
            memory_service.close()
        logger.info("Janapada Memory service stopped")


# Create FastAPI application
app = FastAPI(
    title="Janapada Memory Service",
    description="BigQuery vector search with fallback capabilities",
    version=APP_VERSION,
    lifespan=lifespan,
)

# Add CORS middleware
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize GuardFort security middleware
guard_fort = GuardFort(
    app,
    service_name=SERVICE_NAME,
    enable_auth=True,
    allowed_paths=["/health", "/", "/docs", "/openapi.json", "/.well-known/agent.json"],
)

# Create JSON-RPC server
rpc_server = JsonRpcServer(
    title="Janapada Memory Service",
    version=APP_VERSION,
    description="BigQuery vector search with fallback capabilities",
)


@rpc_server.method("search", "Search for similar vectors using BigQuery")
async def search(
    query_embedding: List[float],
    k: int = 10,
    artifact_types: Optional[List[str]] = None,
    request_id: str = None,
) -> dict:
    """
    Search for similar vectors using BigQuery vector search.

    Args:
        query_embedding: Query vector to search for
        k: Number of results to return (max 100)
        artifact_types: Optional filter by artifact types
        request_id: Optional request ID for tracking

    Returns:
        dict: Search results with metadata

    Raises:
        ValueError: If input validation fails
        RuntimeError: If service is unavailable
    """
    start_time = time.time()

    try:
        # Input validation
        if not query_embedding or not isinstance(query_embedding, list):
            raise ValidationError.from_exception_data(
                "ValidationError",
                [
                    ErrorDetails(
                        type="value_error",
                        loc=("query_embedding",),
                        msg="Query embedding must be a non-empty list of numbers",
                        input=query_embedding,
                    )
                ],
            )

        if k <= 0 or k > MAX_RESULTS_LIMIT:
            raise ValidationError.from_exception_data(
                "ValidationError",
                [
                    ErrorDetails(
                        type="value_error",
                        loc=("k",),
                        msg=f"k must be between 1 and {MAX_RESULTS_LIMIT}",
                        input=k,
                    )
                ],
            )

        logger.info(
            f"Processing search request with {len(query_embedding)} dimensions, k={k} (request_id: {request_id})"
        )

        # Check if memory service is available
        if memory_service is None:
            raise RuntimeError("Memory service not initialized")

        # Perform similarity search
        results = memory_service.similarity_search(
            query_embedding=query_embedding, k=k, artifact_types=artifact_types
        )

        # Performance logging
        elapsed_time = time.time() - start_time
        logger.info(
            f"Search completed: {len(results)} results in {elapsed_time:.2f}s (request_id: {request_id})"
        )

        return {
            "results": [result.to_dict() for result in results],
            "metadata": {
                "total_results": len(results),
                "processing_time": round(elapsed_time, 2),
                "request_id": request_id,
                "embedding_dimensions": len(query_embedding),
                "k_requested": k,
            },
        }

    except ValueError as e:
        logger.warning(f"Input validation error: {e} (request_id: {request_id})")
        raise
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(
            f"Error performing search after {elapsed_time:.2f}s: {e} (request_id: {request_id})"
        )
        raise


# JSON-RPC endpoint
@app.post("/")
async def json_rpc_endpoint(request: Request):
    """Main JSON-RPC 2.0 endpoint."""
    return await rpc_server.handle_request(request)


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint for service monitoring.

    Returns detailed status including BigQuery connectivity,
    fallback status, and performance metrics.
    """
    try:
        if memory_service:
            # Check service health
            health_status = memory_service.health_check()

            if health_status.get("status") == "healthy":
                return {
                    "status": "healthy",
                    "service": SERVICE_NAME,
                    "version": APP_VERSION,
                    "bigquery_status": health_status.get(
                        "bigquery_connection", "unknown"
                    ),
                    "fallback_status": health_status.get("fallback_index", "unknown"),
                    "configuration": {
                        "dataset_id": config.dataset_id if config else "unknown",
                        "project_id": config.project_id if config else "unknown",
                    },
                }
            else:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "service": SERVICE_NAME,
                        "version": APP_VERSION,
                        "error": health_status.get("error", "Service not ready"),
                    },
                )
        else:
            # Service not initialized yet
            return {
                "status": "starting",
                "service": SERVICE_NAME,
                "version": APP_VERSION,
                "message": "Service is starting up",
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": SERVICE_NAME,
                "version": APP_VERSION,
                "error": str(e),
            },
        )


# Agent manifest endpoint
@app.get("/.well-known/agent.json")
async def agent_manifest():
    """Agent discovery manifest endpoint."""
    return {
        "name": "Janapada Memory Service",
        "description": "BigQuery vector search with fallback capabilities",
        "version": APP_VERSION,
        "methods": [
            {
                "name": "search",
                "description": "Search for similar vectors using BigQuery",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_embedding": {
                            "type": "array",
                            "description": "Query vector to search for",
                            "items": {"type": "number"},
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "minimum": 1,
                            "maximum": MAX_RESULTS_LIMIT,
                            "default": 10,
                        },
                        "artifact_types": {
                            "type": "array",
                            "description": "Optional filter by artifact types",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["query_embedding"],
                },
            }
        ],
        "endpoints": {
            "rpc": "/",
            "health": "/health",
            "manifest": "/.well-known/agent.json",
        },
    }


# Development server
if __name__ == "__main__":
    # Load environment configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))

    logger.info(f"Starting development server on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True, log_level="info")
