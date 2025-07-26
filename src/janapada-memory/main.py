"""
Janapada Memory Service - Semantic search agent using Vertex AI embeddings and Matching Engine.

This service implements the memory component of the KonveyN2AI three-tier architecture,
providing semantic search capabilities for code snippets using Google Cloud Vertex AI.

Adapted from konveyor/core/search/ with 60% code reuse and GCP platform migration.
"""

import logging
import os
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Import common models and JSON-RPC infrastructure
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.models import SearchRequest, Snippet, JsonRpcError, JsonRpcErrorCode
from common.rpc_server import JsonRpcServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Janapada Memory Service",
    description="Semantic search agent using Vertex AI embeddings and Matching Engine",
    version="1.0.0",
)

# Initialize JSON-RPC server
rpc_server = JsonRpcServer(
    title="Janapada Memory Service",
    version="1.0.0",
    description="Semantic search agent using Vertex AI embeddings and Matching Engine",
)

# Global variables for caching (will be initialized in startup)
embedding_model = None
matching_engine_index = None
faiss_index = None  # Fallback index


@app.on_event("startup")
async def startup_event():
    """Initialize Vertex AI components on startup."""
    global embedding_model, matching_engine_index

    logger.info("Starting Janapada Memory Service...")

    # Initialize Vertex AI (will be implemented in subtask 6.2)
    try:
        # TODO: Initialize Vertex AI in subtask 6.2
        # vertexai.init(project=os.environ["GOOGLE_CLOUD_PROJECT"], location="us-central1")
        # embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko-001")
        logger.info(
            "Vertex AI initialization placeholder - will be implemented in subtask 6.2"
        )
    except Exception as e:
        logger.warning(f"Vertex AI initialization failed: {e}")

    # Initialize Matching Engine (will be implemented in subtask 6.3)
    try:
        # TODO: Initialize Matching Engine in subtask 6.3
        # index_endpoint = aiplatform.MatchingEngineIndexEndpoint(os.environ["INDEX_ENDPOINT_ID"])
        # matching_engine_index = index_endpoint.get_index(os.environ["INDEX_ID"])
        logger.info(
            "Matching Engine initialization placeholder - will be implemented in subtask 6.3"
        )
    except Exception as e:
        logger.warning(f"Matching Engine initialization failed: {e}")

    # Initialize FAISS fallback (will be implemented in subtask 6.4)
    try:
        # TODO: Initialize FAISS fallback in subtask 6.4
        logger.info(
            "FAISS fallback initialization placeholder - will be implemented in subtask 6.4"
        )
    except Exception as e:
        logger.warning(f"FAISS initialization failed: {e}")

    logger.info("Janapada Memory Service startup complete")


@rpc_server.method("search", "Search for relevant code snippets using semantic search")
def search_snippets(
    query: str, k: int = 5, request_id: Optional[str] = None
) -> dict[str, Any]:
    """
    Search for relevant code snippets using semantic search.

    Adapted from konveyor/core/search/providers/azure_search.py:82-146
    Platform migration: Azure AI Search → Vertex AI Matching Engine

    Args:
        query: Search query text
        k: Number of results to return (1-20)
        request_id: Optional request ID for tracing

    Returns:
        Dictionary containing list of snippets with file_path and content

    Raises:
        JsonRpcError: If search fails or parameters are invalid
    """
    logger.info(
        f"Search request: query='{query[:50]}...', k={k}, request_id={request_id}"
    )

    # Validate parameters
    if not query or not query.strip():
        raise ValueError("Query parameter cannot be empty")

    if k < 1 or k > 20:
        raise ValueError(f"Parameter k must be between 1 and 20, got {k}")

    try:
        # TODO: Implement actual search in subsequent subtasks
        # For now, return mock data to complete subtask 6.1
        mock_snippets = [
            Snippet(
                file_path="src/auth/middleware.py",
                content="def authenticate_request(token: str) -> bool:\n    return validate_token(token)",
            ),
            Snippet(
                file_path="src/auth/tokens.py",
                content="def validate_token(token: str) -> bool:\n    # Token validation logic\n    return True",
            ),
        ]

        # Limit results to requested k
        result_snippets = mock_snippets[:k]

        logger.info(f"Search completed: returned {len(result_snippets)} snippets")

        return {"snippets": [snippet.model_dump() for snippet in result_snippets]}

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise RuntimeError(f"Search operation failed: {str(e)}")


# JSON-RPC endpoint
@app.post("/")
async def json_rpc_endpoint(request: Request):
    """Handle JSON-RPC requests."""
    return await rpc_server.handle_request(request)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check if core components are available
    health_status = {
        "status": "healthy",
        "service": "janapada-memory",
        "version": "1.0.0",
        "components": {
            "embedding_model": "placeholder" if embedding_model is None else "ready",
            "matching_engine": "placeholder"
            if matching_engine_index is None
            else "ready",
            "faiss_fallback": "placeholder" if faiss_index is None else "ready",
        },
    }

    # Determine overall status
    if all(status != "ready" for status in health_status["components"].values()):
        health_status["status"] = "starting"

    return health_status


@app.get("/.well-known/agent.json")
async def agent_manifest():
    """Agent manifest endpoint for service discovery."""
    return rpc_server.get_manifest()


# Add middleware for logging
rpc_server.add_middleware(
    lambda rpc_request, http_request: logger.info(
        f"JSON-RPC call: {rpc_request.method} (id: {rpc_request.id})"
    )
)

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)  # nosec B104
