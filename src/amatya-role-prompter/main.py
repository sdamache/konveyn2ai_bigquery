"""
Amatya Role Prompter - FastAPI application for LLM-powered advice generation.

This service provides role-specific prompting and advice generation using
Google Cloud Vertex AI text models. It implements JSON-RPC 2.0 protocol
for communication with other KonveyN2AI components.
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from pydantic_core import ErrorDetails

# Add path for common modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from common.models import AdviceRequest, Snippet
from common.rpc_server import JsonRpcServer
from guard_fort import GuardFort

# Import service modules
try:
    from .advisor import AdvisorService
    from .config import AmataConfig
except ImportError:
    from advisor import AdvisorService

    from config import AmataConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for service components
advisor_service: AdvisorService = None
config: AmataConfig = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global advisor_service, config

    logger.info("Starting Amatya Role Prompter service...")

    try:
        # Initialize configuration
        config = AmataConfig()
        logger.info("Configuration loaded successfully")

        # Configure CORS middleware with environment-specific settings
        global cors_origins
        cors_origins = config.cors_origins if config.cors_origins else ["*"]
        logger.info(f"CORS origins determined: {cors_origins}")

        # Initialize advisor service
        advisor_service = AdvisorService(config)
        await advisor_service.initialize()
        logger.info("Advisor service initialized successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    finally:
        # Cleanup
        if advisor_service:
            await advisor_service.cleanup()
        logger.info("Amatya Role Prompter service stopped")


# Create FastAPI application
app = FastAPI(
    title="Amatya Role Prompter",
    description="LLM-powered advice generation with role-specific prompting",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware with default configuration
# CORS origins will be configured based on environment in lifespan
cors_origins = ["*"]  # Default configuration
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
    service_name="amatya-role-prompter",
    enable_auth=True,
    allowed_paths=["/health", "/", "/docs", "/openapi.json", "/.well-known/agent.json"],
)

# Create JSON-RPC server
rpc_server = JsonRpcServer(
    title="Amatya Role Prompter",
    version="1.0.0",
    description="LLM-powered advice generation with role-specific prompting",
)


@rpc_server.method("advise", "Generate role-specific advice based on code snippets")
async def advise(
    role: str, question: str, chunks: list[dict], request_id: str = None
) -> dict:
    """
    Generate role-specific advice based on provided code snippets.

    Enhanced with input validation, performance monitoring, and comprehensive
    error handling for production use.

    Args:
        role: User role (e.g., 'backend_developer', 'security_engineer')
        question: User's specific question to be answered
        chunks: List of code snippets with file_path and content
        request_id: Optional request ID for tracking

    Returns:
        dict: Response containing generated advice

    Raises:
        ValueError: If input validation fails
        RuntimeError: If service is unavailable
    """
    start_time = time.time()

    try:
        # Input validation with proper JSON-RPC error codes
        if not role or not isinstance(role, str):
            # Create a proper ValidationError for JSON-RPC INVALID_PARAMS
            raise ValidationError.from_exception_data(
                "ValidationError",
                [
                    ErrorDetails(
                        type="value_error",
                        loc=("role",),
                        msg="Role must be a non-empty string",
                        input=role,
                    )
                ],
            )

        if chunks is None or not isinstance(chunks, list):
            # Create a proper ValidationError for JSON-RPC INVALID_PARAMS
            raise ValidationError.from_exception_data(
                "ValidationError",
                [
                    ErrorDetails(
                        type="value_error",
                        loc=("chunks",),
                        msg="Chunks must be a list",
                        input=chunks,
                    )
                ],
            )

        # Handle empty chunks gracefully - provide general advice
        if len(chunks) == 0:
            logger.info(
                f"Providing general advice for role '{role}' with no specific code context (request_id: {request_id})"
            )
            return {
                "advice": f"Welcome! As a {role}, here's some general guidance to get you started:\n\n"
                "• Review the project documentation and architecture\n"
                "• Understand the codebase structure and patterns\n"
                "• Set up your development environment\n"
                "• Run existing tests to ensure everything works\n"
                "• Start with small, incremental changes\n\n"
                "For more specific guidance, please provide code snippets for analysis.",
                "metadata": {
                    "role": role,
                    "chunks_processed": 0,
                    "processing_time": round(time.time() - start_time, 2),
                    "request_id": request_id,
                    "type": "general_guidance",
                },
            }

        if len(chunks) > 50:  # Reasonable limit for performance
            raise ValueError("Too many chunks provided (max 50)")

        # Convert chunks to Snippet objects with validation
        snippet_objects = []
        for i, chunk in enumerate(chunks):
            try:
                snippet_objects.append(Snippet(**chunk))
            except Exception as e:
                raise ValueError(f"Invalid chunk at index {i}: {e}") from e

        # Create advice request
        advice_request = AdviceRequest(
            role=role, question=question, chunks=snippet_objects
        )

        logger.info(
            f"Processing advice request for role '{role}' with {len(chunks)} chunks (request_id: {request_id})"
        )

        # Check if advisor service is available
        if advisor_service is None:
            # Create a temporary advisor service for testing
            from advisor import AdvisorService

            from config import AmataConfig

            temp_config = AmataConfig()
            temp_advisor = AdvisorService(temp_config)
            await temp_advisor.initialize()

            advice = await temp_advisor.generate_advice(advice_request)
            await temp_advisor.cleanup()
        else:
            # Use the global advisor service
            advice = await advisor_service.generate_advice(advice_request)

        # Performance logging
        elapsed_time = time.time() - start_time
        logger.info(
            f"Generated advice for role '{role}' with {len(chunks)} chunks in {elapsed_time:.2f}s (request_id: {request_id})"
        )

        return {
            "advice": advice,
            "metadata": {
                "role": role,
                "chunks_processed": len(chunks),
                "processing_time": round(elapsed_time, 2),
                "request_id": request_id,
            },
        }

    except ValueError as e:
        logger.warning(f"Input validation error: {e} (request_id: {request_id})")
        raise
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(
            f"Error generating advice after {elapsed_time:.2f}s: {e} (request_id: {request_id})"
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
    Enhanced health check endpoint for comprehensive service monitoring.

    Returns detailed status including AI service availability, configuration,
    performance metrics, and cache statistics for production monitoring.
    """
    try:
        # Check if advisor service is ready
        if advisor_service:
            is_healthy = await advisor_service.is_healthy()
            if is_healthy:
                # Determine active AI service
                ai_services = {
                    "gemini_available": advisor_service.gemini_client is not None,
                    "vertex_ai_available": advisor_service.llm_model is not None,
                }

                # Determine primary mode
                if advisor_service.gemini_client is not None:
                    mode = "gemini_primary"
                elif advisor_service.llm_model is not None:
                    mode = "vertex_ai_fallback"
                else:
                    mode = "mock"

                # Get cache statistics
                cache_stats = {
                    "cache_size": len(advisor_service._response_cache),
                    "cache_ttl": advisor_service._cache_ttl,
                }

                # Get configuration summary
                config_summary = {
                    "environment": config.environment,
                    "gemini_enabled": config.use_gemini,
                    "vertex_ai_model": config.model_name,
                    "max_retries": config.max_retries,
                    "request_timeout": config.request_timeout,
                }

                return {
                    "status": "healthy",
                    "service": "amatya-role-prompter",
                    "version": "1.0.0",
                    "mode": mode,
                    "ai_services": ai_services,
                    "cache_stats": cache_stats,
                    "configuration": config_summary,
                    "cors_origins": config.cors_origins,
                }
            else:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "service": "amatya-role-prompter",
                        "version": "1.0.0",
                        "error": "Advisor service not ready",
                    },
                )
        else:
            # Service not initialized yet (e.g., during startup or testing)
            return {
                "status": "starting",
                "service": "amatya-role-prompter",
                "version": "1.0.0",
                "message": "Service is starting up",
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "amatya-role-prompter",
                "version": "1.0.0",
                "error": str(e),
            },
        )


# Agent manifest endpoint
@app.get("/.well-known/agent.json")
async def agent_manifest():
    """Agent discovery manifest endpoint."""
    return {
        "name": "Amatya Role Prompter",
        "description": "LLM-powered advice generation with role-specific prompting",
        "version": "1.0.0",
        "methods": [
            {
                "name": "advise",
                "description": "Generate role-specific advice based on code snippets",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "description": "User role (e.g., 'backend_developer', 'security_engineer')",
                        },
                        "chunks": {
                            "type": "array",
                            "description": "Code snippets for context",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file_path": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["file_path", "content"],
                            },
                        },
                    },
                    "required": ["role", "chunks"],
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
    # Load configuration for host and port settings
    from config import AmataConfig

    dev_config = AmataConfig()

    # Use configuration-based host and port
    # Host is environment-specific: 127.0.0.1 for production, 0.0.0.0 for development
    host = dev_config.host
    port = dev_config.port

    logger.info(f"Starting development server on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True, log_level="info")
