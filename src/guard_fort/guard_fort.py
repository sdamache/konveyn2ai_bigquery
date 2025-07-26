"""
GuardFort Middleware - Core authentication, logging, and request tracing middleware
for the KonveyN2AI three-component architecture.

This middleware provides:
- Request ID generation and propagation
- Request timing and performance metrics
- Basic authentication framework (stubbed for demo)
- Structured logging with traceability
- Global exception handling
"""

import uuid
import time
import logging
import json
from typing import Callable, Optional, Dict, Any
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class GuardFort:
    """
    Core GuardFort middleware class for KonveyN2AI services.
    
    Provides request tracing, timing, authentication, and logging
    across all three components (Amatya, Janapada, Svami).
    """
    
    def __init__(
        self, 
        app: FastAPI,
        service_name: str = "konveyn2ai-service",
        enable_auth: bool = True,
        log_level: str = "INFO"
    ):
        """
        Initialize GuardFort middleware with FastAPI application.
        
        Args:
            app: FastAPI application instance
            service_name: Name of the service for logging
            enable_auth: Whether to enable authentication checks
            log_level: Logging level (DEBUG, INFO, WARN, ERROR)
        """
        self.app = app
        self.service_name = service_name
        self.enable_auth = enable_auth
        
        # Configure structured logging
        self.logger = logging.getLogger(f"guard_fort.{service_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Add JSON formatter if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"service": "' + service_name + '", "message": %(message)s}'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Register middleware with FastAPI
        self._register_middleware()
        
        self.logger.info(f'"GuardFort middleware initialized for {service_name}"')
    
    def _register_middleware(self):
        """Register the GuardFort middleware with the FastAPI application."""
        
        @self.app.middleware("http")
        async def guard_fort_middleware(request: Request, call_next: Callable):
            return await self._process_request(request, call_next)
    
    async def _process_request(self, request: Request, call_next: Callable) -> Response:
        """
        Core middleware processing logic.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or endpoint in the chain
            
        Returns:
            Response object with GuardFort headers and processing
        """
        # Generate or extract request ID
        request_id = self._get_or_generate_request_id(request)
        
        # Add request ID to request state for access in endpoints
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Authentication check (if enabled)
        if self.enable_auth:
            auth_result = self._validate_authentication(request)
            if not auth_result["valid"]:
                return self._create_auth_error_response(request_id, auth_result["reason"])
        
        try:
            # Process the request through the chain
            response = await call_next(request)
            
            # Calculate request duration
            duration = time.time() - start_time
            
            # Add GuardFort headers to response
            self._add_response_headers(response, request_id)
            
            # Log successful request
            self._log_request(request, response, duration, request_id)
            
            return response
            
        except Exception as e:
            # Calculate duration even for failed requests
            duration = time.time() - start_time
            
            # Log the exception
            self._log_exception(request, request_id, duration, e)
            
            # Return sanitized error response
            return self._create_error_response(request_id, e)
    
    def _get_or_generate_request_id(self, request: Request) -> str:
        """
        Get request ID from headers or generate a new one.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Request ID string (UUID format)
        """
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        return request_id
    
    def _validate_authentication(self, request: Request) -> Dict[str, Any]:
        """
        Validate authentication for the request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Dict with 'valid' boolean and 'reason' string
        """
        # For demo/hackathon purposes, implement stubbed authentication
        # In production, this would validate JWT tokens, API keys, etc.
        
        auth_header = request.headers.get("Authorization", "")
        
        # Allow health check endpoints without authentication
        if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
            return {"valid": True, "reason": "health_check"}
        
        # For demo: accept any Bearer token or allow requests without auth
        if auth_header.startswith("Bearer ") or not auth_header:
            return {"valid": True, "reason": "demo_mode"}
        
        return {"valid": False, "reason": "invalid_auth_format"}
    
    def _add_response_headers(self, response: Response, request_id: str):
        """
        Add GuardFort headers to the response.
        
        Args:
            response: FastAPI response object
            request_id: Request ID to add to headers
        """
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Service"] = self.service_name
        response.headers["X-GuardFort-Version"] = "1.0.0"
    
    def _log_request(
        self, 
        request: Request, 
        response: Response, 
        duration: float, 
        request_id: str
    ):
        """
        Log request details with structured format.
        
        Args:
            request: FastAPI request object
            response: FastAPI response object
            duration: Request duration in seconds
            request_id: Request ID for tracing
        """
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params) if request.query_params else None,
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
            "user_agent": request.headers.get("User-Agent", "unknown"),
            "remote_addr": request.client.host if request.client else "unknown"
        }
        
        # Log as JSON string for structured logging
        self.logger.info(json.dumps(log_data))
    
    def _log_exception(
        self, 
        request: Request, 
        request_id: str, 
        duration: float, 
        exception: Exception
    ):
        """
        Log exception details with context.
        
        Args:
            request: FastAPI request object
            request_id: Request ID for tracing
            duration: Request duration before failure
            exception: Exception that occurred
        """
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "duration_ms": round(duration * 1000, 2),
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "status": "error"
        }
        
        self.logger.error(json.dumps(log_data))
    
    def _create_auth_error_response(self, request_id: str, reason: str) -> JSONResponse:
        """
        Create standardized authentication error response.
        
        Args:
            request_id: Request ID for tracing
            reason: Authentication failure reason
            
        Returns:
            JSONResponse with 401 status
        """
        return JSONResponse(
            status_code=401,
            content={
                "error": "authentication_required",
                "message": "Valid authentication is required to access this resource",
                "request_id": request_id
            },
            headers={
                "X-Request-ID": request_id,
                "X-Service": self.service_name
            }
        )
    
    def _create_error_response(self, request_id: str, exception: Exception) -> JSONResponse:
        """
        Create sanitized error response for exceptions.
        
        Args:
            request_id: Request ID for tracing
            exception: Exception that occurred
            
        Returns:
            JSONResponse with 500 status and sanitized error
        """
        # Don't expose internal error details in production
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error", 
                "message": "An internal error occurred. Please contact support if the issue persists.",
                "request_id": request_id
            },
            headers={
                "X-Request-ID": request_id,
                "X-Service": self.service_name
            }
        )


# Utility function for easy integration
def init_guard_fort(
    app: FastAPI, 
    service_name: str,
    enable_auth: bool = True,
    log_level: str = "INFO"
) -> GuardFort:
    """
    Utility function to easily initialize GuardFort middleware.
    
    Args:
        app: FastAPI application instance
        service_name: Name of the service
        enable_auth: Whether to enable authentication
        log_level: Logging level
        
    Returns:
        GuardFort instance
        
    Example:
        app = FastAPI()
        guard_fort = init_guard_fort(app, "amatya-role-prompter")
    """
    return GuardFort(
        app=app,
        service_name=service_name,
        enable_auth=enable_auth,
        log_level=log_level
    )