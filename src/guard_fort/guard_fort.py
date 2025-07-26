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
import re
from typing import Callable, Optional, Dict, Any, List, Union
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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
        log_level: str = "INFO",
        cors_origins: List[str] = ["*"],
        cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        cors_headers: List[str] = ["*"],
        auth_header_name: str = "Authorization",
        auth_schemes: List[str] = ["Bearer", "ApiKey"],
        allowed_paths: List[str] = ["/health", "/", "/docs", "/openapi.json"],
        security_headers: bool = True
    ):
        """
        Initialize GuardFort middleware with FastAPI application.
        
        Args:
            app: FastAPI application instance
            service_name: Name of the service for logging
            enable_auth: Whether to enable authentication checks
            log_level: Logging level (DEBUG, INFO, WARN, ERROR)
            cors_origins: List of allowed CORS origins
            cors_methods: List of allowed CORS methods
            cors_headers: List of allowed CORS headers
            auth_header_name: Name of the authentication header
            auth_schemes: List of supported authentication schemes
            allowed_paths: List of paths that bypass authentication
            security_headers: Whether to add security headers
        """
        self.app = app
        self.service_name = service_name
        self.enable_auth = enable_auth
        self.cors_origins = cors_origins
        self.cors_methods = cors_methods
        self.cors_headers = cors_headers
        self.auth_header_name = auth_header_name
        self.auth_schemes = auth_schemes
        self.allowed_paths = allowed_paths
        self.security_headers = security_headers
        
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
        
        # Configure CORS if needed
        self._configure_cors()
        
        # Register middleware with FastAPI
        self._register_middleware()
        
        self.logger.info(f'"GuardFort middleware initialized for {service_name}"')
    
    def _configure_cors(self):
        """Configure CORS middleware if origins are specified."""
        if self.cors_origins and self.cors_origins != ["*"]:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.cors_origins,
                allow_credentials=True,
                allow_methods=self.cors_methods,
                allow_headers=self.cors_headers,
            )
            self.logger.info(f'"CORS configured for origins: {self.cors_origins}"')
    
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
            
            # Add security headers if enabled
            if self.security_headers:
                self._add_security_headers(response)
            
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
        # Allow health check and public endpoints without authentication
        if request.url.path in self.allowed_paths:
            return {"valid": True, "reason": "allowed_path"}
        
        auth_header = request.headers.get(self.auth_header_name, "")
        
        if not auth_header:
            return {"valid": False, "reason": "missing_auth_header"}
        
        # Validate authentication scheme
        auth_valid = self._validate_auth_scheme(auth_header)
        if not auth_valid["valid"]:
            return auth_valid
        
        # Extract and validate token
        token_valid = self._validate_token(auth_header)
        return token_valid
    
    def _validate_auth_scheme(self, auth_header: str) -> Dict[str, Any]:
        """
        Validate the authentication scheme format.
        
        Args:
            auth_header: Authorization header value
            
        Returns:
            Dict with validation result
        """
        for scheme in self.auth_schemes:
            if auth_header.startswith(f"{scheme} "):
                return {"valid": True, "reason": f"valid_{scheme.lower()}_scheme"}
        
        return {
            "valid": False, 
            "reason": f"invalid_auth_scheme_expected_{self.auth_schemes}"
        }
    
    def _validate_token(self, auth_header: str) -> Dict[str, Any]:
        """
        Validate the authentication token.
        
        Args:
            auth_header: Authorization header value
            
        Returns:
            Dict with validation result
        """
        # Extract token from header
        try:
            scheme, token = auth_header.split(" ", 1)
        except ValueError:
            return {"valid": False, "reason": "malformed_auth_header"}
        
        if scheme == "Bearer":
            return self._validate_bearer_token(token)
        elif scheme == "ApiKey":
            return self._validate_api_key(token)
        else:
            return {"valid": False, "reason": f"unsupported_scheme_{scheme}"}
    
    def _validate_bearer_token(self, token: str) -> Dict[str, Any]:
        """
        Validate Bearer token (JWT or similar).
        
        Args:
            token: Bearer token value
            
        Returns:
            Dict with validation result
        """
        # For demo/hackathon: Accept specific demo tokens or validate format
        demo_tokens = [
            "demo-token",
            "konveyn2ai-token", 
            "hackathon-demo",
            "test-token"
        ]
        
        if token in demo_tokens:
            return {"valid": True, "reason": "demo_token"}
        
        # Basic JWT format validation (3 parts separated by dots)
        if re.match(r'^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$', token):
            # In production, would verify JWT signature and claims
            return {"valid": True, "reason": "jwt_format_valid"}
        
        # Minimum token length check
        if len(token) >= 8:
            return {"valid": True, "reason": "token_length_valid"}
        
        return {"valid": False, "reason": "invalid_bearer_token"}
    
    def _validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """
        Validate API key.
        
        Args:
            api_key: API key value
            
        Returns:
            Dict with validation result
        """
        # For demo/hackathon: Accept demo API keys
        demo_api_keys = [
            "konveyn2ai-api-key-demo",
            "hackathon-api-key",
            "demo-api-key"
        ]
        
        if api_key in demo_api_keys:
            return {"valid": True, "reason": "demo_api_key"}
        
        # Basic API key format validation (alphanumeric, minimum length)
        if re.match(r'^[A-Za-z0-9_-]{16,}$', api_key):
            return {"valid": True, "reason": "api_key_format_valid"}
        
        return {"valid": False, "reason": "invalid_api_key"}
    
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
    
    def _add_security_headers(self, response: Response):
        """
        Add security headers to the response.
        
        Args:
            response: FastAPI response object
        """
        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' https://api.konveyn2ai.com; "
            "frame-ancestors 'none'"
        )
        
        # XSS Protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Content Type Options
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Frame Options
        response.headers["X-Frame-Options"] = "DENY"
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions Policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=(), "
            "accelerometer=(), ambient-light-sensor=()"
        )
        
        # Strict Transport Security (HTTPS only)
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )
    
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
    log_level: str = "INFO",
    cors_origins: List[str] = ["*"],
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    cors_headers: List[str] = ["*"],
    auth_schemes: List[str] = ["Bearer", "ApiKey"],
    allowed_paths: List[str] = ["/health", "/", "/docs", "/openapi.json"],
    security_headers: bool = True
) -> GuardFort:
    """
    Utility function to easily initialize GuardFort middleware.
    
    Args:
        app: FastAPI application instance
        service_name: Name of the service
        enable_auth: Whether to enable authentication
        log_level: Logging level
        cors_origins: List of allowed CORS origins
        cors_methods: List of allowed CORS methods
        cors_headers: List of allowed CORS headers
        auth_schemes: List of supported authentication schemes
        allowed_paths: List of paths that bypass authentication
        security_headers: Whether to add security headers
        
    Returns:
        GuardFort instance
        
    Examples:
        # Basic usage
        app = FastAPI()
        guard_fort = init_guard_fort(app, "amatya-role-prompter")
        
        # With custom CORS and auth settings
        guard_fort = init_guard_fort(
            app, 
            "janapada-memory",
            cors_origins=["https://konveyn2ai.com"],
            auth_schemes=["Bearer"],
            security_headers=True
        )
    """
    return GuardFort(
        app=app,
        service_name=service_name,
        enable_auth=enable_auth,
        log_level=log_level,
        cors_origins=cors_origins,
        cors_methods=cors_methods,
        cors_headers=cors_headers,
        auth_schemes=auth_schemes,
        allowed_paths=allowed_paths,
        security_headers=security_headers
    )