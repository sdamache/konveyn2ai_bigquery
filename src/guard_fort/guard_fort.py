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
import statistics
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Callable, Optional, Dict, Any, List, Union
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware


class MetricsCollector:
    """
    Advanced metrics collection system for GuardFort middleware.
    
    Tracks performance metrics, error rates, and request patterns
    with configurable retention and aggregation periods.
    """
    
    def __init__(self, retention_minutes: int = 60, max_samples: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            retention_minutes: How long to keep metrics data
            max_samples: Maximum number of samples to retain per metric
        """
        self.retention_minutes = retention_minutes
        self.max_samples = max_samples
        
        # Request metrics
        self.request_durations = defaultdict(lambda: deque(maxlen=max_samples))
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.status_code_counts = defaultdict(int)
        
        # Performance metrics
        self.response_times_by_endpoint = defaultdict(lambda: deque(maxlen=max_samples))
        self.concurrent_requests = 0
        self.peak_concurrent_requests = 0
        
        # Authentication metrics
        self.auth_success_count = 0
        self.auth_failure_count = 0
        self.auth_failure_reasons = defaultdict(int)
        
        # Request pattern metrics
        self.user_agent_counts = defaultdict(int)
        self.ip_address_counts = defaultdict(int)
        self.endpoint_usage = defaultdict(int)
        
        # Time-based metrics
        self.hourly_request_counts = defaultdict(int)
        self.daily_request_counts = defaultdict(int)
        
        # Error tracking
        self.error_details = deque(maxlen=100)  # Keep last 100 errors
        
    def record_request_start(self):
        """Record the start of a request for concurrency tracking."""
        self.concurrent_requests += 1
        if self.concurrent_requests > self.peak_concurrent_requests:
            self.peak_concurrent_requests = self.concurrent_requests
    
    def record_request_end(self, 
                          method: str, 
                          path: str, 
                          status_code: int, 
                          duration: float,
                          user_agent: str = None,
                          ip_address: str = None):
        """
        Record request completion metrics.
        
        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration: Request duration in seconds
            user_agent: User agent string
            ip_address: Client IP address
        """
        self.concurrent_requests = max(0, self.concurrent_requests - 1)
        
        # Record basic metrics
        endpoint_key = f"{method}:{path}"
        self.request_durations[endpoint_key].append(duration)
        self.request_counts[endpoint_key] += 1
        self.status_code_counts[status_code] += 1
        self.response_times_by_endpoint[endpoint_key].append(duration)
        self.endpoint_usage[endpoint_key] += 1
        
        # Record error metrics
        if status_code >= 400:
            self.error_counts[endpoint_key] += 1
        
        # Record usage patterns
        if user_agent:
            self.user_agent_counts[user_agent] += 1
        if ip_address:
            self.ip_address_counts[ip_address] += 1
        
        # Record time-based metrics
        now = datetime.now(timezone.utc)
        hour_key = now.strftime("%Y-%m-%d-%H")
        day_key = now.strftime("%Y-%m-%d")
        self.hourly_request_counts[hour_key] += 1
        self.daily_request_counts[day_key] += 1
    
    def record_auth_result(self, success: bool, reason: str = None):
        """
        Record authentication attempt result.
        
        Args:
            success: Whether authentication succeeded
            reason: Failure reason if unsuccessful
        """
        if success:
            self.auth_success_count += 1
        else:
            self.auth_failure_count += 1
            if reason:
                self.auth_failure_reasons[reason] += 1
    
    def record_error(self, request_id: str, endpoint: str, error_type: str, error_message: str):
        """
        Record detailed error information.
        
        Args:
            request_id: Request ID for tracing
            endpoint: Endpoint where error occurred
            error_type: Type of error
            error_message: Error message
        """
        error_detail = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "endpoint": endpoint,
            "error_type": error_type,
            "error_message": error_message
        }
        self.error_details.append(error_detail)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.
        
        Returns:
            Dictionary containing all collected metrics
        """
        # Calculate performance statistics
        all_durations = []
        endpoint_stats = {}
        
        for endpoint, durations in self.response_times_by_endpoint.items():
            if durations:
                endpoint_stats[endpoint] = {
                    "count": len(durations),
                    "avg_duration_ms": round(statistics.mean(durations) * 1000, 2),
                    "min_duration_ms": round(min(durations) * 1000, 2),
                    "max_duration_ms": round(max(durations) * 1000, 2),
                    "median_duration_ms": round(statistics.median(durations) * 1000, 2)
                }
                all_durations.extend(durations)
        
        # Overall performance stats
        overall_stats = {}
        if all_durations:
            overall_stats = {
                "total_requests": len(all_durations),
                "avg_response_time_ms": round(statistics.mean(all_durations) * 1000, 2),
                "median_response_time_ms": round(statistics.median(all_durations) * 1000, 2),
                "p95_response_time_ms": round(statistics.quantiles(all_durations, n=20)[18] * 1000, 2) if len(all_durations) >= 20 else None,
                "p99_response_time_ms": round(statistics.quantiles(all_durations, n=100)[98] * 1000, 2) if len(all_durations) >= 100 else None
            }
        
        # Error rate calculation
        total_requests = sum(self.request_counts.values())
        total_errors = sum(self.error_counts.values())
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance": {
                "overall": overall_stats,
                "by_endpoint": endpoint_stats,
                "concurrent_requests": self.concurrent_requests,
                "peak_concurrent_requests": self.peak_concurrent_requests
            },
            "requests": {
                "total_count": total_requests,
                "by_endpoint": dict(self.request_counts),
                "by_status_code": dict(self.status_code_counts),
                "error_rate_percent": round(error_rate, 2)
            },
            "authentication": {
                "success_count": self.auth_success_count,
                "failure_count": self.auth_failure_count,
                "failure_reasons": dict(self.auth_failure_reasons),
                "success_rate_percent": round(
                    (self.auth_success_count / (self.auth_success_count + self.auth_failure_count) * 100) 
                    if (self.auth_success_count + self.auth_failure_count) > 0 else 0, 2
                )
            },
            "usage_patterns": {
                "top_endpoints": sorted(
                    self.endpoint_usage.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10],
                "unique_user_agents": len(self.user_agent_counts),
                "unique_ip_addresses": len(self.ip_address_counts),
                "top_user_agents": sorted(
                    self.user_agent_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            },
            "errors": {
                "total_count": total_errors,
                "by_endpoint": dict(self.error_counts),
                "recent_errors": list(self.error_details)[-10:]  # Last 10 errors
            },
            "time_patterns": {
                "hourly_distribution": dict(list(self.hourly_request_counts.items())[-24:]),  # Last 24 hours
                "daily_distribution": dict(list(self.daily_request_counts.items())[-7:])    # Last 7 days
            }
        }


class StructuredLogger:
    """
    Advanced structured logging system with multiple output formats
    and configurable log levels per component.
    """
    
    def __init__(self, 
                 service_name: str, 
                 log_level: str = "INFO",
                 log_format: str = "json",
                 include_trace: bool = True):
        """
        Initialize structured logger.
        
        Args:
            service_name: Name of the service
            log_level: Default logging level
            log_format: Log format (json, structured, simple)
            include_trace: Whether to include trace information
        """
        self.service_name = service_name
        self.log_format = log_format
        self.include_trace = include_trace
        
        # Create logger instance
        self.logger = logging.getLogger(f"guard_fort.{service_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Add custom formatter based on format preference
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = self._create_formatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _create_formatter(self) -> logging.Formatter:
        """Create appropriate formatter based on log format setting."""
        if self.log_format == "json":
            return JsonFormatter(self.service_name, self.include_trace)
        elif self.log_format == "structured":
            return StructuredFormatter(self.service_name, self.include_trace)
        else:
            return SimpleFormatter(self.service_name)
    
    def log_request(self, level: str, data: Dict[str, Any]):
        """
        Log request with structured data.
        
        Args:
            level: Log level (DEBUG, INFO, WARN, ERROR)
            data: Structured log data
        """
        log_method = getattr(self.logger, level.lower())
        
        if self.log_format == "json":
            log_method(json.dumps(data))
        else:
            log_method("Request processed", extra=data)
    
    def log_auth_event(self, success: bool, reason: str = None, request_id: str = None):
        """
        Log authentication event.
        
        Args:
            success: Whether authentication succeeded
            reason: Authentication result reason
            request_id: Request ID for tracing
        """
        data = {
            "event_type": "authentication",
            "success": success,
            "reason": reason,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        level = "INFO" if success else "WARN"
        self.log_request(level, data)
    
    def log_performance_alert(self, metric: str, value: float, threshold: float, request_id: str = None):
        """
        Log performance threshold breach.
        
        Args:
            metric: Performance metric name
            value: Current value
            threshold: Threshold that was breached
            request_id: Request ID for tracing
        """
        data = {
            "event_type": "performance_alert",
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.log_request("WARN", data)
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], request_id: str = None):
        """
        Log security-related event.
        
        Args:
            event_type: Type of security event
            details: Event details
            request_id: Request ID for tracing
        """
        data = {
            "event_type": "security",
            "security_event_type": event_type,
            "details": details,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.log_request("WARN", data)


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def __init__(self, service_name: str, include_trace: bool = True):
        super().__init__()
        self.service_name = service_name
        self.include_trace = include_trace
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service_name,
            "logger": record.name,
            "message": record.getMessage()
        }
        
        if self.include_trace:
            log_data.update({
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            })
        
        # Add any extra fields from the log record
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                              'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process', 'getMessage']:
                    log_data[key] = value
        
        return json.dumps(log_data)


class StructuredFormatter(logging.Formatter):
    """Structured text formatter for readable logging."""
    
    def __init__(self, service_name: str, include_trace: bool = True):
        super().__init__()
        self.service_name = service_name
        self.include_trace = include_trace
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, timezone.utc).isoformat()
        base_msg = f"[{timestamp}] {record.levelname} {self.service_name}: {record.getMessage()}"
        
        if self.include_trace:
            base_msg += f" ({record.module}:{record.funcName}:{record.lineno})"
        
        return base_msg


class SimpleFormatter(logging.Formatter):
    """Simple log formatter for basic output."""
    
    def __init__(self, service_name: str):
        super().__init__(
            f"%(asctime)s - {service_name} - %(levelname)s - %(message)s"
        )


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
        log_format: str = "json",
        cors_origins: List[str] = ["*"],
        cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        cors_headers: List[str] = ["*"],
        auth_header_name: str = "Authorization",
        auth_schemes: List[str] = ["Bearer", "ApiKey"],
        allowed_paths: List[str] = ["/health", "/", "/docs", "/openapi.json"],
        security_headers: bool = True,
        enable_metrics: bool = True,
        metrics_retention_minutes: int = 60,
        performance_thresholds: Dict[str, float] = None,
        include_trace_logs: bool = True
    ):
        """
        Initialize GuardFort middleware with FastAPI application.
        
        Args:
            app: FastAPI application instance
            service_name: Name of the service for logging
            enable_auth: Whether to enable authentication checks
            log_level: Logging level (DEBUG, INFO, WARN, ERROR)
            log_format: Log format (json, structured, simple)
            cors_origins: List of allowed CORS origins
            cors_methods: List of allowed CORS methods
            cors_headers: List of allowed CORS headers
            auth_header_name: Name of the authentication header
            auth_schemes: List of supported authentication schemes
            allowed_paths: List of paths that bypass authentication
            security_headers: Whether to add security headers
            enable_metrics: Whether to enable metrics collection
            metrics_retention_minutes: How long to retain metrics data
            performance_thresholds: Performance alert thresholds
            include_trace_logs: Whether to include trace information in logs
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
        self.enable_metrics = enable_metrics
        
        # Set up performance thresholds with defaults
        self.performance_thresholds = performance_thresholds or {
            "response_time_ms": 5000,  # 5 seconds
            "concurrent_requests": 100,
            "error_rate_percent": 5.0,
            "memory_usage_mb": 512
        }
        
        # Initialize structured logging system
        self.structured_logger = StructuredLogger(
            service_name=service_name,
            log_level=log_level,
            log_format=log_format,
            include_trace=include_trace_logs
        )
        
        # Keep reference to standard logger for backward compatibility
        self.logger = self.structured_logger.logger
        
        # Initialize metrics collection system
        if self.enable_metrics:
            self.metrics = MetricsCollector(
                retention_minutes=metrics_retention_minutes,
                max_samples=1000
            )
        else:
            self.metrics = None
        
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
        
        # Record request start for metrics
        if self.metrics:
            self.metrics.record_request_start()
        
        # Start timing
        start_time = time.time()
        
        # Authentication check (if enabled)
        auth_success = True
        auth_reason = "no_auth_required"
        
        if self.enable_auth:
            auth_result = self._validate_authentication(request)
            auth_success = auth_result["valid"]
            auth_reason = auth_result["reason"]
            
            # Record authentication metrics
            if self.metrics:
                self.metrics.record_auth_result(auth_success, auth_reason)
            
            # Log authentication event
            self.structured_logger.log_auth_event(auth_success, auth_reason, request_id)
            
            if not auth_success:
                duration = time.time() - start_time
                
                # Record failed request metrics
                if self.metrics:
                    self.metrics.record_request_end(
                        request.method, 
                        request.url.path, 
                        401, 
                        duration,
                        request.headers.get("User-Agent"),
                        request.client.host if request.client else None
                    )
                
                return self._create_auth_error_response(request_id, auth_reason)
        
        try:
            # Process the request through the chain
            response = await call_next(request)
            
            # Calculate request duration
            duration = time.time() - start_time
            duration_ms = duration * 1000
            
            # Check performance thresholds and log alerts
            if duration_ms > self.performance_thresholds["response_time_ms"]:
                self.structured_logger.log_performance_alert(
                    "response_time", 
                    duration_ms, 
                    self.performance_thresholds["response_time_ms"],
                    request_id
                )
            
            if self.metrics and self.metrics.concurrent_requests > self.performance_thresholds["concurrent_requests"]:
                self.structured_logger.log_performance_alert(
                    "concurrent_requests",
                    self.metrics.concurrent_requests,
                    self.performance_thresholds["concurrent_requests"],
                    request_id
                )
            
            # Record request metrics
            if self.metrics:
                self.metrics.record_request_end(
                    request.method,
                    request.url.path,
                    response.status_code,
                    duration,
                    request.headers.get("User-Agent"),
                    request.client.host if request.client else None
                )
            
            # Add GuardFort headers to response
            self._add_response_headers(response, request_id)
            
            # Add security headers if enabled
            if self.security_headers:
                self._add_security_headers(response)
            
            # Log successful request with enhanced data
            self._log_request_enhanced(request, response, duration, request_id, auth_reason)
            
            return response
            
        except Exception as e:
            # Calculate duration even for failed requests
            duration = time.time() - start_time
            
            # Record error metrics
            if self.metrics:
                endpoint_key = f"{request.method}:{request.url.path}"
                self.metrics.record_error(
                    request_id, 
                    endpoint_key, 
                    type(e).__name__, 
                    str(e)
                )
                self.metrics.record_request_end(
                    request.method,
                    request.url.path,
                    500,
                    duration,
                    request.headers.get("User-Agent"),
                    request.client.host if request.client else None
                )
            
            # Log the exception with enhanced data
            self._log_exception_enhanced(request, request_id, duration, e)
            
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
    
    def _log_request_enhanced(
        self, 
        request: Request, 
        response: Response, 
        duration: float, 
        request_id: str,
        auth_reason: str = None
    ):
        """
        Enhanced request logging with additional context and structured data.
        
        Args:
            request: FastAPI request object
            response: FastAPI response object
            duration: Request duration in seconds
            request_id: Request ID for tracing
            auth_reason: Authentication result reason
        """
        log_data = {
            "event_type": "request_completed",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params) if request.query_params else None,
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
            "user_agent": request.headers.get("User-Agent", "unknown"),
            "remote_addr": request.client.host if request.client else "unknown",
            "auth_reason": auth_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_length": response.headers.get("content-length"),
            "content_type": response.headers.get("content-type")
        }
        
        # Add performance classification
        if duration * 1000 < 100:
            log_data["performance_category"] = "fast"
        elif duration * 1000 < 1000:
            log_data["performance_category"] = "normal"
        elif duration * 1000 < 5000:
            log_data["performance_category"] = "slow"
        else:
            log_data["performance_category"] = "very_slow"
        
        # Add status category
        if 200 <= response.status_code < 300:
            log_data["status_category"] = "success"
        elif 300 <= response.status_code < 400:
            log_data["status_category"] = "redirect"
        elif 400 <= response.status_code < 500:
            log_data["status_category"] = "client_error"
        else:
            log_data["status_category"] = "server_error"
        
        self.structured_logger.log_request("INFO", log_data)
    
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
    
    def _log_exception_enhanced(
        self, 
        request: Request, 
        request_id: str, 
        duration: float, 
        exception: Exception
    ):
        """
        Enhanced exception logging with additional context and structured data.
        
        Args:
            request: FastAPI request object
            request_id: Request ID for tracing
            duration: Request duration before failure
            exception: Exception that occurred
        """
        log_data = {
            "event_type": "request_failed",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params) if request.query_params else None,
            "duration_ms": round(duration * 1000, 2),
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "user_agent": request.headers.get("User-Agent", "unknown"),
            "remote_addr": request.client.host if request.client else "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "error"
        }
        
        self.structured_logger.log_request("ERROR", log_data)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics summary.
        
        Returns:
            Dictionary containing comprehensive metrics data
        """
        if not self.metrics:
            return {
                "error": "Metrics collection is disabled",
                "enabled": False
            }
        
        return {
            "enabled": True,
            "service": self.service_name,
            **self.metrics.get_metrics_summary()
        }
    
    def add_metrics_endpoint(self, path: str = "/metrics"):
        """
        Add a metrics endpoint to the FastAPI application.
        
        Args:
            path: URL path for the metrics endpoint
        """
        if not self.metrics:
            return
        
        @self.app.get(path)
        async def get_metrics_endpoint():
            """Return comprehensive metrics data in JSON format."""
            return self.get_metrics()
        
        # Add the metrics path to allowed paths (no auth required)
        if path not in self.allowed_paths:
            self.allowed_paths.append(path)
    
    def add_health_endpoint(self, path: str = "/health"):
        """
        Add a health check endpoint with basic metrics.
        
        Args:
            path: URL path for the health endpoint
        """
        @self.app.get(path)
        async def health_check():
            """Return service health status with basic metrics."""
            health_data = {
                "status": "healthy",
                "service": self.service_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0"
            }
            
            if self.metrics:
                # Add basic performance indicators
                metrics_summary = self.metrics.get_metrics_summary()
                health_data.update({
                    "requests_total": metrics_summary.get("requests", {}).get("total_count", 0),
                    "error_rate_percent": metrics_summary.get("requests", {}).get("error_rate_percent", 0),
                    "concurrent_requests": metrics_summary.get("performance", {}).get("concurrent_requests", 0),
                    "avg_response_time_ms": metrics_summary.get("performance", {}).get("overall", {}).get("avg_response_time_ms", 0)
                })
            
            return health_data
        
        # Add the health path to allowed paths (no auth required)
        if path not in self.allowed_paths:
            self.allowed_paths.append(path)
    
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
    log_format: str = "json",
    cors_origins: List[str] = ["*"],
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    cors_headers: List[str] = ["*"],
    auth_schemes: List[str] = ["Bearer", "ApiKey"],
    allowed_paths: List[str] = ["/health", "/", "/docs", "/openapi.json"],
    security_headers: bool = True,
    enable_metrics: bool = True,
    metrics_retention_minutes: int = 60,
    performance_thresholds: Dict[str, float] = None,
    include_trace_logs: bool = True,
    add_metrics_endpoint: bool = True,
    add_health_endpoint: bool = True
) -> GuardFort:
    """
    Utility function to easily initialize GuardFort middleware.
    
    Args:
        app: FastAPI application instance
        service_name: Name of the service
        enable_auth: Whether to enable authentication
        log_level: Logging level
        log_format: Log format (json, structured, simple)
        cors_origins: List of allowed CORS origins
        cors_methods: List of allowed CORS methods
        cors_headers: List of allowed CORS headers
        auth_schemes: List of supported authentication schemes
        allowed_paths: List of paths that bypass authentication
        security_headers: Whether to add security headers
        enable_metrics: Whether to enable metrics collection
        metrics_retention_minutes: How long to retain metrics data
        performance_thresholds: Performance alert thresholds
        include_trace_logs: Whether to include trace information in logs
        add_metrics_endpoint: Whether to automatically add /metrics endpoint
        add_health_endpoint: Whether to automatically add /health endpoint
        
    Returns:
        GuardFort instance
        
    Examples:
        # Basic usage with enhanced logging and metrics
        app = FastAPI()
        guard_fort = init_guard_fort(app, "amatya-role-prompter")
        
        # With custom settings and structured logging
        guard_fort = init_guard_fort(
            app, 
            "janapada-memory",
            log_format="structured",
            cors_origins=["https://konveyn2ai.com"],
            auth_schemes=["Bearer"],
            security_headers=True,
            performance_thresholds={
                "response_time_ms": 3000,
                "concurrent_requests": 50,
                "error_rate_percent": 2.0
            }
        )
        
        # Disable metrics for lightweight setup
        guard_fort = init_guard_fort(
            app,
            "svami-orchestrator", 
            enable_metrics=False,
            log_format="simple"
        )
    """
    # Create GuardFort instance
    guard_fort = GuardFort(
        app=app,
        service_name=service_name,
        enable_auth=enable_auth,
        log_level=log_level,
        log_format=log_format,
        cors_origins=cors_origins,
        cors_methods=cors_methods,
        cors_headers=cors_headers,
        auth_schemes=auth_schemes,
        allowed_paths=allowed_paths,
        security_headers=security_headers,
        enable_metrics=enable_metrics,
        metrics_retention_minutes=metrics_retention_minutes,
        performance_thresholds=performance_thresholds,
        include_trace_logs=include_trace_logs
    )
    
    # Add endpoints if requested
    if add_metrics_endpoint and enable_metrics:
        guard_fort.add_metrics_endpoint()
    
    if add_health_endpoint:
        guard_fort.add_health_endpoint()
    
    return guard_fort