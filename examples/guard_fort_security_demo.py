#!/usr/bin/env python3
"""
GuardFort Security Features Demo

Demonstrates the enhanced authentication and security features of GuardFort middleware
including CORS, security headers, token validation, and API key authentication.

Usage:
    python examples/guard_fort_security_demo.py

Then test with curl:
    # Basic authentication
    curl -H 'Authorization: Bearer demo-token' http://localhost:8001/test

    # API key authentication
    curl -H 'Authorization: ApiKey demo-api-key' http://localhost:8001/test

    # JWT format validation
    curl -H 'Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWV9.TJVA95OrM7E2cBab30RMHrHDcEfxjoYZgeFONFh7HgQ' http://localhost:8001/test

    # Test without authentication (should fail)
    curl http://localhost:8001/test

    # Test with invalid scheme (should fail)
    curl -H 'Authorization: Invalid demo-token' http://localhost:8001/test

    # View security headers
    curl -I -H 'Authorization: Bearer demo-token' http://localhost:8001/test
"""

import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from guard_fort import init_guard_fort

# Create FastAPI application
app = FastAPI(
    title="GuardFort Security Demo",
    description="Demonstration of GuardFort enhanced security features",
    version="1.0.0",
)

# Initialize GuardFort middleware with security features
guard_fort = init_guard_fort(
    app=app,
    service_name="security-demo",
    enable_auth=True,
    log_level="INFO",
    cors_origins=[
        "https://konveyn2ai.com",
        "http://localhost:3000",
    ],  # Specific CORS origins
    cors_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    cors_headers=["Authorization", "Content-Type", "X-Request-ID"],
    auth_schemes=["Bearer", "ApiKey"],  # Support both Bearer tokens and API keys
    allowed_paths=[
        "/health",
        "/",
        "/docs",
        "/openapi.json",
        "/public",
    ],  # Paths that bypass auth
    security_headers=True,  # Enable comprehensive security headers
)

# Demo endpoints


@app.get("/")
async def health_check():
    """Health check endpoint (no authentication required)."""
    return {
        "status": "healthy",
        "service": "guard-fort-security-demo",
        "message": "Enhanced security features enabled!",
        "features": [
            "Bearer token authentication",
            "API key authentication",
            "CORS protection",
            "Security headers (CSP, XSS, etc.)",
            "Request tracing",
            "Structured logging",
        ],
    }


@app.get("/public")
async def public_endpoint():
    """Public endpoint that bypasses authentication."""
    return {
        "message": "This is a public endpoint",
        "note": "No authentication required",
        "security": "Still protected by security headers and CORS",
    }


@app.get("/test")
async def authenticated_endpoint(request: Request):
    """Protected endpoint requiring authentication."""
    return {
        "message": "Successfully authenticated!",
        "request_id": request.state.request_id,
        "auth_status": "valid",
        "endpoint": "protected",
    }


@app.get("/bearer-only")
async def bearer_only_endpoint(request: Request):
    """Endpoint demonstrating Bearer token requirement."""
    return {
        "message": "Bearer token authenticated",
        "request_id": request.state.request_id,
        "token_type": "Bearer",
        "note": "This endpoint accepts Bearer tokens (demo tokens, JWT format, or 8+ char tokens)",
    }


@app.get("/apikey-demo")
async def apikey_demo_endpoint(request: Request):
    """Endpoint demonstrating API key authentication."""
    return {
        "message": "API key authenticated",
        "request_id": request.state.request_id,
        "token_type": "ApiKey",
        "note": "This endpoint accepts API keys (demo keys or 16+ alphanumeric characters)",
    }


@app.post("/secure-data")
async def secure_data_endpoint(request: Request, data: dict = None):
    """POST endpoint with authentication and CORS handling."""
    return {
        "message": "Secure data processed",
        "request_id": request.state.request_id,
        "received_data": data,
        "security_features": [
            "Authentication required",
            "CORS headers applied",
            "Security headers included",
            "Request/response logged",
        ],
    }


@app.get("/headers-demo")
async def headers_demo_endpoint(request: Request):
    """Endpoint to demonstrate security headers in response."""
    return {
        "message": "Check the response headers for security features",
        "request_id": request.state.request_id,
        "security_headers": [
            "Content-Security-Policy",
            "X-XSS-Protection",
            "X-Content-Type-Options",
            "X-Frame-Options",
            "Referrer-Policy",
            "Permissions-Policy",
            "Strict-Transport-Security",
        ],
        "guard_fort_headers": ["X-Request-ID", "X-Service", "X-GuardFort-Version"],
    }


@app.get("/error-demo")
async def error_demo_endpoint():
    """Endpoint that triggers error handling demo."""
    raise HTTPException(status_code=400, detail="Demo error for testing error handling")


def main():
    """Run the security demo server."""
    print("🚀 Starting GuardFort Security Demo Server...")
    print("=" * 60)
    print("🔐 Security Features Enabled:")
    print("   ✅ Enhanced Authentication (Bearer + ApiKey)")
    print("   ✅ CORS Protection (specific origins)")
    print("   ✅ Security Headers (CSP, XSS, etc.)")
    print("   ✅ Request ID Tracing")
    print("   ✅ Structured Logging")
    print("   ✅ Exception Handling")
    print()
    print("📋 Available Endpoints:")
    print("   - GET  /           Health check (no auth)")
    print("   - GET  /public     Public endpoint (no auth)")
    print("   - GET  /test       Basic protected endpoint")
    print("   - GET  /bearer-only   Bearer token demo")
    print("   - GET  /apikey-demo   API key demo")
    print("   - POST /secure-data   POST with authentication")
    print("   - GET  /headers-demo  Security headers demo")
    print("   - GET  /error-demo    Error handling demo")
    print("   - GET  /docs       FastAPI documentation")
    print()
    print("🔧 Authentication Examples:")
    print("   # Bearer Token (demo)")
    print("   curl -H 'Authorization: Bearer demo-token' http://localhost:8001/test")
    print()
    print("   # Bearer Token (JWT format)")
    print(
        "   curl -H 'Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWV9.TJVA95OrM7E2cBab30RMHrHDcEfxjoYZgeFONFh7HgQ' http://localhost:8001/test"
    )
    print()
    print("   # API Key")
    print("   curl -H 'Authorization: ApiKey demo-api-key' http://localhost:8001/test")
    print()
    print("   # View Security Headers")
    print(
        "   curl -I -H 'Authorization: Bearer demo-token' http://localhost:8001/headers-demo"
    )
    print()
    print("   # Test Authentication Failure")
    print("   curl http://localhost:8001/test")
    print()
    print("   # Test CORS Preflight")
    print(
        "   curl -X OPTIONS -H 'Origin: https://konveyn2ai.com' http://localhost:8001/test"
    )
    print()

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")


if __name__ == "__main__":
    main()
