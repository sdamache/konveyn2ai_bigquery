#!/usr/bin/env python3
"""
GuardFort Middleware Demo

Demonstrates the core functionality of GuardFort middleware
including request ID generation, timing, authentication, and logging.

Usage:
    python examples/guard_fort_demo.py

Then visit:
    - http://localhost:8000/ (health check, no auth required)
    - http://localhost:8000/test (basic endpoint with auth)
    - http://localhost:8000/error (triggers exception handling)
    - http://localhost:8000/docs (FastAPI auto-generated docs)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from guard_fort import init_guard_fort

# Create FastAPI application
app = FastAPI(
    title="GuardFort Demo",
    description="Demonstration of GuardFort middleware functionality",
    version="1.0.0"
)

# Initialize GuardFort middleware
guard_fort = init_guard_fort(
    app=app,
    service_name="guard-fort-demo",
    enable_auth=True,
    log_level="INFO"
)

# Demo endpoints

@app.get("/")
async def health_check():
    """Health check endpoint (no authentication required)."""
    return {
        "status": "healthy",
        "service": "guard-fort-demo",
        "message": "GuardFort middleware is working!"
    }

@app.get("/test")
async def test_endpoint(request: Request):
    """Test endpoint that demonstrates request ID access."""
    return {
        "message": "Hello from GuardFort!",
        "request_id": request.state.request_id,
        "path": request.url.path,
        "method": request.method
    }

@app.get("/protected")
async def protected_endpoint(request: Request):
    """Protected endpoint requiring authentication."""
    return {
        "message": "This is a protected resource",
        "request_id": request.state.request_id,
        "auth_status": "authenticated"
    }

@app.get("/error")
async def error_endpoint():
    """Endpoint that triggers an exception for testing error handling."""
    raise ValueError("This is a test exception to demonstrate error handling")

@app.post("/data")
async def data_endpoint(request: Request, data: dict = None):
    """POST endpoint to test request logging with data."""
    return {
        "message": "Data received",
        "request_id": request.state.request_id,
        "received_data": data
    }

@app.get("/slow")
async def slow_endpoint(request: Request):
    """Slow endpoint to test timing functionality."""
    import time
    time.sleep(0.1)  # Simulate slow processing
    return {
        "message": "Slow operation completed",
        "request_id": request.state.request_id,
        "note": "This endpoint took ~100ms to process"
    }

def main():
    """Run the demo server."""
    print("🚀 Starting GuardFort Demo Server...")
    print("📋 Available endpoints:")
    print("   - GET  /           Health check (no auth)")
    print("   - GET  /test       Basic test endpoint")
    print("   - GET  /protected  Protected endpoint")
    print("   - GET  /error      Triggers exception")
    print("   - POST /data       Data processing endpoint")
    print("   - GET  /slow       Slow endpoint for timing demo")
    print("   - GET  /docs       FastAPI documentation")
    print()
    print("🔧 Test with curl:")
    print("   curl http://localhost:8000/test")
    print("   curl -H 'X-Request-ID: my-test-123' http://localhost:8000/test")
    print("   curl -H 'Authorization: Bearer demo-token' http://localhost:8000/protected")
    print("   curl -X POST -H 'Content-Type: application/json' -d '{\"key\":\"value\"}' http://localhost:8000/data")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()