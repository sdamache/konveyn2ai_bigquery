"""
JSON-RPC Server Demo - Example usage of the JsonRpcServer implementation.

This demo shows how to set up a JSON-RPC server with FastAPI and register methods.
"""

import os
import sys

from fastapi import FastAPI, Request

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from common.rpc_server import JsonRpcServer, logging_middleware

# Create FastAPI app
app = FastAPI(title="JSON-RPC Demo Server")

# Create JSON-RPC server instance
rpc_server = JsonRpcServer("Demo Calculator", "1.0.0")

# Add middleware
rpc_server.add_middleware(logging_middleware)


# Register JSON-RPC methods using decorator
@rpc_server.method("add", "Add two numbers together")
def add(a: int, b: int) -> dict:
    """Add two numbers and return the result."""
    return {"sum": a + b, "operation": "addition"}


@rpc_server.method("multiply", "Multiply two numbers")
def multiply(x: float, y: float) -> dict:
    """Multiply two numbers."""
    return {"product": x * y, "operation": "multiplication"}


@rpc_server.method("greet", "Greet a user with optional message")
def greet(name: str, message: str = "Hello") -> dict:
    """Greet a user."""
    return {"greeting": f"{message}, {name}!", "user": name}


@rpc_server.method("get_info", "Get information about the request")
def get_info(request_id: str, context: dict) -> dict:
    """Demonstrate context injection."""
    return {
        "request_id": request_id,
        "server": "Demo Calculator",
        "has_http_context": "http_request" in context,
    }


@rpc_server.method("notify_example", "Example notification handler")
def notify_example(action: str) -> dict:
    """Handle notifications (won't return response if no id)."""
    print(f"Notification received: {action}")
    return {"action": action, "processed": True}


# FastAPI endpoints
@app.post("/rpc")
async def json_rpc_endpoint(request: Request):
    """Main JSON-RPC endpoint."""
    return await rpc_server.handle_request(request)


@app.get("/.well-known/agent.json")
async def agent_manifest():
    """Agent manifest endpoint."""
    return rpc_server.get_manifest()


@app.get("/")
async def root():
    """Root endpoint with usage information."""
    return {
        "service": "JSON-RPC Demo Server",
        "version": "1.0.0",
        "endpoints": {"rpc": "/rpc", "manifest": "/.well-known/agent.json"},
        "methods": list(rpc_server.registry.get_methods().keys()),
        "examples": {
            "single_request": {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "add",
                "params": {"a": 5, "b": 3},
            },
            "batch_request": [
                {
                    "jsonrpc": "2.0",
                    "id": "1",
                    "method": "add",
                    "params": {"a": 1, "b": 2},
                },
                {
                    "jsonrpc": "2.0",
                    "id": "2",
                    "method": "multiply",
                    "params": {"x": 3, "y": 4},
                },
            ],
            "notification": {
                "jsonrpc": "2.0",
                "method": "notify_example",
                "params": {"action": "log_event"},
            },
        },
    }


# Example client code
async def demo_client():
    """Example client demonstrating various JSON-RPC calls."""
    from common.rpc_client import JsonRpcClient

    client = JsonRpcClient("http://localhost:8000/rpc")

    print("=== JSON-RPC Client Demo ===")

    # Single request
    print("\n1. Single request - Add operation:")
    response = await client.call("add", {"a": 10, "b": 5}, "req-1")
    print(f"Response: {response.model_dump()}")

    # Request with context injection
    print("\n2. Request with context injection:")
    response = await client.call("get_info", {}, "req-2")
    print(f"Response: {response.model_dump()}")

    # Method not found
    print("\n3. Method not found:")
    response = await client.call("unknown_method", {}, "req-3")
    print(f"Response: {response.model_dump()}")

    # Notification (no response expected)
    print("\n4. Notification (no response):")
    # Note: This would need a different client method for true notifications
    # For demo, we'll just show the concept
    print("Notification sent to notify_example method")


if __name__ == "__main__":
    import uvicorn

    print("Starting JSON-RPC Demo Server...")
    print("Available methods:", list(rpc_server.registry.get_methods().keys()))
    print("Visit http://localhost:8000 for usage examples")
    print("JSON-RPC endpoint: http://localhost:8000/rpc")
    print("Agent manifest: http://localhost:8000/.well-known/agent.json")

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
