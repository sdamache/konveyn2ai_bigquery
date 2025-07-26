"""
Enhanced Agent Manifest Demo - Showcasing the new AgentManifestGenerator capabilities.

This demo shows:
1. Enhanced manifest generation with detailed method schemas
2. Type hint extraction and parameter documentation
3. Custom capabilities and endpoints
4. Agent discovery and remote method calling
"""

import asyncio
import os
import sys
from typing import Dict, List, Optional

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, Request  # noqa: E402
import uvicorn  # noqa: E402

from common.rpc_server import JsonRpcServer, logging_middleware  # noqa: E402
from common.agent_manifest import AgentDiscovery  # noqa: E402

# Create FastAPI app
app = FastAPI(title="Enhanced Agent Manifest Demo")

# Create JSON-RPC server with enhanced manifest capabilities
rpc_server = JsonRpcServer(
    title="Enhanced Calculator Agent",
    version="2.0.0",
    description="A demonstration agent with enhanced manifest capabilities",
)

# Add middleware
rpc_server.add_middleware(logging_middleware)

# Add custom capabilities
rpc_server.add_capability(
    "mathematical-operations",
    "1.0",
    "Basic mathematical operations including arithmetic and statistics",
)

rpc_server.add_capability(
    "data-processing", "1.0", "Data processing and analysis capabilities"
)

# Add service endpoints
rpc_server.add_endpoint("health", "/health")
rpc_server.add_endpoint("metrics", "/metrics")

# Set metadata
rpc_server.set_metadata("author", "KonveyN2AI Team")
rpc_server.set_metadata("license", "MIT")
rpc_server.set_metadata("repository", "https://github.com/neeharve/KonveyN2AI")


# Register enhanced JSON-RPC methods with detailed type hints and docstrings
@rpc_server.method("add", "Add two or more numbers together")
def add(a: float, b: float, c: Optional[float] = None) -> Dict[str, float]:
    """
    Add two or more numbers together.

    Args:
        a: First number to add
        b: Second number to add
        c: Optional third number to add

    Returns:
        Dictionary containing the sum and operation details
    """
    result = a + b
    if c is not None:
        result += c

    return {
        "sum": result,
        "operation": "addition",
        "operands": [a, b] + ([c] if c is not None else []),
    }


@rpc_server.method("multiply", "Multiply a list of numbers")
def multiply(numbers: List[float]) -> Dict[str, float]:
    """
    Multiply a list of numbers together.

    Args:
        numbers: List of numbers to multiply

    Returns:
        Dictionary containing the product and operation details
    """
    if not numbers:
        return {"product": 0, "operation": "multiplication", "operands": []}

    result = 1
    for num in numbers:
        result *= num

    return {"product": result, "operation": "multiplication", "operands": numbers}


@rpc_server.method("statistics", "Calculate basic statistics for a dataset")
def calculate_statistics(
    data: List[float], include_variance: bool = False
) -> Dict[str, float]:
    """
    Calculate basic statistics for a dataset.

    Args:
        data: List of numerical values
        include_variance: Whether to include variance and standard deviation

    Returns:
        Dictionary containing statistical measures
    """
    if not data:
        return {"error": "Empty dataset"}

    n = len(data)
    mean = sum(data) / n
    sorted_data = sorted(data)

    # Calculate median
    if n % 2 == 0:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        median = sorted_data[n // 2]

    result = {
        "count": n,
        "mean": mean,
        "median": median,
        "min": min(data),
        "max": max(data),
        "sum": sum(data),
    }

    if include_variance:
        variance = sum((x - mean) ** 2 for x in data) / n
        result["variance"] = variance
        result["std_dev"] = variance**0.5

    return result


@rpc_server.method("greet_user", "Greet a user with customizable message")
def greet_user(
    name: str, title: Optional[str] = None, language: str = "en"
) -> Dict[str, str]:
    """
    Greet a user with a customizable message.

    Args:
        name: User's name
        title: Optional title (Mr., Ms., Dr., etc.)
        language: Language code for greeting (en, es, fr)

    Returns:
        Dictionary containing greeting information
    """
    greetings = {"en": "Hello", "es": "Hola", "fr": "Bonjour"}

    greeting = greetings.get(language, "Hello")
    full_name = f"{title} {name}" if title else name

    return {
        "greeting": f"{greeting}, {full_name}!",
        "user": name,
        "language": language,
        "formal": title is not None,
    }


@rpc_server.method("agent_info", "Get information about this agent")
def get_agent_info() -> Dict[str, any]:
    """
    Get comprehensive information about this agent.

    Returns:
        Dictionary containing agent metadata and capabilities
    """
    manifest = rpc_server.get_manifest()

    return {
        "name": manifest["name"],
        "version": manifest["version"],
        "description": manifest["description"],
        "method_count": len(manifest["methods"]),
        "capability_count": len(manifest["capabilities"]),
        "generated_at": manifest["generated_at"],
        "available_methods": list(manifest["methods"].keys()),
    }


# FastAPI endpoints
@app.post("/rpc")
async def json_rpc_endpoint(request: Request):
    """Main JSON-RPC endpoint."""
    return await rpc_server.handle_request(request)


@app.get("/.well-known/agent.json")
async def agent_manifest():
    """Enhanced agent manifest endpoint."""
    return rpc_server.get_manifest()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Enhanced Calculator Agent"}


@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint."""
    manifest = rpc_server.get_manifest()
    return {
        "methods_registered": len(manifest["methods"]),
        "capabilities": len(manifest["capabilities"]),
        "uptime": "running",
    }


@app.get("/")
async def root():
    """Root endpoint with comprehensive service information."""
    manifest = rpc_server.get_manifest()

    return {
        "service": manifest["name"],
        "version": manifest["version"],
        "description": manifest["description"],
        "protocol": manifest["protocol"],
        "endpoints": {
            "rpc": "/rpc",
            "manifest": "/.well-known/agent.json",
            "health": "/health",
            "metrics": "/metrics",
        },
        "methods": list(manifest["methods"].keys()),
        "capabilities": [cap["name"] for cap in manifest["capabilities"]],
        "metadata": manifest["metadata"],
        "examples": {
            "add_numbers": {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "add",
                "params": {"a": 10, "b": 5, "c": 3},
            },
            "calculate_stats": {
                "jsonrpc": "2.0",
                "id": "2",
                "method": "statistics",
                "params": {"data": [1, 2, 3, 4, 5], "include_variance": True},
            },
            "greet_user": {
                "jsonrpc": "2.0",
                "id": "3",
                "method": "greet_user",
                "params": {"name": "Alice", "title": "Dr.", "language": "en"},
            },
        },
    }


# Agent discovery demo
async def demo_agent_discovery():
    """Demonstrate agent discovery and remote method calling."""
    print("\n=== Agent Discovery Demo ===")

    discovery = AgentDiscovery()

    # Discover this agent (assuming it's running on localhost:8000)
    agent_url = "http://localhost:8000"
    manifest = await discovery.discover_agent(agent_url)

    if manifest:
        print(f"Discovered agent: {manifest.name} v{manifest.version}")
        print(f"Description: {manifest.description}")
        print(f"Available methods: {list(manifest.methods.keys())}")

        # Try calling a method
        try:
            response = await discovery.call_agent_method(
                agent_url, "add", {"a": 15, "b": 25}, "discovery-test-1"
            )
            print(f"Remote method call result: {response.result}")
        except Exception as e:
            print(f"Remote method call failed: {e}")
    else:
        print("Failed to discover agent")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Agent Manifest Demo")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument(
        "--discovery", action="store_true", help="Run agent discovery demo"
    )

    args = parser.parse_args()

    if args.discovery:
        asyncio.run(demo_agent_discovery())
    else:
        print(f"Starting Enhanced Agent Manifest Demo on port {args.port}")
        print(
            f"Manifest available at: http://localhost:{args.port}/.well-known/agent.json"
        )
        print(f"Service info at: http://localhost:{args.port}/")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
