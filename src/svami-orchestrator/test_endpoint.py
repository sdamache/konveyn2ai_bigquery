#!/usr/bin/env python3
"""
Quick test script for the Svami orchestrator /answer endpoint.
This validates the endpoint structure and response format.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from main import app
from common.models import QueryRequest, AnswerResponse


def test_answer_endpoint():
    """Test the /answer endpoint with a sample query."""
    client = TestClient(app)

    # Test data
    test_query = {
        "question": "How do I implement authentication middleware?",
        "role": "backend_developer",
    }

    # Mock authentication for testing
    headers = {"Authorization": "Bearer demo-token"}

    # Make request to /answer endpoint
    response = client.post("/answer", json=test_query, headers=headers)

    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! Response: {data}")

        # Validate response structure
        answer_response = AnswerResponse(**data)
        print(f"✅ Response validation passed!")
        print(f"Answer: {answer_response.answer}")
        print(f"Sources: {answer_response.sources}")
        print(f"Request ID: {answer_response.request_id}")

    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")


def test_agent_manifest():
    """Test the agent manifest endpoint."""
    client = TestClient(app)

    response = client.get("/.well-known/agent.json")
    print(f"\nAgent Manifest Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Agent manifest: {data['name']} v{data['version']}")
        print(f"Methods: {list(data['methods'].keys())}")
    else:
        print(f"❌ Manifest error: {response.text}")


if __name__ == "__main__":
    print("Testing Svami Orchestrator endpoints...")
    test_agent_manifest()
    test_answer_endpoint()
    print("✅ All tests completed!")
