#!/usr/bin/env python3
"""
Test script for the Svami orchestrator workflow logic.
This creates mock RPC clients to test the orchestration flow.
"""

import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from main import app
from common.models import JsonRpcResponse, JsonRpcError


def create_mock_clients():
    """Create mock RPC clients for testing."""
    # Mock successful Janapada response
    mock_janapada = AsyncMock()
    mock_janapada.call.return_value = JsonRpcResponse(
        id="test-123",
        result={
            "snippets": [
                {
                    "file_path": "src/auth/middleware.py",
                    "content": "def authenticate_request(token: str) -> bool:\n    return validate_token(token)",
                },
                {
                    "file_path": "src/auth/tokens.py",
                    "content": "def validate_token(token: str) -> bool:\n    # Token validation logic\n    return True",
                },
            ]
        },
    )

    # Mock successful Amatya response
    mock_amatya = AsyncMock()
    mock_amatya.call.return_value = JsonRpcResponse(
        id="test-123",
        result={
            "answer": "To implement authentication middleware, you should create a FastAPI middleware function that validates tokens before processing requests. Based on the code snippets I found, you can use the authenticate_request function to validate Bearer tokens and implement proper error handling for unauthorized requests."
        },
    )

    return mock_janapada, mock_amatya


def test_successful_orchestration():
    """Test the complete orchestration workflow with successful responses."""
    print("Testing successful orchestration workflow...")

    # Import main module and replace clients with mocks
    import main

    mock_janapada, mock_amatya = create_mock_clients()
    main.janapada_client = mock_janapada
    main.amatya_client = mock_amatya

    client = TestClient(app)

    # Test data
    test_query = {
        "question": "How do I implement authentication middleware?",
        "role": "backend_developer",
    }

    headers = {"Authorization": "Bearer demo-token"}

    # Make request
    response = client.post("/answer", json=test_query, headers=headers)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! Full orchestration workflow completed")
        print(f"Answer length: {len(data['answer'])} characters")
        print(f"Sources: {data['sources']}")
        print(f"Request ID: {data['request_id']}")

        # Verify the calls were made
        assert mock_janapada.call.called, "Janapada service should have been called"
        assert mock_amatya.call.called, "Amatya service should have been called"

        # Verify call parameters
        janapada_call = mock_janapada.call.call_args
        assert janapada_call[1]["method"] == "search", "Should call search method"
        assert "query" in janapada_call[1]["params"], "Should include query parameter"

        amatya_call = mock_amatya.call.call_args
        assert amatya_call[1]["method"] == "advise", "Should call advise method"
        assert "role" in amatya_call[1]["params"], "Should include role parameter"
        assert "chunks" in amatya_call[1]["params"], "Should include chunks parameter"

        print("✅ All workflow validations passed!")

    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")


def test_janapada_failure():
    """Test graceful handling when Janapada fails."""
    print("\nTesting Janapada failure scenario...")

    import main

    # Mock failed Janapada response
    mock_janapada = AsyncMock()
    mock_janapada.call.return_value = JsonRpcResponse(
        id="test-123",
        error=JsonRpcError(
            code=-32004,
            message="External service error",
            data={"reason": "Janapada service unavailable"},
        ),
    )

    mock_amatya = AsyncMock()  # Won't be called
    main.janapada_client = mock_janapada
    main.amatya_client = mock_amatya

    client = TestClient(app)
    test_query = {
        "question": "How do I implement authentication middleware?",
        "role": "backend_developer",
    }
    headers = {"Authorization": "Bearer demo-token"}

    response = client.post("/answer", json=test_query, headers=headers)

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Graceful error handling: {data['answer'][:100]}...")
        assert "sorry" in data["answer"].lower(), "Should contain apology"
        assert (
            not mock_amatya.call.called
        ), "Amatya should not be called if Janapada fails"
        print("✅ Janapada failure test passed!")
    else:
        print(f"❌ Unexpected error: {response.status_code}")


def test_no_snippets_found():
    """Test behavior when no snippets are found."""
    print("\nTesting no snippets scenario...")

    import main

    # Mock Janapada with empty results
    mock_janapada = AsyncMock()
    mock_janapada.call.return_value = JsonRpcResponse(
        id="test-123", result={"snippets": []}
    )

    mock_amatya = AsyncMock()  # Won't be called
    main.janapada_client = mock_janapada
    main.amatya_client = mock_amatya

    client = TestClient(app)
    test_query = {"question": "some very obscure question", "role": "developer"}
    headers = {"Authorization": "Bearer demo-token"}

    response = client.post("/answer", json=test_query, headers=headers)

    if response.status_code == 200:
        data = response.json()
        print(f"✅ No snippets handling: {data['answer'][:100]}...")
        assert (
            "couldn't find" in data["answer"].lower()
        ), "Should indicate no results found"
        assert data["sources"] == [], "Should have empty sources"
        assert not mock_amatya.call.called, "Amatya should not be called if no snippets"
        print("✅ No snippets test passed!")


if __name__ == "__main__":
    print("Testing Svami Orchestrator workflow logic...")
    test_successful_orchestration()
    test_janapada_failure()
    test_no_snippets_found()
    print("\n✅ All orchestration tests completed!")
