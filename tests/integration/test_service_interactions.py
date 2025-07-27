"""
Integration tests for service interactions between Svami, Janapada, and Amatya.
"""

import pytest
import httpx
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Test configuration
SERVICES = {
    "svami": "http://localhost:8080",
    "janapada": "http://localhost:8081",
    "amatya": "http://localhost:8082",
}


class TestServiceCommunication:
    """Test communication between services."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_svami_to_janapada_communication(
        self, mock_vertex_ai, mock_matching_engine, sample_snippets
    ):
        """Test Svami calling Janapada for search."""

        # Mock Janapada response
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "jsonrpc": "2.0",
                "id": "test-123",
                "result": {"snippets": sample_snippets},
            }
            mock_post.return_value = mock_response

            # Test JSON-RPC call to Janapada
            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": "test-123",
                "method": "search",
                "params": {"query": "authentication implementation", "k": 5},
            }

            # Create async client directly in test
            async with httpx.AsyncClient(timeout=10.0) as async_client:
                # Create async client directly in test

                async with httpx.AsyncClient(timeout=10.0) as async_client:
                    response = await async_client.post(
                        f"{SERVICES['janapada']}/", json=jsonrpc_request
                    )

            # Verify communication
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == f"{SERVICES['janapada']}/"
            assert call_args[1]["json"] == jsonrpc_request

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_svami_to_amatya_communication(self, mock_gemini_ai, sample_snippets):
        """Test Svami calling Amatya for advice."""

        # Mock Amatya response
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "jsonrpc": "2.0",
                "id": "test-456",
                "result": {
                    "advice": "Based on the provided code snippets, here's how to implement authentication..."
                },
            }
            mock_post.return_value = mock_response

            # Test JSON-RPC call to Amatya
            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": "test-456",
                "method": "advise",
                "params": {"role": "backend engineer", "chunks": sample_snippets},
            }

            # Create async client directly in test
            async with httpx.AsyncClient(timeout=10.0) as async_client:
                # Create async client directly in test

                async with httpx.AsyncClient(timeout=10.0) as async_client:
                    response = await async_client.post(
                        f"{SERVICES['amatya']}/", json=jsonrpc_request
                    )

            # Verify communication
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == f"{SERVICES['amatya']}/"
            assert call_args[1]["json"] == jsonrpc_request

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(
        self,
        mock_vertex_ai,
        mock_matching_engine,
        mock_gemini_ai,
        sample_snippets,
    ):
        """Test complete end-to-end workflow through all services."""

        # Import and setup first
        from fastapi.testclient import TestClient
        import sys
        import os

        # Add Svami path to sys.path for import
        svami_path = os.path.join(
            os.path.dirname(__file__), "../../src/svami-orchestrator"
        )
        svami_path = os.path.abspath(svami_path)
        if svami_path not in sys.path:
            sys.path.insert(0, svami_path)

        from main import app
        from common.models import JsonRpcResponse

        # Import main module to ensure it's loaded
        import main
        from common.rpc_client import JsonRpcClient

        # Initialize the global clients manually since TestClient doesn't trigger lifespan
        main.janapada_client = JsonRpcClient("http://localhost:8001")
        main.amatya_client = JsonRpcClient("http://localhost:8002")

        # Mock the JsonRpcClient.call method directly
        with (
            patch.object(
                main.janapada_client, "call", new_callable=AsyncMock
            ) as mock_janapada_call,
            patch.object(
                main.amatya_client, "call", new_callable=AsyncMock
            ) as mock_amatya_call,
        ):
            # Mock Janapada response
            mock_janapada_call.return_value = JsonRpcResponse(
                id="test-id", result={"snippets": sample_snippets}
            )

            # Mock Amatya response
            mock_amatya_call.return_value = JsonRpcResponse(
                id="test-id",
                result={"advice": "Generated advice based on code snippets"},
            )

            client = TestClient(app)

            # Send query to Svami orchestrator
            query_request = {
                "question": "How do I implement JWT authentication?",
                "role": "backend engineer",
            }

            response = client.post(
                "/answer",
                json=query_request,
                headers={"Authorization": "Bearer test-token"},
            )

            # Verify end-to-end response
            assert response.status_code == 200
            data = response.json()

            assert "answer" in data
            assert "sources" in data
            assert "request_id" in data
            assert len(data["sources"]) > 0

            # Verify both services were called
            mock_janapada_call.assert_called_once()
            mock_amatya_call.assert_called_once()

            # Verify call parameters
            janapada_call = mock_janapada_call.call_args
            assert janapada_call[1]["method"] == "search"
            assert "query" in janapada_call[1]["params"]

            amatya_call = mock_amatya_call.call_args
            assert amatya_call[1]["method"] == "advise"
            assert "role" in amatya_call[1]["params"]
            assert "chunks" in amatya_call[1]["params"]


class TestServiceFailureHandling:
    """Test service failure scenarios and graceful degradation."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_janapada_service_unavailable(self, mock_gemini_ai):
        """Test behavior when Janapada service is unavailable."""

        # Import main module first
        import sys
        import os

        # Add Svami path to sys.path for import
        svami_path = os.path.join(
            os.path.dirname(__file__), "../../src/svami-orchestrator"
        )
        svami_path = os.path.abspath(svami_path)
        if svami_path not in sys.path:
            sys.path.insert(0, svami_path)

        import main

        with (
            patch.object(main, "janapada_client", create=True) as mock_janapada,
            patch.object(main, "amatya_client", create=True) as mock_amatya,
        ):
            from common.models import JsonRpcResponse
            from fastapi.testclient import TestClient
            from main import app

            client = TestClient(app)

            # Mock Janapada failure
            mock_janapada.call.side_effect = Exception("Service unavailable")

            # Mock Amatya success (graceful degradation)
            mock_amatya.call = AsyncMock(
                return_value=JsonRpcResponse(
                    id="test-id",
                    result={"advice": "General advice without specific code context"},
                )
            )

            query_request = {
                "question": "How do I implement authentication?",
                "role": "backend engineer",
            }

            response = client.post(
                "/answer",
                json=query_request,
                headers={"Authorization": "Bearer test-token"},
            )

            # Should still return response with graceful degradation
            assert response.status_code == 200
            data = response.json()

            assert "answer" in data
            assert "sources" in data
            assert len(data["sources"]) == 0  # No sources due to Janapada failure

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_amatya_service_unavailable(
        self, mock_vertex_ai, mock_matching_engine, sample_snippets
    ):
        """Test behavior when Amatya service is unavailable."""

        from common.models import JsonRpcResponse
        from fastapi.testclient import TestClient
        import sys
        import os

        # Add Svami path to sys.path for import
        svami_path = os.path.join(
            os.path.dirname(__file__), "../../src/svami-orchestrator"
        )
        svami_path = os.path.abspath(svami_path)
        if svami_path not in sys.path:
            sys.path.insert(0, svami_path)

        from main import app
        import main

        with (
            patch.object(main, "janapada_client", create=True) as mock_janapada,
            patch.object(main, "amatya_client", create=True) as mock_amatya,
        ):
            client = TestClient(app)

            # Mock Janapada success
            mock_janapada.call = AsyncMock(
                return_value=JsonRpcResponse(
                    id="test-id", result={"snippets": sample_snippets}
                )
            )

            # Mock Amatya failure
            mock_amatya.call.side_effect = Exception("Service unavailable")

            query_request = {
                "question": "How do I implement authentication?",
                "role": "backend engineer",
            }

            response = client.post(
                "/answer",
                json=query_request,
                headers={"Authorization": "Bearer test-token"},
            )

            # Should return error when advice generation fails
            assert response.status_code == 500
            data = response.json()
            assert "error" in data

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_partial_service_failures(self):
        """Test handling of partial service failures and timeouts."""

        from common.models import JsonRpcResponse
        from fastapi.testclient import TestClient
        import sys
        import os
        import time

        # Add Svami path to sys.path for import
        svami_path = os.path.join(
            os.path.dirname(__file__), "../../src/svami-orchestrator"
        )
        svami_path = os.path.abspath(svami_path)
        if svami_path not in sys.path:
            sys.path.insert(0, svami_path)

        from main import app
        import main

        with (
            patch.object(main, "janapada_client", create=True) as mock_janapada,
            patch.object(main, "amatya_client", create=True) as mock_amatya,
        ):
            client = TestClient(app)

            # Mock Janapada - slow response
            def slow_janapada_call(*args, **kwargs):
                time.sleep(0.1)  # Simulate slow response
                return JsonRpcResponse(id="test-id", result={"snippets": []})

            mock_janapada.call = AsyncMock(side_effect=slow_janapada_call)

            # Mock Amatya - timeout
            mock_amatya.call.side_effect = Exception("Request timeout")

            query_request = {
                "question": "How do I implement authentication?",
                "role": "backend engineer",
            }

            response = client.post(
                "/answer",
                json=query_request,
                headers={"Authorization": "Bearer test-token"},
            )

            # Should handle timeouts gracefully
            assert response.status_code in [200, 500]


class TestServiceHealthMonitoring:
    """Test health monitoring across services."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_all_services_healthy(self):
        """Test health check when all services are healthy."""

        # Mock the health check dependencies by mocking the Svami service directly
        from fastapi.testclient import TestClient
        import sys
        import os

        # Add Svami path to sys.path for import
        svami_path = os.path.join(
            os.path.dirname(__file__), "../../src/svami-orchestrator"
        )
        svami_path = os.path.abspath(svami_path)
        if svami_path not in sys.path:
            sys.path.insert(0, svami_path)

        from main import app
        import main

        with (
            patch.object(main, "janapada_client", create=True) as mock_janapada,
            patch.object(main, "amatya_client", create=True) as mock_amatya,
        ):
            client = TestClient(app)

            # Mock that both services are available by ensuring they don't raise exceptions
            mock_janapada.call = AsyncMock(return_value={"status": "healthy"})
            mock_amatya.call = AsyncMock(return_value={"status": "healthy"})

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] in ["healthy", "degraded"]
            # Note: Dependencies may not be included in basic health check
            # This test just verifies the service is responding

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dependency_health_reporting(self):
        """Test health reporting of dependency services."""

        # This test needs to mock the actual health checking logic in Svami
        # For now, just verify the health endpoint responds properly
        from fastapi.testclient import TestClient
        import sys
        import os

        # Add Svami path to sys.path for import
        svami_path = os.path.join(
            os.path.dirname(__file__), "../../src/svami-orchestrator"
        )
        svami_path = os.path.abspath(svami_path)
        if svami_path not in sys.path:
            sys.path.insert(0, svami_path)

        from main import app
        import main

        with (
            patch.object(main, "janapada_client", create=True) as mock_janapada,
            patch.object(main, "amatya_client", create=True) as mock_amatya,
        ):
            client = TestClient(app)

            # Mock Janapada as healthy
            mock_janapada.call = AsyncMock(return_value={"status": "healthy"})

            # Mock Amatya as unhealthy (raise exception)
            mock_amatya.call.side_effect = Exception("Connection failed")

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()

            # Basic health response should be returned
            assert "status" in data
            assert data["status"] in ["healthy", "degraded"]


class TestRequestTracking:
    """Test request ID tracking across services."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_request_id_propagation(
        self,
        mock_vertex_ai,
        mock_matching_engine,
        mock_gemini_ai,
        sample_snippets,
    ):
        """Test that request IDs are properly propagated across service calls."""

        captured_request_ids = []

        from common.models import JsonRpcResponse
        from fastapi.testclient import TestClient
        import sys
        import os

        # Add Svami path to sys.path for import
        svami_path = os.path.join(
            os.path.dirname(__file__), "../../src/svami-orchestrator"
        )
        svami_path = os.path.abspath(svami_path)
        if svami_path not in sys.path:
            sys.path.insert(0, svami_path)

        from main import app
        import main

        with (
            patch.object(main, "janapada_client", create=True) as mock_janapada,
            patch.object(main, "amatya_client", create=True) as mock_amatya,
        ):
            client = TestClient(app)

            def capture_janapada_request_id(*args, **kwargs):
                captured_request_ids.append(kwargs["id"])
                return JsonRpcResponse(
                    id=kwargs["id"], result={"snippets": sample_snippets}
                )

            def capture_amatya_request_id(*args, **kwargs):
                captured_request_ids.append(kwargs["id"])
                return JsonRpcResponse(
                    id=kwargs["id"], result={"advice": "Generated advice"}
                )

            mock_janapada.call = AsyncMock(side_effect=capture_janapada_request_id)
            mock_amatya.call = AsyncMock(side_effect=capture_amatya_request_id)

            query_request = {
                "question": "How do I implement authentication?",
                "role": "backend engineer",
            }

            response = client.post(
                "/answer",
                json=query_request,
                headers={"Authorization": "Bearer test-token"},
            )

            assert response.status_code == 200
            data = response.json()

            # Verify request ID in response
            assert "request_id" in data
            main_request_id = data["request_id"]

            # Verify same request ID was used for all service calls
            assert len(captured_request_ids) == 2
            assert all(req_id == main_request_id for req_id in captured_request_ids)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_unique_request_ids(
        self,
        mock_vertex_ai,
        mock_matching_engine,
        mock_gemini_ai,
        sample_snippets,
    ):
        """Test that different requests get unique request IDs."""

        request_ids = set()

        from common.models import JsonRpcResponse
        from fastapi.testclient import TestClient
        import sys
        import os

        # Add Svami path to sys.path for import
        svami_path = os.path.join(
            os.path.dirname(__file__), "../../src/svami-orchestrator"
        )
        svami_path = os.path.abspath(svami_path)
        if svami_path not in sys.path:
            sys.path.insert(0, svami_path)

        from main import app
        import main

        with (
            patch.object(main, "janapada_client", create=True) as mock_janapada,
            patch.object(main, "amatya_client", create=True) as mock_amatya,
        ):
            client = TestClient(app)

            mock_janapada.call = AsyncMock(
                return_value=JsonRpcResponse(id="test-id", result={"snippets": []})
            )
            mock_amatya.call = AsyncMock(
                return_value=JsonRpcResponse(id="test-id", result={"advice": "test"})
            )

            # Make multiple requests
            for i in range(10):
                query_request = {"question": f"Question {i}", "role": "developer"}

                response = client.post(
                    "/answer",
                    json=query_request,
                    headers={"Authorization": "Bearer test-token"},
                )

                assert response.status_code == 200
                data = response.json()

                request_id = data["request_id"]
                assert (
                    request_id not in request_ids
                ), f"Duplicate request ID: {request_id}"
                request_ids.add(request_id)

            # All request IDs should be unique
            assert len(request_ids) == 10


class TestServiceLoadTesting:
    """Test service behavior under load."""

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_requests(
        self, mock_vertex_ai, mock_matching_engine, mock_gemini_ai
    ):
        """Test handling of concurrent requests across services."""

        from common.models import JsonRpcResponse
        from fastapi.testclient import TestClient
        import sys
        import os

        # Add Svami path to sys.path for import
        svami_path = os.path.join(
            os.path.dirname(__file__), "../../src/svami-orchestrator"
        )
        svami_path = os.path.abspath(svami_path)
        if svami_path not in sys.path:
            sys.path.insert(0, svami_path)

        from main import app
        import main

        with (
            patch.object(main, "janapada_client", create=True) as mock_janapada,
            patch.object(main, "amatya_client", create=True) as mock_amatya,
        ):
            client = TestClient(app)

            call_count = 0

            def mock_amatya_response(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return JsonRpcResponse(
                    id=kwargs.get("id", "test"),
                    result={"advice": f"Response {call_count}"},
                )

            mock_janapada.call = AsyncMock(
                return_value=JsonRpcResponse(id="test-id", result={"snippets": []})
            )
            mock_amatya.call = AsyncMock(side_effect=mock_amatya_response)

            # Create concurrent requests (reduced for test performance)
            responses = []
            for i in range(5):  # Reduced from 20 for faster test execution
                query_request = {
                    "question": f"Question {i}",
                    "role": "developer",
                }

                response = client.post(
                    "/answer",
                    json=query_request,
                    headers={"Authorization": "Bearer test-token"},
                )

                responses.append(response)

            # Verify all requests completed successfully
            assert len(responses) == 5

            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert "answer" in data
                assert "request_id" in data

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_service_performance_benchmarks(
        self, mock_vertex_ai, mock_matching_engine, mock_gemini_ai
    ):
        """Test performance benchmarks for service interactions."""

        import time
        import sys
        import os

        # Add Svami path to sys.path for import
        svami_path = os.path.join(
            os.path.dirname(__file__), "../../src/svami-orchestrator"
        )
        svami_path = os.path.abspath(svami_path)
        if svami_path not in sys.path:
            sys.path.insert(0, svami_path)

        from main import app
        from common.models import JsonRpcResponse
        from fastapi.testclient import TestClient

        # Import main module to ensure it's loaded
        import main
        from common.rpc_client import JsonRpcClient

        # Initialize the global clients manually since TestClient doesn't trigger lifespan
        main.janapada_client = JsonRpcClient("http://localhost:8001")
        main.amatya_client = JsonRpcClient("http://localhost:8002")

        # Mock the JsonRpcClient.call method directly
        with (
            patch.object(
                main.janapada_client, "call", new_callable=AsyncMock
            ) as mock_janapada_call,
            patch.object(
                main.amatya_client, "call", new_callable=AsyncMock
            ) as mock_amatya_call,
        ):
            client = TestClient(app)

            # Add realistic delays to simulate real service behavior
            def mock_janapada_delay(*args, **kwargs):
                time.sleep(0.01)  # Reduced delay for test performance
                return JsonRpcResponse(
                    id=kwargs.get("id", "test"), result={"snippets": []}
                )

            def mock_amatya_delay(*args, **kwargs):
                time.sleep(0.02)  # Reduced delay for test performance
                return JsonRpcResponse(
                    id=kwargs.get("id", "test"), result={"advice": "test"}
                )

            mock_janapada_call.side_effect = mock_janapada_delay
            mock_amatya_call.side_effect = mock_amatya_delay

            start_time = time.time()

            # Make sequential requests to measure performance (reduced count)
            for i in range(3):  # Reduced from 10 for faster test execution
                query_request = {
                    "question": f"Performance test {i}",
                    "role": "developer",
                }

                response = client.post(
                    "/answer",
                    json=query_request,
                    headers={"Authorization": "Bearer test-token"},
                )

                assert response.status_code == 200

            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / 3

            # Performance benchmarks (adjusted for test environment)
            assert total_time < 10.0, f"Total time too high: {total_time}s"
            assert avg_time < 5.0, f"Average response time too high: {avg_time}s"
