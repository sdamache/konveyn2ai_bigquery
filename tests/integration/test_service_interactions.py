"""
Integration tests for service interactions between Svami, Janapada, and Amatya.
"""

import pytest
import httpx
import asyncio
from unittest.mock import patch, MagicMock

# Test configuration
SERVICES = {
    "svami": "http://localhost:8080",
    "janapada": "http://localhost:8081", 
    "amatya": "http://localhost:8082"
}


class TestServiceCommunication:
    """Test communication between services."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_svami_to_janapada_communication(self, async_client, mock_vertex_ai, mock_matching_engine, sample_snippets):
        """Test Svami calling Janapada for search."""
        
        # Mock Janapada response
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "jsonrpc": "2.0",
                "id": "test-123",
                "result": {
                    "snippets": sample_snippets
                }
            }
            mock_post.return_value = mock_response
            
            # Test JSON-RPC call to Janapada
            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": "test-123",
                "method": "search",
                "params": {
                    "query": "authentication implementation",
                    "k": 5
                }
            }
            
            response = await async_client.post(
                f"{SERVICES['janapada']}/",
                json=jsonrpc_request
            )
            
            # Verify communication
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == f"{SERVICES['janapada']}/"
            assert call_args[1]["json"] == jsonrpc_request

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_svami_to_amatya_communication(self, async_client, mock_gemini_ai, sample_snippets):
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
                }
            }
            mock_post.return_value = mock_response
            
            # Test JSON-RPC call to Amatya
            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": "test-456",
                "method": "advise",
                "params": {
                    "role": "backend engineer",
                    "chunks": sample_snippets
                }
            }
            
            response = await async_client.post(
                f"{SERVICES['amatya']}/",
                json=jsonrpc_request
            )
            
            # Verify communication
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == f"{SERVICES['amatya']}/"
            assert call_args[1]["json"] == jsonrpc_request

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, async_client, mock_vertex_ai, mock_matching_engine, mock_gemini_ai, sample_snippets):
        """Test complete end-to-end workflow through all services."""
        
        # Mock both Janapada and Amatya responses
        with patch("httpx.AsyncClient.post") as mock_post:
            def mock_response_factory(url, **kwargs):
                response = MagicMock()
                response.status_code = 200
                
                # Determine which service is being called based on URL
                if "8081" in url:  # Janapada
                    response.json.return_value = {
                        "jsonrpc": "2.0",
                        "id": kwargs["json"]["id"],
                        "result": {"snippets": sample_snippets}
                    }
                elif "8082" in url:  # Amatya  
                    response.json.return_value = {
                        "jsonrpc": "2.0",
                        "id": kwargs["json"]["id"],
                        "result": {"advice": "Generated advice based on code snippets"}
                    }
                
                return response
            
            mock_post.side_effect = mock_response_factory
            
            # Send query to Svami orchestrator
            query_request = {
                "question": "How do I implement JWT authentication?",
                "role": "backend engineer"
            }
            
            response = await async_client.post(
                f"{SERVICES['svami']}/answer",
                json=query_request,
                headers={"Authorization": "Bearer test-token"}
            )
            
            # Verify end-to-end response
            assert response.status_code == 200
            data = response.json()
            
            assert "answer" in data
            assert "sources" in data
            assert "request_id" in data
            assert len(data["sources"]) > 0
            
            # Verify both services were called
            assert mock_post.call_count == 2
            
            # Verify call order and parameters
            calls = mock_post.call_args_list
            
            # First call should be to Janapada
            janapada_call = calls[0]
            assert "8081" in janapada_call[0][0]
            assert janapada_call[1]["json"]["method"] == "search"
            
            # Second call should be to Amatya
            amatya_call = calls[1]
            assert "8082" in amatya_call[0][0]
            assert amatya_call[1]["json"]["method"] == "advise"


class TestServiceFailureHandling:
    """Test service failure scenarios and graceful degradation."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_janapada_service_unavailable(self, async_client, mock_gemini_ai):
        """Test behavior when Janapada service is unavailable."""
        
        with patch("httpx.AsyncClient.post") as mock_post:
            def mock_response_factory(url, **kwargs):
                if "8081" in url:  # Janapada
                    raise httpx.ConnectError("Connection failed")
                elif "8082" in url:  # Amatya
                    response = MagicMock()
                    response.status_code = 200
                    response.json.return_value = {
                        "jsonrpc": "2.0",
                        "id": kwargs["json"]["id"],
                        "result": {"advice": "General advice without specific code context"}
                    }
                    return response
            
            mock_post.side_effect = mock_response_factory
            
            query_request = {
                "question": "How do I implement authentication?",
                "role": "backend engineer"
            }
            
            response = await async_client.post(
                f"{SERVICES['svami']}/answer",
                json=query_request,
                headers={"Authorization": "Bearer test-token"}
            )
            
            # Should still return response with graceful degradation
            assert response.status_code == 200
            data = response.json()
            
            assert "answer" in data
            assert "sources" in data
            assert len(data["sources"]) == 0  # No sources due to Janapada failure

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_amatya_service_unavailable(self, async_client, mock_vertex_ai, mock_matching_engine, sample_snippets):
        """Test behavior when Amatya service is unavailable."""
        
        with patch("httpx.AsyncClient.post") as mock_post:
            def mock_response_factory(url, **kwargs):
                if "8081" in url:  # Janapada
                    response = MagicMock()
                    response.status_code = 200
                    response.json.return_value = {
                        "jsonrpc": "2.0",
                        "id": kwargs["json"]["id"],
                        "result": {"snippets": sample_snippets}
                    }
                    return response
                elif "8082" in url:  # Amatya
                    raise httpx.ConnectError("Connection failed")
            
            mock_post.side_effect = mock_response_factory
            
            query_request = {
                "question": "How do I implement authentication?",
                "role": "backend engineer"
            }
            
            response = await async_client.post(
                f"{SERVICES['svami']}/answer",
                json=query_request,
                headers={"Authorization": "Bearer test-token"}
            )
            
            # Should return error when advice generation fails
            assert response.status_code == 500
            data = response.json()
            assert "error" in data

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_partial_service_failures(self, async_client):
        """Test handling of partial service failures and timeouts."""
        
        with patch("httpx.AsyncClient.post") as mock_post:
            def mock_response_factory(url, **kwargs):
                if "8081" in url:  # Janapada - slow response
                    import time
                    time.sleep(0.1)  # Simulate slow response
                    response = MagicMock()
                    response.status_code = 200
                    response.json.return_value = {
                        "jsonrpc": "2.0",
                        "id": kwargs["json"]["id"],
                        "result": {"snippets": []}
                    }
                    return response
                elif "8082" in url:  # Amatya - timeout
                    raise httpx.ReadTimeout("Request timeout")
            
            mock_post.side_effect = mock_response_factory
            
            query_request = {
                "question": "How do I implement authentication?",
                "role": "backend engineer"
            }
            
            response = await async_client.post(
                f"{SERVICES['svami']}/answer",
                json=query_request,
                headers={"Authorization": "Bearer test-token"}
            )
            
            # Should handle timeouts gracefully
            assert response.status_code in [200, 500]


class TestServiceHealthMonitoring:
    """Test health monitoring across services."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_all_services_healthy(self, async_client):
        """Test health check when all services are healthy."""
        
        with patch("httpx.AsyncClient.get") as mock_get:
            # Mock healthy responses from all services
            healthy_response = MagicMock()
            healthy_response.status_code = 200
            healthy_response.json.return_value = {"status": "healthy"}
            mock_get.return_value = healthy_response
            
            response = await async_client.get(f"{SERVICES['svami']}/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] in ["healthy", "degraded"]
            assert "dependencies" in data

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dependency_health_reporting(self, async_client):
        """Test health reporting of dependency services."""
        
        with patch("httpx.AsyncClient.get") as mock_get:
            def mock_health_response(url, **kwargs):
                response = MagicMock()
                
                if "8081" in url:  # Janapada - healthy
                    response.status_code = 200
                    response.json.return_value = {"status": "healthy"}
                elif "8082" in url:  # Amatya - unhealthy
                    raise httpx.ConnectError("Connection failed")
                
                return response
            
            mock_get.side_effect = mock_health_response
            
            response = await async_client.get(f"{SERVICES['svami']}/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "degraded"
            assert len(data["dependencies"]) == 2
            
            # Check individual service health
            dep_statuses = {dep["name"]: dep["status"] for dep in data["dependencies"]}
            assert dep_statuses.get("janapada") == "healthy"
            assert dep_statuses.get("amatya") == "unhealthy"


class TestRequestTracking:
    """Test request ID tracking across services."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_request_id_propagation(self, async_client, mock_vertex_ai, mock_matching_engine, mock_gemini_ai, sample_snippets):
        """Test that request IDs are properly propagated across service calls."""
        
        captured_request_ids = []
        
        with patch("httpx.AsyncClient.post") as mock_post:
            def capture_request_id(url, **kwargs):
                # Capture request ID from JSON-RPC calls
                if "json" in kwargs and "id" in kwargs["json"]:
                    captured_request_ids.append(kwargs["json"]["id"])
                
                response = MagicMock()
                response.status_code = 200
                
                if "8081" in url:  # Janapada
                    response.json.return_value = {
                        "jsonrpc": "2.0",
                        "id": kwargs["json"]["id"],
                        "result": {"snippets": sample_snippets}
                    }
                elif "8082" in url:  # Amatya
                    response.json.return_value = {
                        "jsonrpc": "2.0", 
                        "id": kwargs["json"]["id"],
                        "result": {"advice": "Generated advice"}
                    }
                
                return response
            
            mock_post.side_effect = capture_request_id
            
            query_request = {
                "question": "How do I implement authentication?",
                "role": "backend engineer"
            }
            
            response = await async_client.post(
                f"{SERVICES['svami']}/answer",
                json=query_request,
                headers={"Authorization": "Bearer test-token"}
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
    async def test_unique_request_ids(self, async_client, mock_vertex_ai, mock_matching_engine, mock_gemini_ai, sample_snippets):
        """Test that different requests get unique request IDs."""
        
        request_ids = set()
        
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "jsonrpc": "2.0",
                "id": "test",
                "result": {"snippets": [], "advice": "test"}
            }
            mock_post.return_value = mock_response
            
            # Make multiple requests
            for i in range(10):
                query_request = {
                    "question": f"Question {i}",
                    "role": "developer"
                }
                
                response = await async_client.post(
                    f"{SERVICES['svami']}/answer",
                    json=query_request,
                    headers={"Authorization": "Bearer test-token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                request_id = data["request_id"]
                assert request_id not in request_ids, f"Duplicate request ID: {request_id}"
                request_ids.add(request_id)
            
            # All request IDs should be unique
            assert len(request_ids) == 10


class TestServiceLoadTesting:
    """Test service behavior under load."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client, mock_vertex_ai, mock_matching_engine, mock_gemini_ai):
        """Test handling of concurrent requests across services."""
        
        with patch("httpx.AsyncClient.post") as mock_post:
            call_count = 0
            
            def mock_response_factory(url, **kwargs):
                nonlocal call_count
                call_count += 1
                
                response = MagicMock()
                response.status_code = 200
                
                if "8081" in url:  # Janapada
                    response.json.return_value = {
                        "jsonrpc": "2.0",
                        "id": kwargs["json"]["id"],
                        "result": {"snippets": []}
                    }
                elif "8082" in url:  # Amatya
                    response.json.return_value = {
                        "jsonrpc": "2.0",
                        "id": kwargs["json"]["id"],
                        "result": {"advice": f"Response {call_count}"}
                    }
                
                return response
            
            mock_post.side_effect = mock_response_factory
            
            # Create concurrent requests
            async def make_request(request_id):
                query_request = {
                    "question": f"Question {request_id}",
                    "role": "developer"
                }
                
                response = await async_client.post(
                    f"{SERVICES['svami']}/answer",
                    json=query_request,
                    headers={"Authorization": "Bearer test-token"}
                )
                
                return request_id, response.status_code, response.json()
            
            # Execute concurrent requests
            tasks = [make_request(i) for i in range(20)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all requests completed successfully
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) == 20
            
            for request_id, status_code, response_data in successful_results:
                assert status_code == 200
                assert "answer" in response_data
                assert "request_id" in response_data

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio 
    async def test_service_performance_benchmarks(self, async_client, mock_vertex_ai, mock_matching_engine, mock_gemini_ai):
        """Test performance benchmarks for service interactions."""
        
        import time
        
        with patch("httpx.AsyncClient.post") as mock_post:
            # Add realistic delays to simulate real service behavior
            def mock_delayed_response(url, **kwargs):
                if "8081" in url:  # Janapada - vector search delay
                    time.sleep(0.1)
                elif "8082" in url:  # Amatya - AI model delay
                    time.sleep(0.2)
                
                response = MagicMock()
                response.status_code = 200
                response.json.return_value = {
                    "jsonrpc": "2.0",
                    "id": kwargs["json"]["id"],
                    "result": {"snippets": [], "advice": "test"}
                }
                return response
            
            mock_post.side_effect = mock_delayed_response
            
            start_time = time.time()
            
            # Make sequential requests to measure performance
            for i in range(10):
                query_request = {
                    "question": f"Performance test {i}",
                    "role": "developer"
                }
                
                response = await async_client.post(
                    f"{SERVICES['svami']}/answer",
                    json=query_request,
                    headers={"Authorization": "Bearer test-token"}
                )
                
                assert response.status_code == 200
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / 10
            
            # Performance benchmarks (adjust based on requirements)
            assert total_time < 50.0, f"Total time too high: {total_time}s"
            assert avg_time < 5.0, f"Average response time too high: {avg_time}s"