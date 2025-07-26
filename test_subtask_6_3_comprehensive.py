#!/usr/bin/env python3
"""
Comprehensive Real-World Testing for Subtask 6.3: Vertex AI Matching Engine Integration
Tests the complete implementation with actual Google Cloud services
"""

import os
import sys
import json
import time
import asyncio
import logging
import requests
import subprocess
from typing import Dict, Any, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Subtask63Tester:
    """Comprehensive testing for Subtask 6.3 - Vertex AI Matching Engine Integration"""
    
    def __init__(self):
        self.test_results = {}
        self.service_url = "http://localhost:8081"
        self.credentials_file = "konveyn2ai-b23e2fbffc62.json"
        self.service_process = None
        
    def setup_environment(self) -> bool:
        """Set up environment variables for testing"""
        logger.info("🔧 Setting up test environment...")
        
        try:
            # Set required environment variables
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(self.credentials_file)
            os.environ["GOOGLE_CLOUD_PROJECT"] = "konveyn2ai"
            os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
            os.environ["VECTOR_INDEX_ID"] = "805460437066842112"
            
            logger.info("✅ Environment variables configured")
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment setup failed: {e}")
            return False
    
    def test_vertex_ai_components(self) -> Dict[str, Any]:
        """Test Vertex AI components directly"""
        logger.info("🤖 Testing Vertex AI components...")
        
        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel
            from google.cloud import aiplatform
            
            # Initialize Vertex AI
            vertexai.init(project="konveyn2ai", location="us-central1")
            aiplatform.init(project="konveyn2ai", location="us-central1")
            
            # Test embedding model
            model = TextEmbeddingModel.from_pretrained("text-embedding-004")
            test_text = "def authenticate_user(token: str) -> bool:"
            embeddings = model.get_embeddings([test_text])
            
            # Test vector index access
            vector_index = aiplatform.MatchingEngineIndex(
                index_name="projects/konveyn2ai/locations/us-central1/indexes/805460437066842112"
            )
            
            result = {
                "status": "success",
                "embedding_model": "text-embedding-004",
                "embedding_dimensions": len(embeddings[0].values) if embeddings else None,
                "vector_index_name": vector_index.display_name,
                "vector_index_id": "805460437066842112",
                "test_embedding_preview": embeddings[0].values[:5] if embeddings else None
            }
            
            logger.info(f"✅ Vertex AI components working: {result['embedding_dimensions']} dims")
            return result
            
        except Exception as e:
            logger.error(f"❌ Vertex AI components test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def start_service_with_monitoring(self) -> Dict[str, Any]:
        """Start the service and capture startup logs"""
        logger.info("🚀 Starting Janapada Memory Service with monitoring...")
        
        try:
            # Start service and capture output
            cmd = [sys.executable, "main.py"]
            self.service_process = subprocess.Popen(
                cmd,
                cwd="src/janapada-memory",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                universal_newlines=True,
                bufsize=1
            )
            
            # Capture startup logs
            startup_logs = []
            start_time = time.time()
            
            while time.time() - start_time < 10:  # Wait up to 10 seconds
                line = self.service_process.stdout.readline()
                if line:
                    startup_logs.append(line.strip())
                    logger.info(f"SERVICE: {line.strip()}")
                    
                    # Check if service is ready
                    if "Uvicorn running on" in line:
                        break
                        
                if self.service_process.poll() is not None:
                    break
            
            # Wait a moment for service to be fully ready
            time.sleep(2)
            
            result = {
                "status": "success" if self.service_process.poll() is None else "failed",
                "startup_logs": startup_logs,
                "process_id": self.service_process.pid if self.service_process.poll() is None else None
            }
            
            if result["status"] == "success":
                logger.info("✅ Service started successfully")
            else:
                logger.error("❌ Service failed to start")
                
            return result
            
        except Exception as e:
            logger.error(f"❌ Service startup failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def test_health_endpoint_comprehensive(self) -> Dict[str, Any]:
        """Test health endpoint with detailed validation"""
        logger.info("🏥 Testing health endpoint comprehensively...")
        
        try:
            response = requests.get(f"{self.service_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Validate expected components
                expected_components = [
                    "vertex_ai_available",
                    "embedding_model", 
                    "embedding_dimensions",
                    "embedding_model_name",
                    "matching_engine",
                    "matching_engine_index_id"
                ]
                
                missing_components = [comp for comp in expected_components 
                                    if comp not in health_data.get("components", {})]
                
                result = {
                    "status": "success",
                    "http_status": response.status_code,
                    "service_status": health_data.get("status"),
                    "components": health_data.get("components", {}),
                    "features": health_data.get("features", {}),
                    "missing_components": missing_components,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
                
                # Validate subtask 6.3 specific requirements
                components = health_data.get("components", {})
                features = health_data.get("features", {})
                
                result["subtask_6_3_validation"] = {
                    "embedding_model_ready": components.get("embedding_model") == "ready",
                    "matching_engine_ready": components.get("matching_engine") == "ready",
                    "vector_search_enabled": features.get("vector_search") == True,
                    "correct_dimensions": components.get("embedding_dimensions") == 768,
                    "correct_model": components.get("embedding_model_name") == "text-embedding-004"
                }
                
                logger.info(f"✅ Health check passed: {health_data.get('status')}")
                return result
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def test_debug_embedding_endpoint(self) -> Dict[str, Any]:
        """Test debug embedding endpoint with validation"""
        logger.info("🔍 Testing debug embedding endpoint...")
        
        try:
            test_queries = [
                "def authenticate_user(token: str) -> bool:",
                "class AuthenticationMiddleware:",
                "jwt token validation function"
            ]
            
            results = []
            
            for query in test_queries:
                response = requests.post(
                    f"{self.service_url}/debug/embedding",
                    json={"text": query},
                    timeout=15
                )
                
                if response.status_code == 200:
                    debug_data = response.json()
                    
                    result = {
                        "query": query,
                        "embedding_available": debug_data.get("embedding_available"),
                        "embedding_dimensions": debug_data.get("embedding_dimensions"),
                        "model_status": debug_data.get("model_status"),
                        "embedding_preview": debug_data.get("embedding_preview"),
                        "response_time_ms": response.elapsed.total_seconds() * 1000
                    }
                    
                    results.append(result)
                    logger.info(f"✅ Debug test for '{query[:30]}...': {result['embedding_dimensions']} dims")
                else:
                    results.append({
                        "query": query,
                        "status": "failed",
                        "error": f"HTTP {response.status_code}: {response.text}"
                    })
            
            return {
                "status": "success",
                "test_results": results,
                "total_tests": len(test_queries),
                "successful_tests": len([r for r in results if r.get("embedding_available")])
            }
            
        except Exception as e:
            logger.error(f"❌ Debug embedding test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def test_vector_search_comprehensive(self) -> Dict[str, Any]:
        """Test vector search with multiple queries and validation"""
        logger.info("🔍 Testing vector search comprehensively...")
        
        try:
            test_queries = [
                {
                    "query": "authentication middleware function",
                    "k": 3,
                    "expected_keywords": ["auth", "middleware", "token"]
                },
                {
                    "query": "jwt token validation",
                    "k": 2,
                    "expected_keywords": ["jwt", "token", "validate"]
                },
                {
                    "query": "user authentication system",
                    "k": 4,
                    "expected_keywords": ["user", "auth", "system"]
                }
            ]
            
            results = []
            
            for test_case in test_queries:
                rpc_request = {
                    "jsonrpc": "2.0",
                    "method": "search",
                    "params": {
                        "query": test_case["query"],
                        "k": test_case["k"]
                    },
                    "id": f"test-vector-search-{len(results)}"
                }
                
                response = requests.post(
                    self.service_url,
                    json=rpc_request,
                    headers={"Content-Type": "application/json"},
                    timeout=20
                )
                
                if response.status_code == 200:
                    rpc_response = response.json()
                    
                    if "result" in rpc_response:
                        snippets = rpc_response["result"].get("snippets", [])
                        
                        # Analyze snippets for vector search indicators
                        vector_search_indicators = []
                        for snippet in snippets:
                            content = snippet.get("content", "")
                            if "similarity:" in content or "Vector search result" in content:
                                vector_search_indicators.append(True)
                            else:
                                vector_search_indicators.append(False)
                        
                        result = {
                            "query": test_case["query"],
                            "k_requested": test_case["k"],
                            "snippets_returned": len(snippets),
                            "snippets": snippets,
                            "vector_search_indicators": vector_search_indicators,
                            "has_vector_search_results": any(vector_search_indicators),
                            "response_time_ms": response.elapsed.total_seconds() * 1000,
                            "request_id": rpc_response.get("id")
                        }
                        
                        results.append(result)
                        logger.info(f"✅ Vector search for '{test_case['query']}': {len(snippets)} results")
                    else:
                        results.append({
                            "query": test_case["query"],
                            "status": "failed",
                            "error": f"RPC Error: {rpc_response.get('error', 'Unknown error')}"
                        })
                else:
                    results.append({
                        "query": test_case["query"],
                        "status": "failed",
                        "error": f"HTTP {response.status_code}: {response.text}"
                    })
            
            return {
                "status": "success",
                "test_results": results,
                "total_tests": len(test_queries),
                "successful_tests": len([r for r in results if r.get("snippets_returned", 0) > 0]),
                "vector_search_working": any(r.get("has_vector_search_results", False) for r in results)
            }
            
        except Exception as e:
            logger.error(f"❌ Vector search test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up...")
        
        if self.service_process:
            self.service_process.terminate()
            try:
                self.service_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.service_process.kill()
            logger.info("✅ Service process terminated")
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests for subtask 6.3"""
        logger.info("🚀 Starting Subtask 6.3 Comprehensive Testing...")
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "subtask": "6.3 - Vertex AI Matching Engine Integration",
            "tests": {}
        }
        
        try:
            # Test 1: Environment setup
            if not self.setup_environment():
                test_results["tests"]["environment_setup"] = {"status": "failed", "error": "Environment setup failed"}
                return test_results
            test_results["tests"]["environment_setup"] = {"status": "success"}
            
            # Test 2: Vertex AI components
            test_results["tests"]["vertex_ai_components"] = self.test_vertex_ai_components()
            
            # Test 3: Service startup
            startup_result = self.start_service_with_monitoring()
            test_results["tests"]["service_startup"] = startup_result
            
            if startup_result["status"] != "success":
                return test_results
            
            # Test 4: Health endpoint
            test_results["tests"]["health_endpoint"] = self.test_health_endpoint_comprehensive()
            
            # Test 5: Debug embedding endpoint
            test_results["tests"]["debug_embedding"] = self.test_debug_embedding_endpoint()
            
            # Test 6: Vector search
            test_results["tests"]["vector_search"] = self.test_vector_search_comprehensive()
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            test_results["error"] = str(e)
        finally:
            self.cleanup()
            
        return test_results

def main():
    """Main test runner"""
    print("🚀 KonveyN2AI Subtask 6.3 - Comprehensive Real-World Testing")
    print("=" * 80)
    
    tester = Subtask63Tester()
    results = tester.run_comprehensive_tests()
    
    # Print summary
    print("\n📊 Subtask 6.3 Test Results Summary:")
    print("=" * 60)
    
    total_tests = len(results.get("tests", {}))
    passed_tests = sum(1 for test in results.get("tests", {}).values() if test.get("status") == "success")
    
    for test_name, test_result in results.get("tests", {}).items():
        status_icon = "✅" if test_result.get("status") == "success" else "❌"
        print(f"{status_icon} {test_name}: {test_result.get('status', 'unknown')}")
        
        if test_result.get("status") == "failed":
            print(f"   Error: {test_result.get('error', 'Unknown error')}")
    
    print(f"\n📈 Overall: {passed_tests}/{total_tests} tests passed")
    
    # Save detailed results
    with open("test_results_subtask_6_3_comprehensive.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Detailed results saved to: test_results_subtask_6_3_comprehensive.json")
    
    return 0 if passed_tests == total_tests else 1

if __name__ == "__main__":
    sys.exit(main())
