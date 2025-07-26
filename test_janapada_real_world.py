#!/usr/bin/env python3
"""
Comprehensive Real-World Testing for Janapada Memory Service
Tests actual Google Cloud service integration for subtasks 6.1 and 6.2
"""

import os
import sys
import json
import asyncio
import logging
import requests
import subprocess
from typing import Dict, Any, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class JanapadaRealWorldTester:
    """Comprehensive real-world testing for Janapada Memory Service"""

    def __init__(self):
        self.test_results = {}
        self.service_url = "http://localhost:8081"
        self.credentials_file = "konveyn2ai-b23e2fbffc62.json"

    def setup_credentials(self) -> bool:
        """Set up Google Cloud credentials from JSON file"""
        logger.info("🔐 Setting up Google Cloud credentials...")

        if not os.path.exists(self.credentials_file):
            logger.error(f"❌ Credentials file not found: {self.credentials_file}")
            return False

        try:
            # Set environment variables for Google Cloud authentication
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(
                self.credentials_file
            )

            # Load project info from credentials file (without displaying sensitive data)
            with open(self.credentials_file, "r") as f:
                creds = json.load(f)
                project_id = creds.get("project_id", "konveyn2ai")

            os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
            os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

            logger.info(f"✅ Credentials configured for project: {project_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to setup credentials: {e}")
            return False

    def test_google_cloud_auth(self) -> Dict[str, Any]:
        """Test Google Cloud authentication"""
        logger.info("🔐 Testing Google Cloud authentication...")

        try:
            import google.auth
            from google.cloud import aiplatform

            # Test default credentials
            credentials, project = google.auth.default()

            # Initialize Vertex AI
            aiplatform.init(project=project, location="us-central1")

            result = {
                "status": "success",
                "project": project,
                "auth_type": type(credentials).__name__,
                "has_service_account": hasattr(credentials, "service_account_email"),
            }

            if hasattr(credentials, "service_account_email"):
                result["service_account"] = credentials.service_account_email

            logger.info("✅ Google Cloud authentication successful")
            return result

        except Exception as e:
            logger.error(f"❌ Google Cloud authentication failed: {e}")
            return {"status": "failed", "error": str(e)}

    def test_vertex_ai_embeddings(self) -> Dict[str, Any]:
        """Test Vertex AI TextEmbeddingModel directly"""
        logger.info("🤖 Testing Vertex AI TextEmbeddingModel...")

        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel

            # Initialize Vertex AI
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "konveyn2ai")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

            vertexai.init(project=project_id, location=location)

            # Load the embedding model
            model = TextEmbeddingModel.from_pretrained("textembedding-gecko-001")

            # Test embedding generation
            test_text = "def authenticate_user(token: str) -> bool:"
            embeddings = model.get_embeddings([test_text])

            if embeddings and len(embeddings) > 0:
                embedding_vector = embeddings[0].values

                result = {
                    "status": "success",
                    "model_name": "textembedding-gecko-001",
                    "embedding_dimensions": len(embedding_vector),
                    "test_text": test_text,
                    "embedding_preview": embedding_vector[:5],  # First 5 values only
                }

                logger.info(
                    f"✅ Vertex AI embeddings working: {len(embedding_vector)} dimensions"
                )
                return result
            else:
                raise Exception("No embeddings returned from model")

        except Exception as e:
            logger.error(f"❌ Vertex AI embeddings test failed: {e}")
            return {"status": "failed", "error": str(e)}

    def start_janapada_service(self) -> bool:
        """Start the Janapada Memory Service"""
        logger.info("🚀 Starting Janapada Memory Service...")

        try:
            # Change to the service directory
            service_dir = "src/janapada-memory"

            # Start the service in background
            cmd = [sys.executable, "main.py"]
            self.service_process = subprocess.Popen(
                cmd,
                cwd=service_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy(),
            )

            # Wait a moment for service to start
            import time

            time.sleep(3)

            # Check if service is running
            if self.service_process.poll() is None:
                logger.info("✅ Janapada Memory Service started successfully")
                return True
            else:
                stdout, stderr = self.service_process.communicate()
                logger.error(f"❌ Service failed to start: {stderr.decode()}")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to start service: {e}")
            return False

    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test the health check endpoint"""
        logger.info("🏥 Testing health check endpoint...")

        try:
            response = requests.get(f"{self.service_url}/health", timeout=10)

            if response.status_code == 200:
                health_data = response.json()

                result = {
                    "status": "success",
                    "service_status": health_data.get("status"),
                    "components": health_data.get("components", {}),
                    "features": health_data.get("features", {}),
                }

                logger.info(f"✅ Health check passed: {health_data.get('status')}")
                return result
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {"status": "failed", "error": str(e)}

    def test_debug_embedding_endpoint(self) -> Dict[str, Any]:
        """Test the debug embedding endpoint"""
        logger.info("🔍 Testing debug embedding endpoint...")

        try:
            test_data = {"text": "def search_code(query: str) -> List[str]:"}

            response = requests.post(
                f"{self.service_url}/debug/embedding", json=test_data, timeout=30
            )

            if response.status_code == 200:
                debug_data = response.json()

                result = {
                    "status": "success",
                    "embedding_available": debug_data.get("embedding_available"),
                    "embedding_dimensions": debug_data.get("embedding_dimensions"),
                    "model_status": debug_data.get("model_status"),
                    "test_text": debug_data.get("text"),
                }

                logger.info(
                    f"✅ Debug embedding test passed: {debug_data.get('embedding_dimensions')} dims"
                )
                return result
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"❌ Debug embedding test failed: {e}")
            return {"status": "failed", "error": str(e)}

    def test_json_rpc_search(self) -> Dict[str, Any]:
        """Test the JSON-RPC search endpoint"""
        logger.info("🔍 Testing JSON-RPC search endpoint...")

        try:
            # JSON-RPC request payload
            rpc_request = {
                "jsonrpc": "2.0",
                "method": "search",
                "params": {"query": "authentication middleware function", "k": 3},
                "id": "test-search-001",
            }

            response = requests.post(
                self.service_url,
                json=rpc_request,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code == 200:
                rpc_response = response.json()

                if "result" in rpc_response:
                    snippets = rpc_response["result"].get("snippets", [])

                    result = {
                        "status": "success",
                        "snippets_count": len(snippets),
                        "snippets": snippets,
                        "request_id": rpc_response.get("id"),
                    }

                    logger.info(
                        f"✅ JSON-RPC search successful: {len(snippets)} snippets returned"
                    )
                    return result
                else:
                    raise Exception(
                        f"RPC Error: {rpc_response.get('error', 'Unknown error')}"
                    )
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"❌ JSON-RPC search test failed: {e}")
            return {"status": "failed", "error": str(e)}

    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up...")

        if hasattr(self, "service_process") and self.service_process:
            self.service_process.terminate()
            self.service_process.wait()
            logger.info("✅ Service process terminated")

    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests"""
        logger.info("🚀 Starting comprehensive real-world testing...")

        test_results = {"timestamp": datetime.now().isoformat(), "tests": {}}

        try:
            # Test 1: Setup credentials
            if not self.setup_credentials():
                test_results["tests"]["credentials_setup"] = {
                    "status": "failed",
                    "error": "Failed to setup credentials",
                }
                return test_results
            test_results["tests"]["credentials_setup"] = {"status": "success"}

            # Test 2: Google Cloud authentication
            test_results["tests"]["google_cloud_auth"] = self.test_google_cloud_auth()

            # Test 3: Vertex AI embeddings
            test_results["tests"][
                "vertex_ai_embeddings"
            ] = self.test_vertex_ai_embeddings()

            # Test 4: Start service
            if not self.start_janapada_service():
                test_results["tests"]["service_startup"] = {
                    "status": "failed",
                    "error": "Failed to start service",
                }
                return test_results
            test_results["tests"]["service_startup"] = {"status": "success"}

            # Test 5: Health endpoint
            test_results["tests"]["health_endpoint"] = self.test_health_endpoint()

            # Test 6: Debug embedding endpoint
            test_results["tests"][
                "debug_embedding"
            ] = self.test_debug_embedding_endpoint()

            # Test 7: JSON-RPC search
            test_results["tests"]["json_rpc_search"] = self.test_json_rpc_search()

        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            test_results["error"] = str(e)
        finally:
            self.cleanup()

        return test_results


def main():
    """Main test runner"""
    print("🚀 KonveyN2AI Janapada Memory Service - Real-World Testing")
    print("=" * 70)

    tester = JanapadaRealWorldTester()
    results = tester.run_comprehensive_tests()

    # Print summary
    print("\n📊 Test Results Summary:")
    print("=" * 50)

    total_tests = len(results.get("tests", {}))
    passed_tests = sum(
        1
        for test in results.get("tests", {}).values()
        if test.get("status") == "success"
    )

    for test_name, test_result in results.get("tests", {}).items():
        status_icon = "✅" if test_result.get("status") == "success" else "❌"
        print(f"{status_icon} {test_name}: {test_result.get('status', 'unknown')}")

        if test_result.get("status") == "failed":
            print(f"   Error: {test_result.get('error', 'Unknown error')}")

    print(f"\n📈 Overall: {passed_tests}/{total_tests} tests passed")

    # Save detailed results
    with open("test_results_janapada_real_world.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"📄 Detailed results saved to: test_results_janapada_real_world.json")

    return 0 if passed_tests == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
