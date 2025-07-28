"""
Test utility functions and helpers for KonveyN2AI testing.
"""

import asyncio
import random
import string
import time
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

import httpx


class JSONRPCTestClient:
    """Test client for JSON-RPC services."""

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url
        self.timeout = timeout

    async def call(
        self, method: str, params: dict[str, Any] = None, request_id: str = None
    ) -> dict[str, Any]:
        """Make a JSON-RPC call."""

        request_id = request_id or self.generate_request_id()

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.base_url, json=payload)
            return response.json()

    @staticmethod
    def generate_request_id() -> str:
        """Generate a unique request ID."""
        return f"test-{int(time.time())}-{''.join(random.choices(string.ascii_lowercase, k=6))}"

    def is_success_response(self, response: dict[str, Any]) -> bool:
        """Check if response indicates success."""
        return "result" in response and "error" not in response

    def is_error_response(self, response: dict[str, Any]) -> bool:
        """Check if response indicates error."""
        return "error" in response and "result" not in response


class MockDataGenerator:
    """Generate mock data for testing."""

    @staticmethod
    def generate_code_snippet(
        file_path: str = None, language: str = "python"
    ) -> dict[str, str]:
        """Generate a mock code snippet."""

        file_path = file_path or f"src/test_{random.randint(1, 1000)}.{language}"

        code_templates = {
            "python": [
                'def {func_name}():\n    """Test function."""\n    return True',
                "class {class_name}:\n    def __init__(self):\n        self.value = 42",
                "import {module}\n\ndef process_data(data):\n    return {module}.process(data)",
                "async def {func_name}(request):\n    result = await process_request(request)\n    return result",
            ],
            "javascript": [
                "function {func_name}() {{\n  return 'Hello World';\n}}",
                "const {const_name} = (data) => {{\n  return data.map(item => item.id);\n}};",
                "class {class_name} {{\n  constructor() {{\n    this.value = 42;\n  }}\n}}",
            ],
            "typescript": [
                "interface {interface_name} {{\n  id: number;\n  name: string;\n}}",
                "function {func_name}(data: {interface_name}[]): number[] {{\n  return data.map(item => item.id);\n}}",
            ],
        }

        template = random.choice(code_templates.get(language, code_templates["python"]))

        # Fill in template variables
        content = template.format(
            func_name=f"test_function_{random.randint(1, 100)}",
            class_name=f"TestClass{random.randint(1, 100)}",
            module=random.choice(["json", "os", "sys", "datetime"]),
            const_name=f"testConst{random.randint(1, 100)}",
            interface_name=f"TestInterface{random.randint(1, 100)}",
        )

        return {"file_path": file_path, "content": content}

    @staticmethod
    def generate_multiple_snippets(
        count: int = 5, languages: list[str] = None
    ) -> list[dict[str, str]]:
        """Generate multiple code snippets."""

        languages = languages or ["python", "javascript", "typescript"]
        snippets = []

        for _i in range(count):
            language = random.choice(languages)
            snippet = MockDataGenerator.generate_code_snippet(language=language)
            snippets.append(snippet)

        return snippets

    @staticmethod
    def generate_query_request(role: str = None) -> dict[str, str]:
        """Generate a mock query request."""

        roles = [
            "backend engineer",
            "frontend developer",
            "devops engineer",
            "data scientist",
            "security analyst",
            "product manager",
            "qa engineer",
            "architect",
        ]

        questions = [
            "How do I implement authentication?",
            "What's the best way to handle database connections?",
            "How can I optimize API performance?",
            "What are the security best practices?",
            "How do I set up CI/CD pipeline?",
            "What's the recommended testing strategy?",
            "How can I implement caching?",
            "What's the best way to handle errors?",
        ]

        return {
            "question": random.choice(questions),
            "role": role or random.choice(roles),
        }

    @staticmethod
    def generate_large_content(size_mb: float = 1.0) -> str:
        """Generate large content for testing."""

        # Calculate target size in characters (approximate)
        target_chars = int(size_mb * 1024 * 1024)

        # Base content to repeat
        base_content = (
            "def test_function():\n    # This is a test function\n    return True\n\n"
        )

        # Calculate how many repetitions needed
        repetitions = target_chars // len(base_content)

        return base_content * repetitions


class TestResponseValidator:
    """Validate test responses against expected schemas."""

    @staticmethod
    def validate_jsonrpc_response(
        response: dict[str, Any], expected_id: str = None
    ) -> bool:
        """Validate JSON-RPC response format."""

        # Check required fields
        if "jsonrpc" not in response:
            return False

        if response["jsonrpc"] != "2.0":
            return False

        if "id" not in response:
            return False

        if expected_id and response["id"] != expected_id:
            return False

        # Must have either result or error, but not both
        has_result = "result" in response
        has_error = "error" in response

        if has_result and has_error:
            return False

        if not has_result and not has_error:
            return False

        return True

    @staticmethod
    def validate_search_response(response: dict[str, Any]) -> bool:
        """Validate search response format."""

        if not TestResponseValidator.validate_jsonrpc_response(response):
            return False

        if "result" not in response:
            return False

        result = response["result"]

        if "snippets" not in result:
            return False

        if not isinstance(result["snippets"], list):
            return False

        # Validate snippet format
        for snippet in result["snippets"]:
            if not isinstance(snippet, dict):
                return False

            if "file_path" not in snippet or "content" not in snippet:
                return False

            if not isinstance(snippet["file_path"], str) or not isinstance(
                snippet["content"], str
            ):
                return False

        return True

    @staticmethod
    def validate_advise_response(response: dict[str, Any]) -> bool:
        """Validate advise response format."""

        if not TestResponseValidator.validate_jsonrpc_response(response):
            return False

        if "result" not in response:
            return False

        result = response["result"]

        if "advice" not in result:
            return False

        if not isinstance(result["advice"], str):
            return False

        if len(result["advice"]) == 0:
            return False

        return True

    @staticmethod
    def validate_health_response(response: dict[str, Any]) -> bool:
        """Validate health check response format."""

        required_fields = ["status", "timestamp"]

        for field in required_fields:
            if field not in response:
                return False

        valid_statuses = ["healthy", "degraded", "unhealthy"]
        if response["status"] not in valid_statuses:
            return False

        # Check dependencies field if present
        if "dependencies" in response:
            if not isinstance(response["dependencies"], list):
                return False

            for dep in response["dependencies"]:
                if not isinstance(dep, dict):
                    return False

                if "name" not in dep or "status" not in dep:
                    return False

                if dep["status"] not in valid_statuses:
                    return False

        return True


class PerformanceTestHelper:
    """Helper for performance testing."""

    @staticmethod
    async def measure_response_time(async_func: Callable, *args, **kwargs) -> float:
        """Measure response time of an async function."""

        start_time = time.time()
        await async_func(*args, **kwargs)
        end_time = time.time()

        return end_time - start_time

    @staticmethod
    async def run_concurrent_requests(
        async_func: Callable, request_count: int = 10, *args, **kwargs
    ) -> list[Any]:
        """Run multiple concurrent requests."""

        tasks = [async_func(*args, **kwargs) for _ in range(request_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return results

    @staticmethod
    def calculate_percentiles(
        values: list[float], percentiles: list[int] = None
    ) -> dict[int, float]:
        """Calculate percentiles from a list of values."""

        percentiles = percentiles or [50, 90, 95, 99]
        sorted_values = sorted(values)
        result = {}

        for p in percentiles:
            index = int((p / 100.0) * len(sorted_values))
            if index >= len(sorted_values):
                index = len(sorted_values) - 1
            result[p] = sorted_values[index]

        return result

    @staticmethod
    def analyze_performance_results(response_times: list[float]) -> dict[str, Any]:
        """Analyze performance test results."""

        if not response_times:
            return {}

        return {
            "count": len(response_times),
            "avg": sum(response_times) / len(response_times),
            "min": min(response_times),
            "max": max(response_times),
            "percentiles": PerformanceTestHelper.calculate_percentiles(response_times),
        }


class ErrorTestHelper:
    """Helper for testing error conditions."""

    @staticmethod
    def create_connection_error():
        """Create a connection error for testing."""
        return httpx.ConnectError("Connection failed")

    @staticmethod
    def create_timeout_error():
        """Create a timeout error for testing."""
        return httpx.ReadTimeout("Request timeout")

    @staticmethod
    def create_http_error(status_code: int = 500):
        """Create an HTTP error for testing."""
        response = MagicMock()
        response.status_code = status_code
        return httpx.HTTPStatusError(
            message=f"HTTP {status_code}", request=MagicMock(), response=response
        )

    @staticmethod
    def create_jsonrpc_error(
        code: int = -32603, message: str = "Internal error"
    ) -> dict[str, Any]:
        """Create a JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": "test",
            "error": {"code": code, "message": message},
        }


class MockServiceFactory:
    """Factory for creating mock services."""

    @staticmethod
    def create_mock_janapada(snippets: list[dict[str, str]] = None):
        """Create a mock Janapada service."""

        mock = AsyncMock()

        async def mock_search(method: str, params: dict[str, Any], **kwargs):
            if method == "search":
                return {
                    "snippets": snippets
                    or MockDataGenerator.generate_multiple_snippets(3)
                }
            else:
                raise ValueError(f"Unknown method: {method}")

        mock.call.side_effect = mock_search
        return mock

    @staticmethod
    def create_mock_amatya(advice: str = None):
        """Create a mock Amatya service."""

        mock = AsyncMock()

        async def mock_advise(method: str, params: dict[str, Any], **kwargs):
            if method == "advise":
                role = params.get("role", "developer")
                default_advice = (
                    f"As a {role}, here's my advice based on the provided code..."
                )
                return {"advice": advice or default_advice}
            else:
                raise ValueError(f"Unknown method: {method}")

        mock.call.side_effect = mock_advise
        return mock

    @staticmethod
    def create_failing_service(error: Exception = None):
        """Create a service that always fails."""

        mock = AsyncMock()
        mock.call.side_effect = error or Exception("Service failure")
        return mock


class TestDataLoader:
    """Load test data from files."""

    @staticmethod
    def load_sample_code_files(directory: str) -> list[dict[str, str]]:
        """Load sample code files from a directory."""

        import glob
        import os

        snippets = []

        if not os.path.exists(directory):
            return snippets

        # Find all code files
        patterns = ["*.py", "*.js", "*.ts", "*.java", "*.cpp", "*.go"]

        for pattern in patterns:
            files = glob.glob(os.path.join(directory, "**", pattern), recursive=True)

            for file_path in files:
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()

                    # Make path relative to directory
                    relative_path = os.path.relpath(file_path, directory)

                    snippets.append({"file_path": relative_path, "content": content})
                except Exception:
                    # Skip files that can't be read
                    continue

        return snippets


class AssertionHelpers:
    """Custom assertion helpers for testing."""

    @staticmethod
    def assert_valid_uuid(value: str):
        """Assert that a value is a valid UUID."""
        import uuid

        try:
            uuid.UUID(value)
        except ValueError:
            raise AssertionError(f"'{value}' is not a valid UUID")

    @staticmethod
    def assert_response_time(actual_time: float, max_time: float):
        """Assert that response time is within acceptable limits."""
        if actual_time > max_time:
            raise AssertionError(
                f"Response time {actual_time:.3f}s exceeds maximum {max_time:.3f}s"
            )

    @staticmethod
    def assert_contains_all(container: list[Any], items: list[Any]):
        """Assert that container contains all specified items."""
        missing = [item for item in items if item not in container]
        if missing:
            raise AssertionError(f"Container missing items: {missing}")

    @staticmethod
    def assert_no_duplicates(items: list[Any]):
        """Assert that a list contains no duplicates."""
        unique_items = set(items)
        if len(unique_items) != len(items):
            raise AssertionError(
                f"List contains duplicates: {len(items)} items, {len(unique_items)} unique"
            )

    @staticmethod
    def assert_json_structure(
        data: dict[str, Any], expected_structure: dict[str, type]
    ):
        """Assert that JSON data matches expected structure."""
        for key, expected_type in expected_structure.items():
            if key not in data:
                raise AssertionError(f"Missing key: {key}")

            if not isinstance(data[key], expected_type):
                raise AssertionError(
                    f"Key '{key}' has type {type(data[key])}, expected {expected_type}"
                )


# Export commonly used classes and functions
__all__ = [
    "JSONRPCTestClient",
    "MockDataGenerator",
    "TestResponseValidator",
    "PerformanceTestHelper",
    "ErrorTestHelper",
    "MockServiceFactory",
    "TestDataLoader",
    "AssertionHelpers",
]
