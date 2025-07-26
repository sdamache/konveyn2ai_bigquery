"""
Tests for the enhanced agent manifest system.

Tests cover:
1. AgentManifestGenerator functionality
2. Method schema extraction from type hints and docstrings
3. Agent discovery and remote method calling
4. Integration with JsonRpcServer
"""

import json
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest
import httpx

from src.common.agent_manifest import (
    AgentCapability,
    AgentDiscovery,
    AgentManifest,
    AgentManifestGenerator,
    MethodSchema,
    ParameterSchema,
)
from src.common.rpc_server import JsonRpcServer


class TestParameterSchema:
    """Test ParameterSchema model."""
    
    def test_parameter_schema_creation(self):
        """Test creating a parameter schema."""
        param = ParameterSchema(
            name="test_param",
            type="str",
            required=True,
            description="A test parameter"
        )
        
        assert param.name == "test_param"
        assert param.type == "str"
        assert param.required is True
        assert param.description == "A test parameter"
        assert param.default is None


class TestMethodSchema:
    """Test MethodSchema model."""
    
    def test_method_schema_creation(self):
        """Test creating a method schema."""
        params = [
            ParameterSchema(name="a", type="int", required=True),
            ParameterSchema(name="b", type="int", required=False, default=0)
        ]
        
        method = MethodSchema(
            name="add",
            description="Add two numbers",
            parameters=params,
            return_type="int",
            handler="add_handler"
        )
        
        assert method.name == "add"
        assert method.description == "Add two numbers"
        assert len(method.parameters) == 2
        assert method.return_type == "int"
        assert method.handler == "add_handler"


class TestAgentManifestGenerator:
    """Test AgentManifestGenerator functionality."""
    
    @pytest.fixture
    def generator(self):
        """Create a test manifest generator."""
        return AgentManifestGenerator(
            name="Test Agent",
            version="1.0.0",
            description="A test agent"
        )
    
    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.name == "Test Agent"
        assert generator.version == "1.0.0"
        assert generator.description == "A test agent"
        assert len(generator.capabilities) == 5  # Default capabilities
        
    def test_register_method(self, generator):
        """Test method registration."""
        def test_method(a: int, b: str = "default") -> dict:
            """A test method."""
            return {"a": a, "b": b}
        
        generator.register_method("test", test_method, "Test method")
        
        assert "test" in generator.methods
        assert generator.methods["test"] == test_method
        assert generator.method_descriptions["test"] == "Test method"
    
    def test_add_capability(self, generator):
        """Test adding custom capabilities."""
        capability = AgentCapability(
            name="custom-feature",
            version="1.0",
            description="A custom feature"
        )
        
        initial_count = len(generator.capabilities)
        generator.add_capability(capability)
        
        assert len(generator.capabilities) == initial_count + 1
        assert capability in generator.capabilities
    
    def test_parameter_schema_extraction(self, generator):
        """Test parameter schema extraction from function signatures."""
        def complex_method(
            required_param: str,
            optional_param: int = 42,
            list_param: List[str] = None,
            dict_param: Optional[Dict[str, int]] = None
        ) -> Dict[str, any]:
            """
            A complex method for testing.
            
            Args:
                required_param: A required string parameter
                optional_param: An optional integer parameter
                list_param: A list of strings
                dict_param: An optional dictionary parameter
            """
            return {}
        
        generator.register_method("complex", complex_method)
        manifest = generator.generate_manifest()
        
        method_schema = manifest.methods["complex"]
        params = {p.name: p for p in method_schema.parameters}
        
        # Check required parameter
        assert "required_param" in params
        assert params["required_param"].required is True
        assert params["required_param"].type == "str"
        
        # Check optional parameter with default
        assert "optional_param" in params
        assert params["optional_param"].required is False
        assert params["optional_param"].default == 42
        assert params["optional_param"].type == "int"
        
        # Check list parameter
        assert "list_param" in params
        assert params["list_param"].required is False
        
        # Check return type
        assert method_schema.return_type != "Any"
    
    def test_docstring_parsing(self, generator):
        """Test docstring parameter parsing."""
        def documented_method(param1: str, param2: int) -> str:
            """
            A well-documented method.
            
            Args:
                param1: The first parameter description
                param2: The second parameter description
            """
            return "result"
        
        generator.register_method("documented", documented_method)
        manifest = generator.generate_manifest()
        
        method_schema = manifest.methods["documented"]
        params = {p.name: p for p in method_schema.parameters}
        
        assert params["param1"].description == "The first parameter description"
        assert params["param2"].description == "The second parameter description"
    
    def test_manifest_generation(self, generator):
        """Test complete manifest generation."""
        def test_method(a: int) -> dict:
            """Test method."""
            return {"result": a}
        
        generator.register_method("test", test_method, "Test method")
        generator.add_endpoint("health", "/health")
        generator.set_metadata("author", "Test Author")
        
        manifest = generator.generate_manifest()
        
        assert isinstance(manifest, AgentManifest)
        assert manifest.name == "Test Agent"
        assert manifest.version == "1.0.0"
        assert manifest.description == "A test agent"
        assert "test" in manifest.methods
        assert "health" in manifest.endpoints
        assert manifest.metadata["author"] == "Test Author"
        assert manifest.generated_at is not None
    
    def test_to_dict_and_json(self, generator):
        """Test manifest serialization."""
        def simple_method() -> str:
            return "hello"
        
        generator.register_method("simple", simple_method)
        
        # Test to_dict
        manifest_dict = generator.to_dict()
        assert isinstance(manifest_dict, dict)
        assert manifest_dict["name"] == "Test Agent"
        assert "simple" in manifest_dict["methods"]
        
        # Test to_json
        manifest_json = generator.to_json()
        assert isinstance(manifest_json, str)
        parsed = json.loads(manifest_json)
        assert parsed["name"] == "Test Agent"


class TestAgentDiscovery:
    """Test AgentDiscovery functionality."""
    
    @pytest.fixture
    def discovery(self):
        """Create a test agent discovery instance."""
        return AgentDiscovery(timeout=10)
    
    @pytest.mark.asyncio
    async def test_discover_agent_success(self, discovery):
        """Test successful agent discovery."""
        mock_manifest = {
            "name": "Remote Agent",
            "version": "1.0.0",
            "protocol": "json-rpc-2.0",
            "methods": {},
            "capabilities": [],
            "endpoints": {},
            "metadata": {},
            "generated_at": "2024-01-01T00:00:00"
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = mock_manifest
            mock_response.raise_for_status.return_value = None
            
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            
            manifest = await discovery.discover_agent("http://example.com")
            
            assert manifest is not None
            assert manifest.name == "Remote Agent"
            assert manifest.version == "1.0.0"
            assert "http://example.com" in discovery._manifest_cache
    
    @pytest.mark.asyncio
    async def test_discover_agent_failure(self, discovery):
        """Test agent discovery failure."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )
            
            manifest = await discovery.discover_agent("http://invalid.com")
            
            assert manifest is None
            assert "http://invalid.com" not in discovery._manifest_cache
    
    @pytest.mark.asyncio
    async def test_call_agent_method(self, discovery):
        """Test calling a method on a remote agent."""
        # First, cache a manifest
        mock_manifest = AgentManifest(
            name="Remote Agent",
            version="1.0.0",
            methods={"test_method": MethodSchema(
                name="test_method",
                description="Test method",
                parameters=[],
                return_type="dict",
                handler="test_handler"
            )}
        )
        discovery._manifest_cache["http://example.com"] = mock_manifest
        
        # Mock the RPC client call
        with patch("src.common.agent_manifest.JsonRpcClient") as mock_client_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.result = {"success": True}
            mock_client.call = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            response = await discovery.call_agent_method(
                "http://example.com",
                "test_method",
                {"param": "value"},
                "test-id"
            )
            
            assert response.result == {"success": True}
            mock_client.call.assert_called_once_with(
                "test_method",
                {"param": "value"},
                "test-id"
            )
    
    def test_get_cached_manifest(self, discovery):
        """Test getting cached manifest."""
        mock_manifest = AgentManifest(
            name="Cached Agent",
            version="1.0.0"
        )
        discovery._manifest_cache["http://cached.com"] = mock_manifest
        
        cached = discovery.get_cached_manifest("http://cached.com")
        assert cached == mock_manifest
        
        not_cached = discovery.get_cached_manifest("http://not-cached.com")
        assert not_cached is None
    
    def test_list_agent_methods(self, discovery):
        """Test listing methods for a cached agent."""
        mock_manifest = AgentManifest(
            name="Method Agent",
            version="1.0.0",
            methods={
                "method1": MethodSchema(name="method1", description="", parameters=[], return_type="", handler=""),
                "method2": MethodSchema(name="method2", description="", parameters=[], return_type="", handler="")
            }
        )
        discovery._manifest_cache["http://methods.com"] = mock_manifest
        
        methods = discovery.list_agent_methods("http://methods.com")
        assert set(methods) == {"method1", "method2"}
        
        no_methods = discovery.list_agent_methods("http://no-agent.com")
        assert no_methods == []


class TestJsonRpcServerIntegration:
    """Test integration with JsonRpcServer."""
    
    @pytest.fixture
    def server(self):
        """Create a test server with enhanced manifest."""
        return JsonRpcServer(
            title="Integration Test Server",
            version="2.0.0",
            description="Server for testing manifest integration"
        )
    
    def test_enhanced_manifest_generation(self, server):
        """Test that server uses enhanced manifest generator."""
        @server.method("test_method", "A test method")
        def test_handler(param: str) -> dict:
            """Test handler with parameter."""
            return {"param": param}
        
        manifest = server.get_manifest()
        
        # Check enhanced manifest structure
        assert "generated_at" in manifest
        assert "description" in manifest
        assert manifest["description"] == "Server for testing manifest integration"
        
        # Check method details
        assert "test_method" in manifest["methods"]
        method = manifest["methods"]["test_method"]
        assert "parameters" in method
        assert len(method["parameters"]) == 1
        assert method["parameters"][0]["name"] == "param"
    
    def test_custom_capabilities(self, server):
        """Test adding custom capabilities."""
        server.add_capability("custom-feature", "1.0", "Custom feature")
        
        manifest = server.get_manifest()
        capability_names = [cap["name"] for cap in manifest["capabilities"]]
        assert "custom-feature" in capability_names
    
    def test_endpoints_and_metadata(self, server):
        """Test adding endpoints and metadata."""
        server.add_endpoint("health", "/health")
        server.set_metadata("author", "Test Team")
        
        manifest = server.get_manifest()
        assert manifest["endpoints"]["health"] == "/health"
        assert manifest["metadata"]["author"] == "Test Team"
