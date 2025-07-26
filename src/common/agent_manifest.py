"""
Agent Manifest Generator for KonveyN2AI multi-agent system.

This module provides enhanced agent manifest generation capabilities including:
- Detailed method documentation from type hints and docstrings
- Versioning and capability advertisement
- Agent discovery and method calling utilities
- Enhanced manifest format for better interoperability
"""

import inspect
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Type, get_type_hints
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, Field

from .models import JsonRpcRequest, JsonRpcResponse
from .rpc_client import JsonRpcClient

logger = logging.getLogger(__name__)


class ParameterSchema(BaseModel):
    """Schema for a method parameter."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type")
    required: bool = Field(..., description="Whether parameter is required")
    default: Optional[Any] = Field(None, description="Default value if any")
    description: Optional[str] = Field(None, description="Parameter description")


class MethodSchema(BaseModel):
    """Schema for a JSON-RPC method."""

    name: str = Field(..., description="Method name")
    description: str = Field(..., description="Method description")
    parameters: List[ParameterSchema] = Field(
        default_factory=list, description="Method parameters"
    )
    return_type: str = Field(..., description="Return type")
    handler: str = Field(..., description="Handler function name")
    examples: Optional[List[Dict[str, Any]]] = Field(None, description="Usage examples")


class AgentCapability(BaseModel):
    """Agent capability description."""

    name: str = Field(..., description="Capability name")
    version: str = Field(..., description="Capability version")
    description: str = Field(..., description="Capability description")


class AgentManifest(BaseModel):
    """Complete agent manifest."""

    name: str = Field(..., description="Agent name")
    version: str = Field(..., description="Agent version")
    protocol: str = Field(default="json-rpc-2.0", description="Communication protocol")
    description: Optional[str] = Field(None, description="Agent description")
    methods: Dict[str, MethodSchema] = Field(
        default_factory=dict, description="Available methods"
    )
    capabilities: List[AgentCapability] = Field(
        default_factory=list, description="Agent capabilities"
    )
    endpoints: Dict[str, str] = Field(
        default_factory=dict, description="Service endpoints"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Generation timestamp",
    )


class AgentManifestGenerator:
    """Enhanced agent manifest generator."""

    def __init__(self, name: str, version: str, description: Optional[str] = None):
        self.name = name
        self.version = version
        self.description = description
        self.methods: Dict[str, Callable[..., Any]] = {}
        self.method_descriptions: Dict[str, str] = {}
        self.capabilities: List[AgentCapability] = []
        self.endpoints: Dict[str, str] = {}
        self.metadata: Dict[str, Any] = {}

        # Add default capabilities
        self._add_default_capabilities()

    def _add_default_capabilities(self) -> None:
        """Add default JSON-RPC capabilities."""
        default_caps = [
            AgentCapability(
                name="json-rpc-2.0",
                version="2.0",
                description="JSON-RPC 2.0 protocol support",
            ),
            AgentCapability(
                name="single-requests",
                version="1.0",
                description="Single request processing",
            ),
            AgentCapability(
                name="batch-requests",
                version="1.0",
                description="Batch request processing",
            ),
            AgentCapability(
                name="notifications",
                version="1.0",
                description="Notification handling (no response)",
            ),
            AgentCapability(
                name="error-handling",
                version="1.0",
                description="Structured error responses",
            ),
        ]
        self.capabilities.extend(default_caps)

    def register_method(
        self, name: str, handler: Callable[..., Any], description: str = ""
    ) -> None:
        """Register a method with the manifest generator."""
        self.methods[name] = handler
        self.method_descriptions[name] = description
        logger.info(f"Registered method '{name}' in manifest generator")

    def add_capability(self, capability: AgentCapability) -> None:
        """Add a custom capability."""
        self.capabilities.append(capability)

    def add_endpoint(self, name: str, url: str) -> None:
        """Add a service endpoint."""
        self.endpoints[name] = url

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata field."""
        self.metadata[key] = value

    def _extract_parameter_schema(
        self, func: Callable[..., Any]
    ) -> List[ParameterSchema]:
        """Extract parameter schema from function signature and type hints."""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        parameters = []

        # Parse docstring for parameter descriptions
        param_docs = self._parse_docstring_params(func)

        for param_name, param in sig.parameters.items():
            # Skip special parameters
            if param_name in ("self", "request_id", "context"):
                continue

            # Get type information
            param_type = type_hints.get(param_name, param.annotation)
            type_str = self._format_type(param_type)

            # Determine if required
            required = param.default == inspect.Parameter.empty
            default_value = None if required else param.default

            # Get description from docstring
            description = param_docs.get(param_name)

            parameters.append(
                ParameterSchema(
                    name=param_name,
                    type=type_str,
                    required=required,
                    default=default_value,
                    description=description,
                )
            )

        return parameters

    def _parse_docstring_params(self, func: Callable[..., Any]) -> Dict[str, str]:
        """Parse parameter descriptions from docstring."""
        if not func.__doc__:
            return {}

        param_docs = {}
        lines = func.__doc__.split("\n")
        current_param = None

        for line in lines:
            line = line.strip()
            if line.startswith("Args:") or line.startswith("Parameters:"):
                continue
            elif ":" in line and not line.startswith(" "):
                # New parameter
                parts = line.split(":", 1)
                if len(parts) == 2:
                    current_param = parts[0].strip()
                    param_docs[current_param] = parts[1].strip()
            elif current_param and line:
                # Continuation of parameter description
                param_docs[current_param] += " " + line

        return param_docs

    def _format_type(self, type_annotation: Any) -> str:
        """Format type annotation as string."""
        if type_annotation == inspect.Parameter.empty:
            return "Any"

        if hasattr(type_annotation, "__name__"):
            return type_annotation.__name__

        return str(type_annotation)

    def _get_return_type(self, func: Callable[..., Any]) -> str:
        """Get return type from function signature."""
        sig = inspect.signature(func)
        return_annotation = sig.return_annotation

        if return_annotation == inspect.Signature.empty:
            return "Any"

        return self._format_type(return_annotation)

    def generate_manifest(self) -> AgentManifest:
        """Generate the complete agent manifest."""
        method_schemas = {}

        for method_name, handler in self.methods.items():
            description = self.method_descriptions.get(method_name, "")
            if not description and handler.__doc__:
                description = handler.__doc__.split("\n")[0].strip()

            method_schema = MethodSchema(
                name=method_name,
                description=description,
                parameters=self._extract_parameter_schema(handler),
                return_type=self._get_return_type(handler),
                handler=handler.__name__,
            )
            method_schemas[method_name] = method_schema

        return AgentManifest(
            name=self.name,
            version=self.version,
            description=self.description,
            methods=method_schemas,
            capabilities=self.capabilities,
            endpoints=self.endpoints,
            metadata=self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        manifest = self.generate_manifest()
        return manifest.model_dump()

    def to_json(self) -> str:
        """Convert manifest to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class AgentDiscovery:
    """Utilities for discovering and calling methods on other agents."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._manifest_cache: Dict[str, AgentManifest] = {}

    async def discover_agent(self, base_url: str) -> Optional[AgentManifest]:
        """Discover an agent by fetching its manifest."""
        manifest_url = urljoin(base_url.rstrip("/") + "/", ".well-known/agent.json")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(manifest_url)
                response.raise_for_status()

                manifest_data = response.json()
                manifest = AgentManifest(**manifest_data)

                # Cache the manifest
                self._manifest_cache[base_url] = manifest
                logger.info(f"Discovered agent '{manifest.name}' at {base_url}")

                return manifest

        except Exception as e:
            logger.error(f"Failed to discover agent at {base_url}: {e}")
            return None

    async def call_agent_method(
        self,
        base_url: str,
        method: str,
        params: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> JsonRpcResponse:
        """Call a method on a remote agent."""
        # Ensure we have the agent's manifest
        if base_url not in self._manifest_cache:
            await self.discover_agent(base_url)

        # Construct RPC endpoint URL
        rpc_url = urljoin(base_url.rstrip("/") + "/", "rpc")

        # Create RPC client and make the call
        client = JsonRpcClient(rpc_url, timeout=self.timeout)
        return await client.call(method, params, request_id)

    def get_cached_manifest(self, base_url: str) -> Optional[AgentManifest]:
        """Get cached manifest for an agent."""
        return self._manifest_cache.get(base_url)

    def list_agent_methods(self, base_url: str) -> List[str]:
        """List available methods for a cached agent."""
        manifest = self.get_cached_manifest(base_url)
        return list(manifest.methods.keys()) if manifest else []
