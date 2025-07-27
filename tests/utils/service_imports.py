"""
Centralized service import utilities for KonveyN2AI tests.

This module provides a single source of truth for importing service modules
and eliminates the need for complex sys.path manipulation in individual tests.

Industry best practice: centralized import management with clear, reusable patterns.
"""

import os
import sys
import importlib
from typing import Any, Dict, Optional
from pathlib import Path


class ServiceImporter:
    """
    Manages service imports with proper path handling and module isolation.

    This replaces 50+ duplicated import patterns with a single, maintainable utility.
    """

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.src_path = self.project_root / "src"
        self._imported_modules: Dict[str, Any] = {}
        self._original_path = None

    def get_service_module(self, service_name: str) -> Any:
        """
        Import a service module with proper path handling.

        Args:
            service_name: One of 'svami', 'amatya', 'janapada'

        Returns:
            The imported service main module

        Example:
            >>> importer = ServiceImporter()
            >>> svami_main = importer.get_service_module('svami')
            >>> app = svami_main.app
        """
        if service_name in self._imported_modules:
            return self._imported_modules[service_name]

        service_paths = {
            "svami": "svami-orchestrator",
            "amatya": "amatya-role-prompter",
            "janapada": "janapada-memory",
        }

        if service_name not in service_paths:
            raise ValueError(
                f"Unknown service: {service_name}. Must be one of {list(service_paths.keys())}"
            )

        service_path = self.src_path / service_paths[service_name]

        if not service_path.exists():
            raise FileNotFoundError(f"Service path not found: {service_path}")

        # Add service path to Python path temporarily
        service_path_str = str(service_path)
        if service_path_str not in sys.path:
            sys.path.insert(0, service_path_str)

        try:
            # Import the main module
            import main

            # Reload to ensure fresh import
            importlib.reload(main)
            self._imported_modules[service_name] = main
            return main
        except ImportError as e:
            raise ImportError(f"Failed to import {service_name} service: {e}")

    def get_service_app(self, service_name: str) -> Any:
        """
        Get the FastAPI app instance for a service.

        Args:
            service_name: One of 'svami', 'amatya', 'janapada'

        Returns:
            The FastAPI app instance

        Example:
            >>> importer = ServiceImporter()
            >>> app = importer.get_service_app('svami')
            >>> client = TestClient(app)
        """
        main_module = self.get_service_module(service_name)
        if not hasattr(main_module, "app"):
            raise AttributeError(
                f"{service_name} service main module has no 'app' attribute"
            )
        return main_module.app

    def cleanup(self):
        """Clean up imported modules and restore sys.path."""
        # Remove service paths from sys.path
        for service_name in ["svami", "amatya", "janapada"]:
            service_paths = {
                "svami": "svami-orchestrator",
                "amatya": "amatya-role-prompter",
                "janapada": "janapada-memory",
            }
            service_path = str(self.src_path / service_paths[service_name])
            if service_path in sys.path:
                sys.path.remove(service_path)

        # Clear module cache for main modules
        modules_to_remove = [
            key
            for key in sys.modules.keys()
            if key == "main" or key.startswith("main.")
        ]
        for module_key in modules_to_remove:
            if module_key in sys.modules:
                del sys.modules[module_key]

        self._imported_modules.clear()


# Global instance for convenience
_service_importer = ServiceImporter()


def get_service_app(service_name: str) -> Any:
    """
    Convenience function to get a service app.

    This is the main function tests should use instead of complex import patterns.

    Args:
        service_name: One of 'svami', 'amatya', 'janapada'

    Returns:
        The FastAPI app instance

    Example:
        >>> from tests.utils.service_imports import get_service_app
        >>> app = get_service_app('svami')
        >>> client = TestClient(app)
    """
    return _service_importer.get_service_app(service_name)


def get_service_main(service_name: str) -> Any:
    """
    Convenience function to get a service main module.

    Args:
        service_name: One of 'svami', 'amatya', 'janapada'

    Returns:
        The service main module

    Example:
        >>> from tests.utils.service_imports import get_service_main
        >>> main = get_service_main('svami')
        >>> # Access global variables like main.janapada_client
    """
    return _service_importer.get_service_module(service_name)


def cleanup_service_imports():
    """
    Clean up all imported service modules.

    Call this in test teardown to prevent module contamination.

    Example:
        >>> from tests.utils.service_imports import cleanup_service_imports
        >>> cleanup_service_imports()  # In test teardown
    """
    _service_importer.cleanup()


# For backward compatibility and specific use cases
def import_common_models():
    """Import common models and utilities."""
    src_path = str(Path(__file__).parent.parent.parent / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    try:
        from common.models import (
            JsonRpcResponse,
            QueryRequest,
            AdviceRequest,
            SearchRequest,
        )
        from common.rpc_client import JsonRpcClient

        return {
            "JsonRpcResponse": JsonRpcResponse,
            "QueryRequest": QueryRequest,
            "AdviceRequest": AdviceRequest,
            "SearchRequest": SearchRequest,
            "JsonRpcClient": JsonRpcClient,
        }
    except ImportError as e:
        raise ImportError(f"Failed to import common models: {e}")


# Convenience constants
SUPPORTED_SERVICES = ["svami", "amatya", "janapada"]
SERVICE_NAMES = {
    "svami": "Svami Orchestrator",
    "amatya": "Amatya Role Prompter",
    "janapada": "Janapada Memory",
}
