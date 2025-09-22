"""
Centralized service import utilities for KonveyN2AI tests.

This module provides a single source of truth for importing service modules
and eliminates the need for complex sys.path manipulation in individual tests.

Industry best practice: centralized import management with clear, reusable patterns.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any


class ServiceImporter:
    """
    Manages service imports with proper path handling and module isolation.

    This replaces 50+ duplicated import patterns with a single, maintainable utility.
    """

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.src_path = self.project_root / "src"
        self._imported_modules: dict[str, Any] = {}
        self._original_path = None

    def get_service_module(self, service_name: str) -> Any:
        """
        Import a service module with proper isolation using importlib.

        Uses industry-standard importlib.util.spec_from_file_location() to avoid
        sys.modules conflicts when importing multiple main.py files from different
        directories. Each service gets a unique module name for clean isolation.

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
            "janapada": "janapada_memory",
        }

        if service_name not in service_paths:
            raise ValueError(
                f"Unknown service: {service_name}. Must be one of {list(service_paths.keys())}"
            )

        service_path = self.src_path / service_paths[service_name]
        main_py_path = service_path / "main.py"

        if not service_path.exists():
            raise FileNotFoundError(f"Service path not found: {service_path}")

        if not main_py_path.exists():
            raise FileNotFoundError(f"main.py not found: {main_py_path}")

        try:
            # Use unique module name to prevent sys.modules conflicts
            # This is the industry-standard approach for isolating same-named modules
            unique_module_name = f"{service_name}_main_module"

            # Create module spec from file location
            spec = importlib.util.spec_from_file_location(
                unique_module_name, str(main_py_path)
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create module spec for {main_py_path}")

            # Ensure both src path and service path are available during module execution
            src_path_str = str(self.src_path)
            service_path_str = str(service_path)
            paths_added = []

            # Add src path for common modules
            if src_path_str not in sys.path:
                sys.path.insert(0, src_path_str)
                paths_added.append(src_path_str)

            # Add service path for local service modules
            if service_path_str not in sys.path:
                sys.path.insert(0, service_path_str)
                paths_added.append(service_path_str)

            try:
                # Create and execute module - this avoids sys.modules conflicts
                main_module = importlib.util.module_from_spec(spec)

                # IMPORTANT: Add to sys.modules before execution for patch target resolution
                sys.modules[unique_module_name] = main_module

                spec.loader.exec_module(main_module)

                # Cache for reuse
                self._imported_modules[service_name] = main_module
                return main_module
            finally:
                # Clean up sys.path - remove in reverse order
                for path_str in reversed(paths_added):
                    if path_str in sys.path:
                        sys.path.remove(path_str)

        except ImportError as e:
            raise ImportError(f"Failed to import {service_name} service: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading {service_name} service: {e}")

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
        """
        Clean up imported service modules.

        With the new importlib approach, cleanup is much safer since we use
        unique module names and don't modify sys.path.
        """
        # Simply clear our internal cache - no risky sys.modules manipulation needed
        # The unique module names prevent conflicts so cleanup is optional
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


def get_service_patch_target(service_name: str, class_name: str) -> str:
    """
    Get the correct patch target path for a service class with unique module names.

    Since we use unique module names like 'amatya_main_module', tests need the
    correct patch target path for mocking.

    Args:
        service_name: One of 'svami', 'amatya', 'janapada'
        class_name: Name of the class to patch (e.g., 'AdvisorService')

    Returns:
        The correct patch target string for unittest.mock.patch

    Example:
        >>> from tests.utils.service_imports import get_service_patch_target
        >>> target = get_service_patch_target('amatya', 'AdvisorService')
        >>> with patch(target) as mock_service:
        ...     # Now patch will work correctly
    """
    # First ensure the module is loaded to get the correct sys.modules name
    _service_importer.get_service_module(service_name)

    # Return the unique module name that's actually in sys.modules
    unique_module_name = f"{service_name}_main_module"
    return f"{unique_module_name}.{class_name}"


# For backward compatibility and specific use cases
def import_common_models():
    """Import common models and utilities."""
    src_path = str(Path(__file__).parent.parent.parent / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    try:
        from common.models import (
            AdviceRequest,
            JsonRpcResponse,
            QueryRequest,
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
