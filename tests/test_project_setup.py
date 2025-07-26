"""
Test project setup and basic functionality.

This module tests that the project structure is correct and basic
imports work as expected.
"""

import sys
import os
from pathlib import Path

# Add src to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_project_structure():
    """Test that the project directory structure is correct."""
    expected_dirs = [
        "src",
        "src/amatya-role-prompter",
        "src/janapada-memory", 
        "src/svami-orchestrator",
        "src/common",
        "tests",
        "docs",
        "scripts"
    ]
    
    for dir_path in expected_dirs:
        full_path = project_root / dir_path
        assert full_path.exists(), f"Directory {dir_path} should exist"
        assert full_path.is_dir(), f"{dir_path} should be a directory"


def test_python_files_exist():
    """Test that essential Python files exist."""
    expected_files = [
        "requirements.txt",
        "pyproject.toml",
        ".env.example",
        "src/__init__.py",
        "src/common/__init__.py",
        "src/common/config.py",
        "src/amatya-role-prompter/__init__.py",
        "src/janapada-memory/__init__.py",
        "src/svami-orchestrator/__init__.py"
    ]
    
    for file_path in expected_files:
        full_path = project_root / file_path
        assert full_path.exists(), f"File {file_path} should exist"
        assert full_path.is_file(), f"{file_path} should be a file"


def test_imports():
    """Test that basic imports work."""
    try:
        # Test that we can import from the common module
        from common.config import config
        assert config is not None
    except ImportError as e:
        assert False, f"Import failed: {e}"


def test_config_loading():
    """Test that configuration can be loaded."""
    from common.config import config
    
    # Test that config object exists
    assert config is not None
    
    # Test that default values are set
    assert config.GOOGLE_CLOUD_PROJECT == "konveyn2ai"
    assert config.GOOGLE_CLOUD_LOCATION == "us-central1"
    assert config.VECTOR_DIMENSIONS == 3072
    assert config.SIMILARITY_METRIC == "cosine"
    assert config.APPROXIMATE_NEIGHBORS_COUNT == 150


if __name__ == "__main__":
    test_project_structure()
    test_python_files_exist()
    test_imports()
    test_config_loading()
    print("All tests passed!")
