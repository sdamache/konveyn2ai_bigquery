"""
Test project setup and basic functionality.

This module tests that the project structure is correct and basic
imports work as expected.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Project root for file path testing
project_root = Path(__file__).parent.parent


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
        "scripts",
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
        "src/svami-orchestrator/__init__.py",
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
        raise AssertionError(f"Import failed: {e}") from e


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


def test_component_requirements_exist():
    """Test that component-specific requirements files exist."""
    component_requirements = [
        "src/amatya-role-prompter/requirements.txt",
        "src/janapada-memory/requirements.txt",
        "src/svami-orchestrator/requirements.txt",
        "src/common/requirements.txt",
    ]

    for req_file in component_requirements:
        full_path = project_root / req_file
        assert full_path.exists(), f"Requirements file {req_file} should exist"
        assert full_path.is_file(), f"{req_file} should be a file"

        # Check that file is not empty
        content = full_path.read_text()
        assert len(content.strip()) > 0, f"{req_file} should not be empty"


def test_poetry_configuration():
    """Test that Poetry configuration is present in pyproject.toml."""
    pyproject_path = project_root / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml should exist"

    content = pyproject_path.read_text()
    assert "[tool.poetry]" in content, "Poetry configuration should be present"
    assert (
        "[tool.poetry.dependencies]" in content
    ), "Poetry dependencies should be configured"
    assert (
        "[tool.poetry.group.dev.dependencies]" in content
    ), "Poetry dev dependencies should be configured"


def test_code_quality_tools():
    """Test that code quality tools configuration exists."""
    # Test pre-commit configuration
    precommit_path = project_root / ".pre-commit-config.yaml"
    assert precommit_path.exists(), ".pre-commit-config.yaml should exist"

    content = precommit_path.read_text()
    assert "black" in content, "Black should be configured in pre-commit"

    # Test CONTRIBUTING.md exists
    contributing_path = project_root / "CONTRIBUTING.md"
    assert contributing_path.exists(), "CONTRIBUTING.md should exist"

    contributing_content = contributing_path.read_text()
    assert (
        "Code Style Guidelines" in contributing_content
    ), "Code style guidelines should be documented"
    assert "pre-commit" in contributing_content, "Pre-commit should be documented"


def test_configuration_management():
    """Test that configuration management system is properly set up."""
    # Test configuration documentation exists
    config_docs_path = project_root / "docs" / "configuration.md"
    assert config_docs_path.exists(), "Configuration documentation should exist"

    config_docs_content = config_docs_path.read_text()
    assert (
        "Environment Variables" in config_docs_content
    ), "Environment variables should be documented"
    assert (
        "GOOGLE_API_KEY" in config_docs_content
    ), "Required API keys should be documented"
    assert (
        "Configuration Usage" in config_docs_content
    ), "Usage examples should be provided"

    # Test that config validation works
    from common.config import config

    # Test environment detection methods exist
    assert hasattr(config, "is_development"), "is_development method should exist"
    assert hasattr(config, "is_production"), "is_production method should exist"
    assert hasattr(
        config, "validate_required_keys"
    ), "validate_required_keys method should exist"

    # Test default values are set correctly
    assert config.GOOGLE_CLOUD_PROJECT == "konveyn2ai", "Default project should be set"
    assert (
        config.GOOGLE_CLOUD_LOCATION == "us-central1"
    ), "Default location should be set"
    assert config.VECTOR_DIMENSIONS == 3072, "Default vector dimensions should be set"


def test_ci_cd_pipeline():
    """Test that CI/CD pipeline configuration is properly set up."""
    # Test GitHub Actions workflow exists
    workflows_dir = project_root / ".github" / "workflows"
    assert workflows_dir.exists(), "GitHub workflows directory should exist"

    ci_workflow = workflows_dir / "ci.yml"
    assert ci_workflow.exists(), "CI workflow file should exist"

    ci_content = ci_workflow.read_text()
    assert "name: CI" in ci_content, "CI workflow should be named correctly"
    assert "python-version:" in ci_content, "Python version matrix should be configured"
    assert "pre-commit run" in ci_content, "Pre-commit hooks should run in CI"
    assert "pytest" in ci_content, "Tests should run in CI"
    # Note: Code quality checks (black, ruff, mypy, bandit) are now handled by pre-commit
    assert (
        "code quality" in ci_content.lower()
    ), "Code quality should be mentioned in CI"

    # Test status badges in README
    readme_path = project_root / "README.md"
    readme_content = readme_path.read_text()
    assert "[![CI]" in readme_content, "CI status badge should be in README"
    assert "[![Python" in readme_content, "Python version badge should be in README"
    assert "[![Code style: black]" in readme_content, "Black badge should be in README"


if __name__ == "__main__":
    test_project_structure()
    test_python_files_exist()
    test_imports()
    test_config_loading()
    test_component_requirements_exist()
    test_poetry_configuration()
    test_code_quality_tools()
    test_configuration_management()
    test_ci_cd_pipeline()
    print("All tests passed!")
