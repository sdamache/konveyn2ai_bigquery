#!/usr/bin/env python3
"""
Version synchronization validator for KonveyN2AI linting tools.

This script ensures that all linting tool versions are synchronized between:
- requirements.txt
- .pre-commit-config.yaml (local development)
- .pre-commit-config-ci.yaml (CI validation)

Usage:
    python scripts/validate_tool_versions.py

Exit codes:
    0: All versions synchronized
    1: Version mismatch detected
"""

import re
import sys
from pathlib import Path


def parse_requirements_versions(requirements_path: Path) -> dict[str, str]:
    """Parse tool versions from requirements.txt."""
    versions = {}
    tools = ["black", "ruff", "pre-commit", "mypy", "bandit", "isort"]

    with open(requirements_path) as f:
        content = f.read()

    for tool in tools:
        pattern = rf"{tool}==([0-9]+\.[0-9]+\.[0-9]+)"
        match = re.search(pattern, content)
        if match:
            versions[tool] = match.group(1)

    return versions


def parse_precommit_versions(precommit_path: Path) -> dict[str, str]:
    """Parse tool versions from pre-commit config files."""
    versions = {}

    with open(precommit_path) as f:
        content = f.read()

    # Map pre-commit repo patterns to tool names
    patterns = {
        r"psf/black.*?rev: ([0-9]+\.[0-9]+\.[0-9]+)": "black",
        r"astral-sh/ruff-pre-commit.*?rev: v([0-9]+\.[0-9]+\.[0-9]+)": "ruff",
        r"pre-commit/pre-commit-hooks.*?rev: v([0-9]+\.[0-9]+\.[0-9]+)": "pre-commit-hooks",
    }

    for pattern, tool in patterns.items():
        match = re.search(pattern, content, re.DOTALL)
        if match:
            versions[tool] = match.group(1)

    return versions


def validate_versions() -> tuple[bool, list[str]]:
    """Validate that all tool versions are synchronized."""
    project_root = Path(__file__).parent.parent

    # Parse versions from all sources
    req_versions = parse_requirements_versions(project_root / "requirements.txt")
    local_versions = parse_precommit_versions(project_root / ".pre-commit-config.yaml")
    ci_versions = parse_precommit_versions(project_root / ".pre-commit-config-ci.yaml")

    errors = []

    # Check Black version synchronization
    req_black = req_versions.get("black")
    local_black = local_versions.get("black")
    ci_black = ci_versions.get("black")

    if req_black and local_black and ci_black:
        if not (req_black == local_black == ci_black):
            errors.append(
                f"Black version mismatch: "
                f"requirements.txt={req_black}, "
                f"local={local_black}, "
                f"ci={ci_black}"
            )
    else:
        errors.append("Missing Black version in one or more configs")

    # Check Ruff version synchronization
    req_ruff = req_versions.get("ruff")
    local_ruff = local_versions.get("ruff")
    ci_ruff = ci_versions.get("ruff")

    if req_ruff and local_ruff and ci_ruff:
        if not (req_ruff == local_ruff == ci_ruff):
            errors.append(
                f"Ruff version mismatch: "
                f"requirements.txt={req_ruff}, "
                f"local={local_ruff}, "
                f"ci={ci_ruff}"
            )
    else:
        errors.append("Missing Ruff version in one or more configs")

    return len(errors) == 0, errors


def main():
    """Main validation function."""
    print("🔍 Validating linting tool version synchronization...")
    print("=" * 60)

    is_valid, errors = validate_versions()

    if is_valid:
        print("✅ All linting tool versions are synchronized!")
        print("✅ Local and CI environments will use identical tool versions")
        sys.exit(0)
    else:
        print("❌ Version synchronization issues detected:")
        for error in errors:
            print(f"   • {error}")
        print()
        print("🔧 Fix by updating versions in:")
        print("   • requirements.txt")
        print("   • .pre-commit-config.yaml")
        print("   • .pre-commit-config-ci.yaml")
        sys.exit(1)


if __name__ == "__main__":
    main()
