#!/usr/bin/env python3
"""
Simple test runner for KonveyN2AI with interactive options.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return the result."""
    print(f"🔄 {description}")
    print(f"   Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} - SUCCESS")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"   Error: {e.stderr}")
        return False, e.stderr


def main():
    """Main test runner with interactive menu."""

    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    print("🧪 KonveyN2AI Test Runner")
    print("=" * 40)

    # Check if virtual environment is activated
    if not sys.prefix != sys.base_prefix:
        print("⚠️  Virtual environment not detected.")
        print("   Please activate venv: source venv/bin/activate")
        print(
            "   Or run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        )
        sys.exit(1)

    while True:
        print("\nTest Options:")
        print("1. 🧪 Run unit tests only")
        print("2. 🔗 Run integration tests only")
        print("3. 🚀 Run all tests")
        print("4. 📊 Run tests with coverage")
        print("5. 🐳 Run Docker tests")
        print("6. ⚡ Run quick smoke tests")
        print("7. 🧹 Run linting and formatting checks")
        print("8. 🔒 Run security checks")
        print("9. 📈 Generate coverage report")
        print("0. 🚪 Exit")

        choice = input("\nSelect option (0-9): ").strip()

        if choice == "0":
            print("👋 Goodbye!")
            break

        elif choice == "1":
            success, _ = run_command(
                ["pytest", "tests/unit/", "-v"], "Running unit tests"
            )

        elif choice == "2":
            success, _ = run_command(
                ["pytest", "tests/integration/", "-v", "-m", "not docker"],
                "Running integration tests (excluding Docker)",
            )

        elif choice == "3":
            success, _ = run_command(
                ["pytest", "tests/", "-v", "-m", "not docker"],
                "Running all tests (excluding Docker)",
            )

        elif choice == "4":
            success, _ = run_command(
                [
                    "pytest",
                    "tests/",
                    "-v",
                    "-m",
                    "not docker",
                    "--cov=src",
                    "--cov-report=html",
                    "--cov-report=term-missing",
                ],
                "Running all tests with coverage",
            )
            if success:
                print("📊 Coverage report generated in htmlcov/index.html")

        elif choice == "5":
            print("🐳 Docker tests require Docker to be running")
            confirm = input("Continue? (y/N): ").strip().lower()
            if confirm == "y":
                success, _ = run_command(
                    ["pytest", "tests/", "-v", "-m", "docker"],
                    "Running Docker-based tests",
                )

        elif choice == "6":
            # Quick smoke tests - run a few key tests
            success, _ = run_command(
                [
                    "pytest",
                    "tests/unit/common/test_models.py",
                    "tests/unit/svami/test_main.py::TestSvamiOrchestrator::test_health_endpoint",
                    "-v",
                ],
                "Running quick smoke tests",
            )

        elif choice == "7":
            print("🧹 Running code quality checks...")

            # Ruff linting
            success1, _ = run_command(
                ["ruff", "check", "src/", "tests/"], "Linting with ruff"
            )

            # Black formatting check
            success2, _ = run_command(
                ["black", "--check", "src/", "tests/"],
                "Checking code formatting with black",
            )

            # isort import sorting check
            success3, _ = run_command(
                ["isort", "--check-only", "src/", "tests/"],
                "Checking import sorting with isort",
            )

            # MyPy type checking
            success4, _ = run_command(
                ["mypy", "src/", "--ignore-missing-imports"], "Type checking with mypy"
            )

            if all([success1, success2, success3, success4]):
                print("✅ All code quality checks passed!")
            else:
                print("❌ Some code quality checks failed")

        elif choice == "8":
            success, _ = run_command(
                ["bandit", "-r", "src/"], "Running security checks with bandit"
            )

        elif choice == "9":
            success, _ = run_command(["coverage", "html"], "Generating coverage report")
            if success:
                print("📊 Coverage report generated in htmlcov/index.html")

        else:
            print("❌ Invalid option. Please choose 0-9.")
            continue

        # Ask if user wants to continue
        if choice != "0":
            print("\n" + "=" * 50)
            cont = input("Press Enter to continue or 'q' to quit: ").strip().lower()
            if cont == "q":
                break


if __name__ == "__main__":
    main()
