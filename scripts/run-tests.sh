#!/bin/bash

# Test runner script for KonveyN2AI
# Runs different types of tests with appropriate configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/venv"

echo -e "${BLUE}🧪 KonveyN2AI Test Runner${NC}"
echo "================================="

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    print_error "Virtual environment not found at $VENV_PATH"
    print_warning "Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    print_error "pytest not found. Installing test dependencies..."
    pip install -r requirements.txt
fi

# Default test options
PYTEST_ARGS=""
TEST_TYPE="all"
COVERAGE=true
PARALLEL=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            shift
            ;;
        --e2e)
            TEST_TYPE="e2e"
            shift
            ;;
        --docker)
            TEST_TYPE="docker"
            shift
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --unit           Run only unit tests"
            echo "  --integration    Run only integration tests"
            echo "  --e2e           Run only end-to-end tests"
            echo "  --docker        Run only Docker-based tests"
            echo "  --no-coverage   Skip coverage reporting"
            echo "  --parallel      Run tests in parallel"
            echo "  --verbose       Verbose output"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            PYTEST_ARGS="$PYTEST_ARGS $1"
            shift
            ;;
    esac
done

# Set up pytest arguments based on test type
case $TEST_TYPE in
    unit)
        PYTEST_ARGS="$PYTEST_ARGS -m unit"
        print_status "Running unit tests only"
        ;;
    integration)
        PYTEST_ARGS="$PYTEST_ARGS -m integration"
        print_status "Running integration tests only"
        ;;
    e2e)
        PYTEST_ARGS="$PYTEST_ARGS -m e2e"
        print_status "Running end-to-end tests only"
        ;;
    docker)
        PYTEST_ARGS="$PYTEST_ARGS -m docker"
        print_status "Running Docker-based tests only"
        ;;
    all)
        print_status "Running all tests"
        ;;
esac

# Add coverage if enabled
if [ "$COVERAGE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS --cov=src --cov-report=html --cov-report=xml --cov-report=term-missing"
    print_status "Coverage reporting enabled"
fi

# Add parallel execution if enabled
if [ "$PARALLEL" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS -n auto"
    print_status "Parallel execution enabled"
fi

# Add verbose output if enabled
if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS -v"
fi

# Change to project root
cd "$PROJECT_ROOT"

# Clean previous coverage data
if [ "$COVERAGE" = true ]; then
    rm -f .coverage coverage.xml
    rm -rf htmlcov/
fi

print_status "Starting tests..."
echo ""

# Run tests
if pytest $PYTEST_ARGS; then
    echo ""
    print_status "All tests passed!"
    
    if [ "$COVERAGE" = true ]; then
        echo ""
        print_status "Coverage report generated:"
        echo "  - HTML: htmlcov/index.html"
        echo "  - XML: coverage.xml"
        
        # Show coverage summary
        echo ""
        echo -e "${BLUE}Coverage Summary:${NC}"
        coverage report --show-missing | tail -1
    fi
    
    exit 0
else
    echo ""
    print_error "Some tests failed!"
    exit 1
fi