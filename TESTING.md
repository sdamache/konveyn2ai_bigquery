# Testing Guide for KonveyN2AI

This document provides comprehensive information about testing in the KonveyN2AI project.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Types](#test-types)
- [Writing Tests](#writing-tests)
- [Coverage Requirements](#coverage-requirements)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

## Overview

KonveyN2AI uses a comprehensive testing strategy with:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test service interactions
- **Docker Tests**: Test containerized deployments
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test system performance under load

### Test Framework Stack

- **pytest**: Primary testing framework
- **pytest-asyncio**: Async test support
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking utilities
- **pytest-httpx**: HTTP client testing
- **docker**: Container testing

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests
│   ├── svami/              # Svami Orchestrator tests
│   ├── janapada/           # Janapada Memory tests
│   ├── amatya/             # Amatya Role Prompter tests
│   └── common/             # Common models tests
├── integration/            # Integration tests
│   └── test_service_interactions.py
├── fixtures/               # Test fixtures
│   └── docker_fixtures.py  # Docker-related fixtures
└── utils/                  # Test utilities
    └── test_helpers.py      # Helper functions and classes
```

## Running Tests

### Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run all tests (excluding Docker)
pytest tests/ -m "not docker"

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Interactive Test Runner

```bash
# Use the interactive test runner
python test_runner.py
```

### Shell Script Runner

```bash
# Use the shell script for advanced options
./scripts/run-tests.sh --help
./scripts/run-tests.sh --unit --coverage
./scripts/run-tests.sh --integration
./scripts/run-tests.sh --docker
```

### Specific Test Types

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only  
pytest tests/integration/ -v

# Docker tests (requires Docker running)
pytest tests/ -m docker -v

# Performance tests
pytest tests/ -m slow -v

# Exclude slow tests
pytest tests/ -m "not slow" -v
```

## Test Types

### Unit Tests

Test individual components in isolation with mocked dependencies.

**Example:**
```python
def test_query_request_validation():
    """Test QueryRequest model validation."""
    with pytest.raises(ValidationError):
        QueryRequest(question="", role="developer")
```

**Location**: `tests/unit/`
**Markers**: `@pytest.mark.unit`

### Integration Tests

Test interactions between services and components.

**Example:**
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_workflow(async_client):
    """Test complete workflow through all services."""
    response = await async_client.post("/answer", json=query_data)
    assert response.status_code == 200
```

**Location**: `tests/integration/`
**Markers**: `@pytest.mark.integration`

### Docker Tests

Test containerized services and Docker Compose orchestration.

**Example:**
```python
@pytest.mark.docker
def test_docker_services_health(docker_services):
    """Test that all Docker services are healthy."""
    for service in docker_services:
        assert check_service_health(service)
```

**Location**: Mixed with other tests
**Markers**: `@pytest.mark.docker`

### Performance Tests

Test system performance and response times.

**Example:**
```python
@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test handling of concurrent requests."""
    tasks = [make_request() for _ in range(100)]
    results = await asyncio.gather(*tasks)
    assert all(r.status_code == 200 for r in results)
```

**Markers**: `@pytest.mark.slow`

## Writing Tests

### Test Naming Convention

- Test files: `test_*.py` or `*_test.py`
- Test functions: `test_*`
- Test classes: `Test*`

### Using Fixtures

```python
def test_with_mock_services(mock_janapada_client, mock_amatya_client):
    """Test using shared fixtures."""
    # Test implementation
    pass

def test_with_sample_data(sample_snippets, sample_query):
    """Test using sample data fixtures."""
    # Test implementation
    pass
```

### Async Tests

```python
@pytest.mark.asyncio
async def test_async_operation():
    """Test async operations."""
    result = await some_async_function()
    assert result is not None
```

### Mocking External Dependencies

```python
@patch("google.generativeai.GenerativeModel")
def test_with_mocked_gemini(mock_model):
    """Test with mocked external service."""
    mock_model.return_value.generate_content.return_value.text = "test response"
    # Test implementation
```

### Error Testing

```python
def test_error_handling():
    """Test error handling scenarios."""
    with pytest.raises(ValidationError):
        invalid_operation()
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input_data,expected", [
    ("valid_input", "expected_output"),
    ("another_input", "another_output"),
])
def test_multiple_scenarios(input_data, expected):
    """Test multiple scenarios."""
    result = process_input(input_data)
    assert result == expected
```

## Coverage Requirements

### Coverage Targets

- **Overall**: 80% minimum
- **Unit Tests**: 90% minimum
- **Critical Paths**: 100% required

### Coverage Commands

```bash
# Generate coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing

# Check coverage threshold
pytest --cov=src --cov-fail-under=80

# View HTML report
open htmlcov/index.html
```

### Coverage Configuration

Coverage settings are in `.coveragerc`:

```ini
[run]
source = src
omit = */tests/*, */venv/*

[report]
show_missing = True
skip_covered = False
```

## CI/CD Integration

### GitHub Actions

Tests run automatically on:
- Push to main/develop branches
- Pull requests
- Multiple Python versions (3.9, 3.10, 3.11)

### Workflow Jobs

1. **Linting**: Code quality checks
2. **Type Checking**: MyPy validation
3. **Unit Tests**: Fast, isolated tests
4. **Integration Tests**: Service interaction tests
5. **Docker Tests**: Container validation
6. **Security Scans**: Vulnerability checks
7. **Coverage**: Coverage threshold enforcement

### Quality Gates

All tests must pass for:
- Pull request approval
- Merge to main branch
- Production deployment

## Troubleshooting

### Common Issues

#### Tests Not Found

```bash
# Check test discovery
pytest --collect-only

# Verify Python path
export PYTHONPATH=$PWD:$PYTHONPATH
```

#### Import Errors

```bash
# Install dependencies
pip install -r requirements.txt

# Check virtual environment
which python
which pytest
```

#### Docker Tests Failing

```bash
# Check Docker status
docker info

# Start Docker services
docker-compose up -d

# Check service health
curl http://localhost:8080/health
```

#### Slow Tests

```bash
# Skip slow tests
pytest -m "not slow"

# Run specific fast tests
pytest tests/unit/common/
```

#### Coverage Issues

```bash
# Clean coverage data
rm -f .coverage coverage.xml
rm -rf htmlcov/

# Regenerate coverage
pytest --cov=src --cov-report=html
```

### Performance Optimization

#### Parallel Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto
```

#### Test Selection

```bash
# Run only changed tests
pytest --lf  # Last failed
pytest --ff  # Failed first

# Run specific markers
pytest -m "unit and not slow"
```

### Debug Mode

```bash
# Verbose output
pytest -v -s

# Debug on failure
pytest --pdb

# Debug specific test
pytest tests/unit/svami/test_main.py::test_specific -v -s
```

## Test Data Management

### Sample Data

Use shared fixtures for consistent test data:

```python
@pytest.fixture
def sample_snippets():
    return MockDataGenerator.generate_multiple_snippets(5)
```

### Environment Variables

Test environment variables are managed in `conftest.py`:

```python
@pytest.fixture
def mock_env_vars():
    with patch.dict(os.environ, test_env_vars):
        yield test_env_vars
```

### External Services

Mock external services to avoid dependencies:

```python
@pytest.fixture
def mock_vertex_ai():
    with patch("vertexai.init"):
        yield mock_vertex_setup
```

## Best Practices

### Test Independence

- Each test should be independent
- Use fixtures for setup/teardown
- Don't rely on test execution order

### Mocking Strategy

- Mock external dependencies
- Keep real business logic unmocked
- Use realistic mock data

### Assertion Clarity

```python
# Good: Clear assertion message
assert response.status_code == 200, f"Expected 200, got {response.status_code}"

# Good: Specific assertions
assert "error" not in response.json()
assert len(response.json()["sources"]) > 0
```

### Test Documentation

- Use descriptive test names
- Include docstrings for complex tests
- Comment non-obvious test logic

### Resource Management

- Clean up resources in fixtures
- Use context managers
- Avoid resource leaks

## Maintenance

### Regular Tasks

1. **Update test dependencies**: Keep pytest and plugins updated
2. **Review coverage**: Ensure coverage targets are met
3. **Cleanup obsolete tests**: Remove tests for deprecated features
4. **Performance monitoring**: Track test execution times

### Test Review Process

1. **Code Review**: All test changes require review
2. **Coverage Check**: New code must maintain coverage thresholds
3. **Integration Testing**: Test changes with full system
4. **Documentation**: Update test documentation for significant changes

---

For additional help, see:
- [pytest documentation](https://docs.pytest.org/)
- [Project CLAUDE.md](./CLAUDE.md) for development guidelines
- [Docker testing guide](./README.docker.md) for container testing