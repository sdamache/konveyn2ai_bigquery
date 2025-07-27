# Import Standards for KonveyN2AI

This document defines the standard import patterns for KonveyN2AI testing and development. Following these patterns ensures consistent, maintainable, and scalable code across the project.

## Quick Start

### For Service Tests
```python
# ✅ CORRECT - Use centralized utilities
from tests.utils.test_clients import create_mocked_svami_client

# Create fully configured test client with mocks
setup = create_mocked_svami_client()
client = setup['client']
mock_janapada_call = setup['janapada_call']  
mock_amatya_call = setup['amatya_call']
```

### For Simple Tests
```python
# ✅ CORRECT - Use simple client factory
from tests.utils.test_clients import create_simple_client

client = create_simple_client('amatya')
response = client.get("/health")
```

### For Unit Tests
```python
# ✅ CORRECT - Use service imports
from tests.utils.service_imports import get_service_app

app = get_service_app('janapada')
```

## Architecture

### Import Hierarchy
```
tests/
├── utils/
│   ├── service_imports.py    # Service module importing
│   └── test_clients.py       # TestClient factory & mocking
├── conftest.py               # Global fixtures
└── [test files]              # Clean, simple imports only
```

### Design Principles
1. **Single Source of Truth**: All import logic centralized in `tests/utils/`
2. **Zero Duplication**: No repeated import patterns across test files
3. **Environment Consistency**: Identical behavior in local and CI environments
4. **Industry Standard**: Following Python packaging best practices

## Common Patterns

### Integration Testing Pattern
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_service_workflow():
    """Standard pattern for service integration tests."""
    
    # Clean import - no sys.path manipulation needed
    from tests.utils.test_clients import create_mocked_svami_client
    
    # Get configured client and mocks
    setup = create_mocked_svami_client()
    client = setup['client']
    mock_janapada_call = setup['janapada_call']
    mock_amatya_call = setup['amatya_call']
    
    # Configure mock responses
    mock_janapada_call.return_value = JsonRpcResponse(
        id="test", result={"snippets": []}
    )
    
    # Test logic here
    response = client.post("/answer", json=request_data)
    assert response.status_code == 200
```

### Unit Testing Pattern  
```python
def test_service_endpoint():
    """Standard pattern for service unit tests."""
    
    # Clean import - service-specific client
    from tests.utils.test_clients import create_simple_client
    
    # Get service client
    client = create_simple_client('janapada')
    
    # Test logic here
    response = client.get("/health")
    assert response.status_code == 200
```

### Service Module Access Pattern
```python
def test_service_internals():
    """Access service internals for testing."""
    
    from tests.utils.service_imports import get_service_main
    
    # Get service main module for internal access
    main = get_service_main('svami')
    
    # Test internal functionality
    assert hasattr(main, 'janapada_client')
```

## Migration Guide

### From Old Pattern (❌ DON'T USE)
```python
# ❌ OLD - Complex sys.path manipulation
import sys
import os

svami_path = os.path.join(os.path.dirname(__file__), "../../src/svami-orchestrator")
svami_path = os.path.abspath(svami_path)
if svami_path not in sys.path:
    sys.path.insert(0, svami_path)

from main import app
import main
from common.rpc_client import JsonRpcClient

main.janapada_client = JsonRpcClient("http://localhost:8001")
main.amatya_client = JsonRpcClient("http://localhost:8002")

with (
    patch.object(main.janapada_client, "call", new_callable=AsyncMock) as mock_janapada,
    patch.object(main.amatya_client, "call", new_callable=AsyncMock) as mock_amatya,
):
    client = TestClient(app)
    # ... test logic
```

### To New Pattern (✅ USE THIS)
```python
# ✅ NEW - Clean centralized utilities
from tests.utils.test_clients import create_mocked_svami_client

setup = create_mocked_svami_client()
client = setup['client']
mock_janapada_call = setup['janapada_call']
mock_amatya_call = setup['amatya_call']

# Mock context and test logic
with (mock_janapada_call, mock_amatya_call):
    # ... test logic
```

**Benefits**:
- 95% reduction in code (25+ lines → 6 lines)
- Zero sys.path manipulation
- Consistent CI/local behavior
- Industry-standard pattern

## Available Utilities

### test_clients.py Functions

#### `create_mocked_svami_client()`
Creates a fully configured Svami test client with mocked dependencies.

**Returns**: 
```python
{
    'client': TestClient,           # FastAPI TestClient instance
    'service_client': ServiceTestClient,  # Enhanced wrapper
    'janapada_call': AsyncMock,     # Mock for Janapada service calls
    'amatya_call': AsyncMock,       # Mock for Amatya service calls  
    'JsonRpcResponse': Class        # Response model class
}
```

**Example**:
```python
setup = create_mocked_svami_client()
setup['janapada_call'].return_value = JsonRpcResponse(id="test", result={"snippets": []})
response = setup['client'].post("/answer", json=request_data)
```

#### `create_test_client(service_name, auto_init=True)`
Creates a ServiceTestClient for any service with optional auto-initialization.

**Parameters**:
- `service_name`: One of 'svami', 'amatya', 'janapada'  
- `auto_init`: Whether to auto-initialize service clients (default: True)

**Returns**: `ServiceTestClient` instance

**Example**:
```python
service_client = create_test_client('amatya')
client = service_client.get_client()
main = service_client.get_main_module()
```

#### `create_simple_client(service_name)`
Creates a basic TestClient without additional setup for simple tests.

**Parameters**:
- `service_name`: One of 'svami', 'amatya', 'janapada'

**Returns**: `TestClient` instance

**Example**:
```python
client = create_simple_client('janapada')
response = client.get("/health")
```

### service_imports.py Functions

#### `get_service_app(service_name)`
Get the FastAPI app instance for a service.

**Parameters**:
- `service_name`: One of 'svami', 'amatya', 'janapada'

**Returns**: FastAPI app instance

#### `get_service_main(service_name)`  
Get the main module for a service (for accessing global variables).

**Parameters**:
- `service_name`: One of 'svami', 'amatya', 'janapada'

**Returns**: Service main module

#### `import_common_models()`
Import common models and utilities.

**Returns**: 
```python
{
    'JsonRpcResponse': JsonRpcResponse,
    'QueryRequest': QueryRequest,
    'AdviceRequest': AdviceRequest, 
    'SearchRequest': SearchRequest,
    'JsonRpcClient': JsonRpcClient
}
```

#### `cleanup_service_imports()`
Clean up imported service modules to prevent contamination.

**Usage**: Call in test teardown functions.

## Test Data Factories

### Request Factories
```python
from tests.utils.test_clients import (
    create_sample_query_request,
    create_sample_search_request, 
    create_sample_advice_request
)

# Get sample data for testing
query_data = create_sample_query_request()
search_data = create_sample_search_request()
advice_data = create_sample_advice_request()
```

### Error Simulation
```python
from tests.utils.test_clients import simulate_service_failure

# Simulate different failure types
simulate_service_failure(mock_call, "timeout")
simulate_service_failure(mock_call, "connection")
simulate_service_failure(mock_call, "server_error")
```

## Troubleshooting

### Common Issues

#### Import Error: "No module named 'tests.utils'"
**Solution**: Ensure you're running tests with `PYTHONPATH=src` or from project root.

```bash
# ✅ Correct way to run tests
PYTHONPATH=src python -m pytest tests/
```

#### Service Import Error
**Solution**: Verify service name is correct and service files exist.

```python
# ✅ Valid service names
SUPPORTED_SERVICES = ['svami', 'amatya', 'janapada']
```

#### Mock Not Working
**Solution**: Ensure you're using the returned mock objects from the utilities.

```python
# ✅ Correct mock usage
setup = create_mocked_svami_client()
mock_call = setup['janapada_call']  # Use the returned mock
mock_call.return_value = expected_response
```

### Performance Tips

1. **Use simple clients** for basic tests that don't need mocking
2. **Call cleanup functions** in test teardown to prevent contamination
3. **Reuse service clients** within test classes when possible
4. **Avoid sys.path manipulation** - use utilities instead

## Best Practices

### DO ✅
- Use centralized utilities from `tests/utils/`
- Import only what you need for each test
- Use the appropriate client factory for your test type
- Follow the established patterns consistently
- Call cleanup functions in test teardown

### DON'T ❌  
- Manipulate `sys.path` in test files
- Create `TestClient` instances manually
- Duplicate import logic across tests
- Import service modules directly
- Use different import patterns in different files

## Examples by Use Case

### Health Endpoint Testing
```python
def test_service_health():
    from tests.utils.test_clients import create_simple_client
    
    client = create_simple_client('amatya')
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### Service Communication Testing
```python
@pytest.mark.integration
async def test_service_communication():
    from tests.utils.test_clients import create_mocked_svami_client
    
    setup = create_mocked_svami_client()
    client = setup['client']
    mock_janapada = setup['janapada_call']
    
    # Configure mock
    mock_janapada.return_value = JsonRpcResponse(
        id="test", result={"snippets": []}
    )
    
    # Test communication
    response = client.post("/answer", json=query_data)
    assert response.status_code == 200
    mock_janapada.assert_called_once()
```

### Error Handling Testing
```python
async def test_service_failure_handling():
    from tests.utils.test_clients import create_test_client, simulate_service_failure
    
    service_client = create_test_client('svami')
    client = service_client.get_client()
    main = service_client.get_main_module()
    
    with patch.object(main, "janapada_client", create=True) as mock_client:
        simulate_service_failure(mock_client.call, "timeout")
        
        response = client.post("/answer", json=query_data)
        # Test graceful degradation
        assert response.status_code == 200
        assert "service unavailable" in response.json()["message"]
```

## Contributing

When adding new test utilities:

1. **Follow the established patterns** in `tests/utils/`
2. **Add comprehensive docstrings** with examples
3. **Include error handling** for invalid inputs
4. **Update this documentation** with new functions
5. **Add unit tests** for new utility functions

## References

- **Python Packaging**: [PEP 517](https://peps.python.org/pep-0517/), [PEP 518](https://peps.python.org/pep-0518/)
- **pytest Best Practices**: [pytest documentation](https://docs.pytest.org/)
- **FastAPI Testing**: [FastAPI testing guide](https://fastapi.tiangolo.com/tutorial/testing/)
- **Industry Examples**: requests, django, flask testing patterns

---

**Document Version**: 1.0.0  
**Last Updated**: 2025-07-27  
**Status**: ✅ Active Standard