# Contributing to KonveyN2AI

Thank you for your interest in contributing to KonveyN2AI! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/neeharve/KonveyN2AI.git
   cd KonveyN2AI
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install pre-commit hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Code Style Guidelines

### Python Code Style

We use several tools to maintain consistent code quality:

- **Black**: Code formatting with 88-character line length
- **Ruff**: Fast Python linter for code quality
- **MyPy**: Static type checking
- **isort**: Import sorting

### Code Formatting

- **Line Length**: Maximum 88 characters (Black default)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Use double quotes for strings
- **Imports**: Sorted alphabetically, grouped by standard library, third-party, and local imports

### Type Hints

- Use type hints for all function parameters and return values
- Import types from `typing` module when needed
- Use `Optional[Type]` for nullable parameters

Example:
```python
from typing import Optional, List
from pydantic import BaseModel

def process_data(
    data: List[str],
    config: Optional[BaseModel] = None
) -> Dict[str, Any]:
    """Process data with optional configuration."""
    # Implementation here
    pass
```

### Documentation

- Use docstrings for all modules, classes, and functions
- Follow Google-style docstring format
- Include type information in docstrings when helpful

Example:
```python
def search_embeddings(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search for similar embeddings using vector similarity.

    Args:
        query: The search query string
        k: Number of results to return (default: 5)

    Returns:
        List of dictionaries containing search results with scores

    Raises:
        ValueError: If query is empty or k is negative
    """
    pass
```

## Git Workflow

### Branch Naming

Use the following format for branch names:
- `feat/task-<id>-<description>` - New features
- `fix/task-<id>-<description>` - Bug fixes
- `docs/task-<id>-<description>` - Documentation updates
- `refactor/task-<id>-<description>` - Code refactoring

### Commit Messages

Follow the conventional commit format:
```
<type>: <description> (task <id>)

<optional body>

<optional footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```
feat: implement vector search functionality (task 6)

- Add Vertex AI embeddings integration
- Implement FAISS fallback mechanism
- Add comprehensive error handling

Closes #123
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_project_setup.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names that explain what is being tested
- Follow the Arrange-Act-Assert pattern
- Mock external dependencies (APIs, databases)

## Pre-commit Hooks

Pre-commit hooks run automatically before each commit to ensure code quality:

- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure files end with newline
- **check-yaml**: Validate YAML files
- **black**: Format Python code
- **ruff**: Lint Python code
- **mypy**: Type checking
- **isort**: Sort imports
- **bandit**: Security checks

To run hooks manually:
```bash
pre-commit run --all-files
```

## Architecture Guidelines

### Component Structure

Follow the 3-tier architecture:
- **Amatya Role Prompter**: Role-based prompting and user interaction
- **Janapada Memory**: Memory management and vector embeddings
- **Svami Orchestrator**: Workflow orchestration and coordination

### Configuration

- Use environment variables for configuration
- Store sensitive data (API keys) in `.env` file
- Use `src/common/config.py` for centralized configuration management

### Error Handling

- Use specific exception types
- Provide meaningful error messages
- Log errors appropriately
- Implement graceful fallbacks where possible

## Questions?

If you have questions about contributing, please:
1. Check existing issues and discussions
2. Create a new issue with the `question` label
3. Reach out to the team through the project channels

Thank you for contributing to KonveyN2AI!
