# Configuration Management

KonveyN2AI uses environment variables for configuration management, providing flexibility for different deployment environments while maintaining security for sensitive data.

## Configuration Files

### .env.example
Template file containing all required and optional environment variables with example values. Copy this file to `.env` and fill in your actual values.

### src/common/config.py
Central configuration module that loads and validates environment variables, providing a single source of truth for application configuration.

## Environment Variables

### Required API Keys

These API keys are required for the application to function:

| Variable | Description | Example |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | API key for Claude model access via Augment Agent | `sk-ant-api03-...` |
| `GOOGLE_API_KEY` | API key for Google Gemini models | `AIzaSy...` |

### Optional API Keys

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `PERPLEXITY_API_KEY` | API key for research capabilities | None | `pplx-...` |

### Google Cloud Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `GOOGLE_CLOUD_PROJECT` | Google Cloud project ID | `konveyn2ai` | `my-project-123` |
| `GOOGLE_CLOUD_LOCATION` | Google Cloud region | `us-central1` | `us-west1` |

### Application Configuration

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `ENVIRONMENT` | Application environment | `development` | `development`, `production`, `testing` |
| `DEBUG` | Enable debug mode | `true` | `true`, `false` |
| `LOG_LEVEL` | Logging level | `info` | `debug`, `info`, `warning`, `error` |

### Vector Index Configuration

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `VECTOR_DIMENSIONS` | Embedding vector dimensions | `3072` | Must match model output |
| `SIMILARITY_METRIC` | Vector similarity metric | `cosine` | `cosine`, `euclidean`, `dot_product` |
| `APPROXIMATE_NEIGHBORS_COUNT` | Number of approximate neighbors | `150` | Affects search quality vs speed |

### Service URLs

For local development and service communication:

| Variable | Description | Default |
|----------|-------------|---------|
| `AMATYA_SERVICE_URL` | Amatya Role Prompter service URL | `http://localhost:8001` |
| `JANAPADA_SERVICE_URL` | Janapada Memory service URL | `http://localhost:8002` |
| `SVAMI_SERVICE_URL` | Svami Orchestrator service URL | `http://localhost:8003` |

## Configuration Usage

### Loading Configuration

```python
from src.common.config import config

# Access configuration values
project_id = config.GOOGLE_CLOUD_PROJECT
api_key = config.GOOGLE_API_KEY
debug_mode = config.DEBUG
```

### Environment Detection

```python
from src.common.config import config

if config.is_development():
    print("Running in development mode")

if config.is_production():
    print("Running in production mode")
```

### Configuration Validation

The configuration system automatically validates required API keys:

```python
from src.common.config import config

try:
    config.validate_required_keys()
    print("All required configuration is present")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Environment-Specific Configuration

### Development Environment

```bash
# .env for development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=debug
GOOGLE_CLOUD_PROJECT=konveyn2ai-dev
```

### Production Environment

```bash
# .env for production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info
GOOGLE_CLOUD_PROJECT=konveyn2ai-prod
```

### Testing Environment

```bash
# .env.test for testing
ENVIRONMENT=testing
DEBUG=false
LOG_LEVEL=warning
# Use test project and mock services
GOOGLE_CLOUD_PROJECT=konveyn2ai-test
```

## Security Considerations

1. **Never commit .env files** - They contain sensitive API keys
2. **Use different API keys** for different environments
3. **Rotate API keys regularly** in production
4. **Use Google Cloud IAM** for fine-grained permissions
5. **Monitor API usage** to detect unauthorized access

## Configuration Best Practices

1. **Use descriptive variable names** that clearly indicate their purpose
2. **Provide sensible defaults** for non-sensitive configuration
3. **Document all configuration options** with examples
4. **Validate configuration** at application startup
5. **Use environment-specific configurations** for different deployment stages
6. **Keep sensitive data separate** from application code

## Troubleshooting

### Common Issues

**Missing API Keys**
```
ValueError: Missing required environment variables: GOOGLE_API_KEY
```
Solution: Copy `.env.example` to `.env` and fill in your API keys.

**Invalid Google Cloud Project**
```
google.api_core.exceptions.NotFound: Project 'invalid-project' not found
```
Solution: Verify `GOOGLE_CLOUD_PROJECT` matches your actual Google Cloud project ID.

**Service Connection Errors**
```
ConnectionError: Failed to connect to http://localhost:8001
```
Solution: Check that service URLs in configuration match running services.

### Configuration Debugging

Enable debug logging to see configuration loading:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from src.common.config import config
# Configuration loading details will be logged
```

## Adding New Configuration

To add new configuration options:

1. **Add to .env.example** with documentation
2. **Add to Config class** in `src/common/config.py`
3. **Update this documentation** with the new option
4. **Add tests** to verify the new configuration works
5. **Update deployment scripts** if needed

Example:
```python
# In src/common/config.py
class Config:
    # ... existing configuration ...
    NEW_FEATURE_ENABLED: bool = os.getenv("NEW_FEATURE_ENABLED", "false").lower() == "true"
```
