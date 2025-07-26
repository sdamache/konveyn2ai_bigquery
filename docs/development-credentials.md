# Development Credentials Setup Guide

This guide explains how to set up Google Cloud credentials for local development of the KonveyN2AI project.

## 🔐 Security First

**CRITICAL**: Service account keys are stored in the `credentials/` directory and are automatically ignored by Git. Never commit credential files to the repository.

## Authentication Options

### Option 1: Application Default Credentials (Recommended)

This is the recommended approach for individual developers:

```bash
# Install Google Cloud SDK if not already installed
# https://cloud.google.com/sdk/docs/install

# Authenticate with your Google account
gcloud auth application-default login

# Set the default project
gcloud config set project konveyn2ai
```

**Pros:**
- More secure (uses your personal Google account)
- Automatically handles token refresh
- No key files to manage

**Cons:**
- Requires individual Google account setup
- May have different permissions than service accounts

### Option 2: Service Account Keys (Development Only)

For development environments that need service account access:

```bash
# Copy the environment template
cp .env.example .env

# Edit .env and uncomment ONE of the credential lines:
# GOOGLE_APPLICATION_CREDENTIALS=credentials/konveyn2ai-vertex-ai-key.json
# GOOGLE_APPLICATION_CREDENTIALS=credentials/konveyn2ai-cloud-run-key.json
```

## Service Account Key Files

Two service account keys are available for local development:

### 1. Vertex AI Service Account
- **File**: `credentials/konveyn2ai-vertex-ai-key.json`
- **Email**: `konveyn2ai-vertex-ai@konveyn2ai.iam.gserviceaccount.com`
- **Use For**: Amatya Role Prompter, Janapada Memory components
- **Permissions**: Vertex AI User role

### 2. Cloud Run Service Account
- **File**: `credentials/konveyn2ai-cloud-run-key.json`
- **Email**: `konveyn2ai-cloud-run@konveyn2ai.iam.gserviceaccount.com`
- **Use For**: Svami Orchestrator component
- **Permissions**: Cloud Run Admin, Artifact Registry Writer

## Environment Configuration

### Required Environment Variables

```bash
# Core Google Cloud settings
PROJECT_ID=konveyn2ai
REGION=us-central1
VECTOR_INDEX_ID=805460437066842112

# API Keys (get from Google Cloud Console)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_GEMINI_API_KEY=your_gemini_api_key_here

# Service URLs (for inter-component communication)
AMATYA_SERVICE_URL=https://amatya-role-prompter-72021522495.us-central1.run.app
JANAPADA_SERVICE_URL=https://janapada-memory-72021522495.us-central1.run.app
SVAMI_SERVICE_URL=https://svami-orchestrator-72021522495.us-central1.run.app
```

### Quick Setup Script

```bash
#!/bin/bash
# setup-dev-credentials.sh

echo "🚀 Setting up KonveyN2AI development credentials..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud CLI not installed. Please install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Setup Application Default Credentials
echo "Setting up Application Default Credentials..."
gcloud auth application-default login
gcloud config set project konveyn2ai

# Copy environment template
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file from template"
else
    echo "⚠️  .env file already exists"
fi

echo "✅ Development credentials setup complete!"
echo "📝 Next steps:"
echo "   1. Edit .env file with your API keys"
echo "   2. Run: source venv/bin/activate"
echo "   3. Test with: python test_credentials.py"
```

## Testing Credentials

### Basic Credential Test

Create and run this test script:

```python
# test_credentials.py
import os
from google.cloud import aiplatform
import google.auth

def test_credentials():
    try:
        # Initialize AI Platform
        aiplatform.init(project='konveyn2ai', location='us-central1')
        print("✅ Google Cloud credentials working")

        # Test vector index access
        indexes = list(aiplatform.MatchingEngineIndex.list())
        print(f"✅ Found {len(indexes)} vector indexes")

        # Check credentials type
        credentials, project = google.auth.default()
        print(f"✅ Auth type: {type(credentials).__name__}")
        print(f"✅ Project: {project}")

        return True

    except Exception as e:
        print(f"❌ Credential test failed: {e}")
        return False

if __name__ == "__main__":
    test_credentials()
```

### Component-Specific Testing

```python
# test_component_access.py
from google.cloud import aiplatform
from google.cloud import run_v2
from google.cloud import artifactregistry_v1

def test_vertex_ai_access():
    """Test Vertex AI permissions (Amatya, Janapada)"""
    try:
        aiplatform.init(project='konveyn2ai', location='us-central1')
        indexes = list(aiplatform.MatchingEngineIndex.list())
        print(f"✅ Vertex AI: {len(indexes)} indexes accessible")
        return True
    except Exception as e:
        print(f"❌ Vertex AI access failed: {e}")
        return False

def test_cloud_run_access():
    """Test Cloud Run permissions (Svami)"""
    try:
        client = run_v2.ServicesClient()
        parent = "projects/konveyn2ai/locations/us-central1"
        services = list(client.list_services(parent=parent))
        print(f"✅ Cloud Run: {len(services)} services accessible")
        return True
    except Exception as e:
        print(f"❌ Cloud Run access failed: {e}")
        return False

def test_artifact_registry_access():
    """Test Artifact Registry permissions"""
    try:
        client = artifactregistry_v1.ArtifactRegistryClient()
        parent = "projects/konveyn2ai/locations/us-central1"
        repos = list(client.list_repositories(parent=parent))
        print(f"✅ Artifact Registry: {len(repos)} repositories accessible")
        return True
    except Exception as e:
        print(f"❌ Artifact Registry access failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing component-specific access...")
    test_vertex_ai_access()
    test_cloud_run_access()
    test_artifact_registry_access()
```

## Troubleshooting

### Common Issues

1. **"Default credentials not found"**
   ```bash
   gcloud auth application-default login
   ```

2. **"Permission denied" errors**
   - Check that you're using the correct service account key
   - Verify the service account has the required IAM roles

3. **"Project not found"**
   ```bash
   gcloud config set project konveyn2ai
   ```

4. **Import errors for Google Cloud libraries**
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Key Regeneration

If service account keys are compromised:

```bash
# Delete old key (get key ID from Google Cloud Console)
gcloud iam service-accounts keys delete KEY_ID \
  --iam-account=konveyn2ai-vertex-ai@konveyn2ai.iam.gserviceaccount.com

# Generate new key
gcloud iam service-accounts keys create credentials/konveyn2ai-vertex-ai-key.json \
  --iam-account=konveyn2ai-vertex-ai@konveyn2ai.iam.gserviceaccount.com \
  --project=konveyn2ai
```

## Security Best Practices

1. **Never commit credential files** - They're in `.gitignore` but double-check
2. **Use Application Default Credentials when possible** - More secure than key files
3. **Rotate service account keys regularly** - Especially in shared environments
4. **Use least privilege principle** - Only grant necessary permissions
5. **Monitor credential usage** - Check Google Cloud audit logs

## Team Development

For team members:

1. Each developer should use their own Google account with Application Default Credentials
2. Service account keys are for CI/CD and specific development scenarios only
3. All team members need the `Vertex AI User` role on the project
4. Shared `.env` values should be documented in team chat/wiki (without secrets)

---

**Next Steps**: After setting up credentials, proceed to implement the three-component architecture with proper authentication.
