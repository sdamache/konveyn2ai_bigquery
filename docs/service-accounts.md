# Google Cloud Service Accounts Configuration

This document outlines the service accounts created for the KonveyN2AI project and their respective IAM roles.

## Service Accounts Created

### 1. KonveyN2AI Vertex AI Service Account
- **Email**: `konveyn2ai-vertex-ai@konveyn2ai.iam.gserviceaccount.com`
- **Display Name**: KonveyN2AI Vertex AI Service Account
- **Purpose**: Handles all Vertex AI operations including vector index management and AI model interactions
- **IAM Roles**:
  - `roles/aiplatform.user` - Vertex AI User role for accessing AI Platform services

### 2. KonveyN2AI Cloud Run Service Account
- **Email**: `konveyn2ai-cloud-run@konveyn2ai.iam.gserviceaccount.com`
- **Display Name**: KonveyN2AI Cloud Run Service Account
- **Purpose**: Manages Cloud Run deployments and container registry operations
- **IAM Roles**:
  - `roles/run.admin` - Cloud Run Admin role for managing Cloud Run services
  - `roles/artifactregistry.writer` - Artifact Registry Writer role for container image management

### 3. Default Compute Service Account (Pre-existing)
- **Email**: `72021522495-compute@developer.gserviceaccount.com`
- **Display Name**: Default compute service account
- **Purpose**: Default GCP compute service account
- **IAM Roles**:
  - `roles/editor` - Editor role (standard for default compute service account)

## Security Best Practices

1. **Principle of Least Privilege**: Each service account has only the minimum permissions required for its specific function
2. **Separate Concerns**: Different service accounts for different services (Vertex AI vs Cloud Run)
3. **No User Account Keys**: Service accounts use workload identity where possible

## Usage in KonveyN2AI Architecture

- **Amatya Role Prompter**: Uses Vertex AI service account for AI model interactions
- **Janapada Memory**: Uses Vertex AI service account for vector operations and embeddings
- **Svami Orchestrator**: Uses Cloud Run service account for deployment and orchestration
- **All Components**: Leverage these service accounts for secure Google Cloud API access

## Testing Service Account Permissions

To verify service account permissions are correctly configured:

```bash
# Test Vertex AI permissions
gcloud ai operations list --project=konveyn2ai

# Test Cloud Run permissions
gcloud run services list --project=konveyn2ai --region=us-central1

# Test Artifact Registry permissions
gcloud artifacts repositories list --project=konveyn2ai --location=us-central1
```

## Next Steps

1. Generate and download service account keys for local development (Task 2.4)
2. Configure Workload Identity for production environments (Task 2.3)
3. Integrate service accounts into the three-component architecture

## Project Configuration

- **Project ID**: konveyn2ai
- **Project Number**: 72021522495
- **Region**: us-central1
- **Created**: 2025-07-25

---

**Note**: This configuration supports the hackathon requirement for real functionality with proper Google Cloud integration.
