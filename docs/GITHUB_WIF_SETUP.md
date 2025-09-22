# GitHub Actions with Google Cloud ADC Setup Guide

This guide explains how to configure GitHub Actions to authenticate with Google Cloud services using Workload Identity Federation (WIF), similar to local ADC authentication.

## Overview

Workload Identity Federation allows GitHub Actions to authenticate with Google Cloud without storing service account keys. This is the recommended secure approach that works similarly to your local ADC setup.

## Benefits

- **No service account keys to manage**: No need to rotate or secure JSON key files
- **Automatic authentication**: Works like ADC - the code doesn't need to change
- **Secure by default**: Uses short-lived tokens instead of long-lived keys
- **Repository-scoped**: Only your repository can use the authentication

## Setup Instructions

### 1. Prerequisites

- Google Cloud CLI (`gcloud`) installed locally
- Authenticated with Google Cloud: `gcloud auth login`
- Project permissions to create service accounts and IAM bindings

### 2. Run the Setup Script

```bash
# Set your GitHub username/organization
export GITHUB_ORG=your-github-username
export GITHUB_REPO=konveyn2ai_bigquery

# Make the script executable
chmod +x setup-github-wif.sh

# Run the setup script
./setup-github-wif.sh
```

The script will:
1. Enable required Google Cloud APIs
2. Create a service account for GitHub Actions
3. Grant BigQuery and Vertex AI permissions
4. Create a Workload Identity Pool and Provider
5. Configure service account impersonation

### 3. Add GitHub Secrets

After running the script, you'll see output like:
```
WIF_PROVIDER: projects/123456/locations/global/workloadIdentityPools/github-pool/providers/github-provider
WIF_SERVICE_ACCOUNT: github-actions-sa@konveyn2ai.iam.gserviceaccount.com
```

Add these as GitHub repository secrets:
1. Go to your repository settings: `https://github.com/YOUR_ORG/konveyn2ai_bigquery/settings/secrets/actions`
2. Click "New repository secret"
3. Add `WIF_PROVIDER` with the provider value
4. Add `WIF_SERVICE_ACCOUNT` with the service account email

### 4. How It Works in GitHub Actions

The GitHub workflows have been updated to use WIF authentication:

```yaml
jobs:
  test:
    permissions:
      contents: 'read'
      id-token: 'write'  # Required for OIDC token generation

    steps:
    - name: Authenticate to Google Cloud
      uses: 'google-github-actions/auth@v2'
      with:
        workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
        service_account: ${{ secrets.WIF_SERVICE_ACCOUNT }}
        project_id: 'konveyn2ai'

    - name: Run tests
      env:
        GOOGLE_CLOUD_PROJECT: 'konveyn2ai'
        BIGQUERY_INGESTION_DATASET_ID: 'source_ingestion'
      run: |
        # Your tests run here with automatic authentication
        python -m pytest tests/
```

## Code Compatibility

The Python code remains unchanged. It uses the same authentication pattern as locally:

```python
from google.auth import default
from google.cloud import bigquery

# This works both locally with ADC and in GitHub Actions with WIF
credentials, project = default()
client = bigquery.Client(credentials=credentials, project=project)
```

## Environment Variables

The following environment variables are automatically set in GitHub Actions:
- `GOOGLE_CLOUD_PROJECT`: konveyn2ai
- `BIGQUERY_INGESTION_DATASET_ID`: source_ingestion
- ADC is automatically configured by the auth action

## Security Notes

- **Repository-scoped**: Only workflows from your repository can use these credentials
- **Short-lived tokens**: Tokens expire after 1 hour
- **No keys to rotate**: Unlike service account keys, there's nothing to rotate
- **Audit trail**: All authentication is logged in Google Cloud audit logs

## Troubleshooting

### Permission Denied Errors
If you see permission errors, verify:
1. The service account has the required roles (BigQuery Data Editor, Job User)
2. The WIF configuration includes your repository path correctly
3. The secrets are properly set in GitHub

### Authentication Failures
Check that:
1. The `id-token: write` permission is set in the workflow
2. The auth action is using v2: `google-github-actions/auth@v2`
3. The WIF provider and service account secrets are correctly formatted

### Local Development
For local development, continue using:
```bash
gcloud auth application-default login
```

This creates the same ADC environment that the GitHub Actions workflows use.

## Additional Resources

- [Google Cloud Workload Identity Federation](https://cloud.google.com/iam/docs/workload-identity-federation)
- [GitHub OIDC tokens](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [google-github-actions/auth](https://github.com/google-github-actions/auth)