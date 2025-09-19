#!/bin/bash
set -euo pipefail

# Setup GitHub Secrets for Google Cloud Deployment with Workload Identity
# This script configures ADC-based authentication for GitHub Actions

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-konveyn2ai}"
POOL_ID="github-actions-pool"
PROVIDER_ID="github-actions-provider"
SA_NAME="github-deployment"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
REPO="${GITHUB_REPOSITORY:-sdamache/konveyn2ai_bigquery}"

echo "Setting up GitHub secrets for Google Cloud deployment..."
echo "Project: $PROJECT_ID"
echo "Repository: $REPO"

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "Error: No active gcloud authentication found. Please run 'gcloud auth login'"
    exit 1
fi

# Check if gh is authenticated
if ! gh auth status > /dev/null 2>&1; then
    echo "Error: GitHub CLI not authenticated. Please run 'gh auth login'"
    exit 1
fi

# Enable necessary APIs
echo "Enabling required APIs..."
gcloud services enable iam.googleapis.com \
    iamcredentials.googleapis.com \
    sts.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    bigquery.googleapis.com \
    --project=$PROJECT_ID

# Create Workload Identity Pool if it doesn't exist
echo "Setting up Workload Identity Pool..."
if ! gcloud iam workload-identity-pools describe $POOL_ID \
    --location=global --project=$PROJECT_ID > /dev/null 2>&1; then
    
    gcloud iam workload-identity-pools create $POOL_ID \
        --location=global \
        --display-name="GitHub Actions Pool" \
        --description="Workload Identity Pool for GitHub Actions" \
        --project=$PROJECT_ID
fi

# Create Workload Identity Provider if it doesn't exist
echo "Setting up Workload Identity Provider..."
if ! gcloud iam workload-identity-pools providers describe $PROVIDER_ID \
    --location=global --workload-identity-pool=$POOL_ID \
    --project=$PROJECT_ID > /dev/null 2>&1; then
    
    gcloud iam workload-identity-pools providers create-oidc $PROVIDER_ID \
        --location=global \
        --workload-identity-pool=$POOL_ID \
        --display-name="GitHub Actions Provider" \
        --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.actor=assertion.actor,attribute.aud=assertion.aud" \
        --issuer-uri="https://token.actions.githubusercontent.com" \
        --project=$PROJECT_ID
fi

# Create service account if it doesn't exist
echo "Setting up service account..."
if ! gcloud iam service-accounts describe $SA_EMAIL --project=$PROJECT_ID > /dev/null 2>&1; then
    gcloud iam service-accounts create $SA_NAME \
        --display-name="GitHub Deployment Service Account" \
        --description="Service account for GitHub Actions deployments" \
        --project=$PROJECT_ID
fi

# Grant necessary roles to service account
echo "Granting IAM roles to service account..."
ROLES=(
    "roles/run.admin"
    "roles/iam.serviceAccountUser"
    "roles/cloudsql.client"
    "roles/bigquery.admin"
    "roles/storage.admin"
    "roles/logging.logWriter"
    "roles/monitoring.metricWriter"
)

for role in "${ROLES[@]}"; do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SA_EMAIL" \
        --role="$role" \
        --condition=None
done

# Allow the GitHub repository to impersonate the service account
echo "Configuring Workload Identity binding..."
gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL \
    --role="roles/iam.workloadIdentityUser" \
    --member="principalSet://iam.googleapis.com/projects/$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')/locations/global/workloadIdentityPools/$POOL_ID/attribute.repository/$REPO" \
    --project=$PROJECT_ID

# Get the Workload Identity Provider resource name
WIF_PROVIDER="projects/$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')/locations/global/workloadIdentityPools/$POOL_ID/providers/$PROVIDER_ID"

# Set GitHub secrets
echo "Setting GitHub repository secrets..."

# Required secrets for Workload Identity
gh secret set WIF_PROVIDER --body "$WIF_PROVIDER" --repo "$REPO"
gh secret set WIF_SERVICE_ACCOUNT --body "$SA_EMAIL" --repo "$REPO"

# Additional secrets (if they don't exist, set to placeholder values)
if ! gh secret list --repo "$REPO" | grep -q "GOOGLE_API_KEY"; then
    echo "Warning: GOOGLE_API_KEY not set. Please set this manually with your Google API key."
    gh secret set GOOGLE_API_KEY --body "YOUR_GOOGLE_API_KEY_HERE" --repo "$REPO"
fi

if ! gh secret list --repo "$REPO" | grep -q "INDEX_ID"; then
    echo "Warning: INDEX_ID not set. Please set this manually if you have a vector index."
    gh secret set INDEX_ID --body "YOUR_INDEX_ID_HERE" --repo "$REPO"
fi

# Set DEMO_TOKEN as a variable (not secret)
if ! gh variable list --repo "$REPO" | grep -q "DEMO_TOKEN"; then
    gh variable set DEMO_TOKEN --body "demo-access-token" --repo "$REPO"
fi

echo ""
echo "✅ GitHub secrets setup complete!"
echo ""
echo "Workload Identity Provider: $WIF_PROVIDER"
echo "Service Account: $SA_EMAIL"
echo ""
echo "Next steps:"
echo "1. Update GOOGLE_API_KEY secret with your actual Google API key"
echo "2. Update INDEX_ID secret if you have a vector index ID"
echo "3. Test the deployment by pushing to main branch"
echo ""
echo "To verify setup:"
echo "  gh secret list --repo $REPO"
echo "  gh variable list --repo $REPO"