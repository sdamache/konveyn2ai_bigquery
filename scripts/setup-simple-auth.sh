#!/bin/bash
set -euo pipefail

# Simple GitHub Secrets Setup using Service Account Key
# Alternative approach if Workload Identity setup is too complex

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-konveyn2ai}"
SA_NAME="github-deployment"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
REPO="${GITHUB_REPOSITORY:-sdamache/konveyn2ai_bigquery}"
KEY_FILE="/tmp/github-sa-key.json"

echo "Setting up GitHub secrets with service account key..."
echo "Project: $PROJECT_ID"
echo "Repository: $REPO"

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "Error: No active gcloud authentication. Please run 'gcloud auth login'"
    exit 1
fi

if ! gh auth status > /dev/null 2>&1; then
    echo "Error: GitHub CLI not authenticated. Please run 'gh auth login'"
    exit 1
fi

# Enable APIs
echo "Enabling required APIs..."
gcloud services enable iam.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    bigquery.googleapis.com \
    --project=$PROJECT_ID

# Create service account
echo "Creating service account..."
if ! gcloud iam service-accounts describe $SA_EMAIL --project=$PROJECT_ID > /dev/null 2>&1; then
    gcloud iam service-accounts create $SA_NAME \
        --display-name="GitHub Deployment SA" \
        --project=$PROJECT_ID
fi

# Grant roles
echo "Granting IAM roles..."
ROLES=(
    "roles/run.admin"
    "roles/iam.serviceAccountUser"
    "roles/bigquery.admin"
    "roles/storage.admin"
)

for role in "${ROLES[@]}"; do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SA_EMAIL" \
        --role="$role" \
        --condition=None
done

# Create and download service account key
echo "Creating service account key..."
gcloud iam service-accounts keys create $KEY_FILE \
    --iam-account=$SA_EMAIL \
    --project=$PROJECT_ID

# Set GitHub secret
echo "Setting GOOGLE_CLOUD_CREDENTIALS secret..."
gh secret set GOOGLE_CLOUD_CREDENTIALS --body "$(cat $KEY_FILE)" --repo "$REPO"

# Set other secrets with placeholder values
if ! gh secret list --repo "$REPO" | grep -q "GOOGLE_API_KEY"; then
    gh secret set GOOGLE_API_KEY --body "YOUR_API_KEY_HERE" --repo "$REPO"
fi

if ! gh secret list --repo "$REPO" | grep -q "INDEX_ID"; then
    gh secret set INDEX_ID --body "YOUR_INDEX_ID_HERE" --repo "$REPO"
fi

# Clean up key file
rm -f $KEY_FILE

echo ""
echo "✅ GitHub secrets setup complete!"
echo "Service Account: $SA_EMAIL"
echo ""
echo "Next steps:"
echo "1. Update GOOGLE_API_KEY with your actual API key"
echo "2. Update INDEX_ID if needed"
echo "3. Test deployment by pushing to main"