#!/bin/bash

# Setup Google Artifact Registry for KonveyN2AI
# This script creates the artifact registry repository for storing container images

set -e

# Configuration variables
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"konveyn2ai"}
REGION=${GOOGLE_CLOUD_LOCATION:-"us-central1"}
REPOSITORY_NAME="konveyn2ai-repo"

echo "🚀 Setting up Artifact Registry for KonveyN2AI"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Repository: $REPOSITORY_NAME"

# Enable required APIs
echo "📡 Enabling required Google Cloud APIs..."
gcloud services enable \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    --project=$PROJECT_ID

# Create Artifact Registry repository
echo "📦 Creating Artifact Registry repository..."
gcloud artifacts repositories create $REPOSITORY_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="Container images for KonveyN2AI services" \
    --project=$PROJECT_ID || echo "Repository may already exist"

# Configure Docker authentication
echo "🔐 Configuring Docker authentication for Artifact Registry..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Create service account for Cloud Run services
SERVICE_ACCOUNT_NAME="konveyn2ai-service"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "👤 Creating service account for Cloud Run services..."
gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
    --display-name="KonveyN2AI Service Account" \
    --description="Service account for KonveyN2AI Cloud Run services" \
    --project=$PROJECT_ID || echo "Service account may already exist"

# Grant necessary IAM roles to the service account
echo "🔑 Granting IAM roles to service account..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/artifactregistry.reader"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/monitoring.metricWriter"

# Create Cloud Build trigger (optional - can be done manually in console)
echo "🔧 Setting up Cloud Build trigger..."
cat > cloudbuild-trigger.yaml << EOF
name: konveyn2ai-deploy-trigger
description: Deploy KonveyN2AI services to Cloud Run
github:
  owner: neeharve
  name: KonveyN2AI
  push:
    branch: ^main$
filename: cloudbuild.yaml
substitutions:
  _ARTIFACT_REGISTRY_REGION: $REGION
  _REPOSITORY: $REPOSITORY_NAME
  _CLOUD_RUN_REGION: $REGION
  _SERVICE_ACCOUNT_EMAIL: $SERVICE_ACCOUNT_EMAIL
  _INDEX_ENDPOINT_ID: ${INDEX_ENDPOINT_ID}
  _INDEX_ID: ${INDEX_ID}
  _GOOGLE_API_KEY: ${GOOGLE_API_KEY}
  _CLOUD_RUN_HASH: uc
EOF

# Output important information
echo "✅ Artifact Registry setup complete!"
echo ""
echo "📋 Important Information:"
echo "Repository URL: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}"
echo "Service Account: ${SERVICE_ACCOUNT_EMAIL}"
echo ""
echo "🎯 Next Steps:"
echo "1. Set up Cloud Build trigger using the generated cloudbuild-trigger.yaml"
echo "2. Run the deployment script: ./deploy-to-cloud-run.sh"
echo "3. Test the deployed services"
echo ""
echo "💡 Manual Cloud Build trigger creation:"
echo "gcloud builds triggers create github --repo-name=KonveyN2AI --repo-owner=neeharve --branch-pattern=^main$ --build-config=cloudbuild.yaml --project=$PROJECT_ID"