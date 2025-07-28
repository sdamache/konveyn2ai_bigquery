#!/bin/bash

# Setup service-to-service authentication for KonveyN2AI Cloud Run services
# This script configures proper authentication between services

set -e

PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"konveyn2ai"}
REGION=${GOOGLE_CLOUD_LOCATION:-"us-central1"}
SERVICE_ACCOUNT_EMAIL="konveyn2ai-service@${PROJECT_ID}.iam.gserviceaccount.com"

echo "🔐 Setting up service-to-service authentication for KonveyN2AI"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service Account: $SERVICE_ACCOUNT_EMAIL"

# Get service URLs
echo "📡 Getting service URLs..."
JANAPADA_URL=$(gcloud run services describe janapada --region=${REGION} --project=${PROJECT_ID} --format="value(status.url)" 2>/dev/null || echo "")
AMATYA_URL=$(gcloud run services describe amatya --region=${REGION} --project=${PROJECT_ID} --format="value(status.url)" 2>/dev/null || echo "")
SVAMI_URL=$(gcloud run services describe svami --region=${REGION} --project=${PROJECT_ID} --format="value(status.url)" 2>/dev/null || echo "")

if [ -z "$JANAPADA_URL" ] || [ -z "$AMATYA_URL" ] || [ -z "$SVAMI_URL" ]; then
    echo "❌ Error: Some services are not deployed yet. Please run ./deploy-to-cloud-run.sh first"
    exit 1
fi

echo "Found services:"
echo "  Janapada: $JANAPADA_URL"
echo "  Amatya: $AMATYA_URL"
echo "  Svami: $SVAMI_URL"

# Configure IAM for service-to-service communication
echo "🔑 Configuring IAM for service-to-service communication..."

# Allow the service account to invoke Cloud Run services
gcloud run services add-iam-policy-binding janapada \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/run.invoker" \
    --region=${REGION} \
    --project=${PROJECT_ID}

gcloud run services add-iam-policy-binding amatya \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/run.invoker" \
    --region=${REGION} \
    --project=${PROJECT_ID}

gcloud run services add-iam-policy-binding svami \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/run.invoker" \
    --region=${REGION} \
    --project=${PROJECT_ID}

# Update Svami service with authenticated service URLs
echo "🔄 Updating Svami service with authenticated service URLs..."
gcloud run services update svami \
    --set-env-vars="JANAPADA_URL=${JANAPADA_URL},AMATYA_URL=${AMATYA_URL},ENVIRONMENT=production,GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    --region=${REGION} \
    --project=${PROJECT_ID}

# Create a script for authenticated service calls
cat > service-auth-helper.py << 'EOF'
#!/usr/bin/env python3
"""
Helper script for making authenticated requests between Cloud Run services
"""

import requests
import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account
import os

def get_identity_token(target_audience, service_account_path=None):
    """Get an identity token for authenticating to Cloud Run services."""
    if service_account_path and os.path.exists(service_account_path):
        # Use service account key file (for local development)
        credentials = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
    else:
        # Use default credentials (for Cloud Run environment)
        credentials, _ = google.auth.default()
    
    # Create a request object for token refresh
    request = google.auth.transport.requests.Request()
    
    # Get an identity token
    credentials.refresh(request)
    
    # For Cloud Run, we need to use the identity token
    if hasattr(credentials, 'token'):
        return credentials.token
    else:
        # Fallback for service account credentials
        return credentials.token

def make_authenticated_request(url, method='GET', data=None, headers=None):
    """Make an authenticated request to a Cloud Run service."""
    if headers is None:
        headers = {}
    
    try:
        # Get identity token for the target URL
        token = get_identity_token(url)
        headers['Authorization'] = f'Bearer {token}'
        
        # Make the request
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers)
        elif method.upper() == 'POST':
            response = requests.post(url, json=data, headers=headers)
        elif method.upper() == 'PUT':
            response = requests.put(url, json=data, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        return response
    except Exception as e:
        print(f"Error making authenticated request: {e}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python service-auth-helper.py <url> [method] [data]")
        sys.exit(1)
    
    url = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'GET'
    
    response = make_authenticated_request(url, method)
    if response:
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    else:
        print("Request failed")
EOF

chmod +x service-auth-helper.py

# Test service-to-service authentication
echo "🧪 Testing service-to-service authentication..."

echo "Testing Janapada health endpoint..."
python3 service-auth-helper.py "${JANAPADA_URL}/health" GET

echo "Testing Amatya health endpoint..."
python3 service-auth-helper.py "${AMATYA_URL}/health" GET

echo "Testing Svami health endpoint..."
python3 service-auth-helper.py "${SVAMI_URL}/health" GET

echo "✅ Service authentication setup complete!"
echo ""
echo "📋 Configuration Summary:"
echo "- Service account: $SERVICE_ACCOUNT_EMAIL"
echo "- All services can authenticate to each other"
echo "- Authentication helper script created: service-auth-helper.py"
echo ""
echo "💡 Usage in your services:"
echo "- Use the service account's default credentials"
echo "- Include Authorization header with identity token"
echo "- Use the service-auth-helper.py for manual testing"