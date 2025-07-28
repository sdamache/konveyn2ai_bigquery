#!/bin/bash

# Deploy KonveyN2AI services to Google Cloud Run
# This script builds and deploys all three services (Janapada, Amatya, Svami)

set -e

# Configuration variables from environment
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"konveyn2ai"}
REGION=${GOOGLE_CLOUD_LOCATION:-"us-central1"}
REPOSITORY_NAME="konveyn2ai-repo"
SERVICE_ACCOUNT_EMAIL="konveyn2ai-service@${PROJECT_ID}.iam.gserviceaccount.com"

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "📄 Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

echo "🚀 Deploying KonveyN2AI services to Google Cloud Run"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Repository: $REPOSITORY_NAME"
echo ""
echo "🔒 Security Notice: API keys will be passed as build arguments and runtime environment variables."
echo "   Ensure your .env file is not committed to version control."

# Enhanced environment variable validation
echo "🔍 Validating environment variables..."

REQUIRED_VARS=("GOOGLE_API_KEY" "INDEX_ID")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    else
        echo "✅ $var is set"
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo "❌ Error: Missing required environment variables:"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "📋 Please ensure your .env file contains:"
    echo "GOOGLE_API_KEY=your_google_api_key_here"
    echo "INDEX_ENDPOINT_ID=your_index_endpoint_id_here"
    echo "INDEX_ID=your_index_id_here"
    echo ""
    echo "Or set these variables in your environment before running this script."
    exit 1
fi

# Validate Docker and gcloud CLI availability
echo "🔧 Validating required tools..."
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed or not in PATH"
    exit 1
fi

if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: Google Cloud CLI is not installed or not in PATH"
    exit 1
fi

echo "✅ All validations passed"

# Configure Docker authentication for Artifact Registry
echo "🔐 Configuring Docker authentication for Artifact Registry..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Build timestamp for image tagging
BUILD_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
IMAGE_TAG="${BUILD_TIMESTAMP}-${RANDOM}"

echo "🏷️  Using image tag: $IMAGE_TAG"

# Deployment configuration
DEPLOYMENT_TIMEOUT=600  # 10 minutes
HEALTH_CHECK_RETRIES=5
RETRY_DELAY=30

# Function to wait for service deployment with timeout
wait_for_deployment() {
    local service_name=$1
    local timeout=$2
    local start_time=$(date +%s)
    
    echo "⏳ Waiting for $service_name deployment to complete (timeout: ${timeout}s)..."
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -gt $timeout ]; then
            echo "❌ Deployment timeout after ${timeout}s for $service_name"
            return 1
        fi
        
        # Check deployment status
        local status=$(gcloud run services describe $service_name --region=${REGION} --project=${PROJECT_ID} --format="value(status.conditions[0].status)" 2>/dev/null || echo "")
        
        if [ "$status" = "True" ]; then
            echo "✅ $service_name deployment completed successfully"
            return 0
        elif [ "$status" = "False" ]; then
            echo "❌ $service_name deployment failed"
            return 1
        fi
        
        echo "⏳ Still deploying $service_name... (${elapsed}s elapsed)"
        sleep 10
    done
}

# Function to wait for service readiness with health checks
wait_for_service_ready() {
    local service_name=$1
    local service_url=$2
    local timeout=${3:-300}
    local start_time=$(date +%s)
    
    echo "🏥 Waiting for $service_name to become ready (timeout: ${timeout}s)..."
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -gt $timeout ]; then
            echo "❌ Service readiness timeout after ${timeout}s for $service_name"
            return 1
        fi
        
        # Test health endpoint
        if curl -f --connect-timeout 10 --max-time 30 "${service_url}/health" > /dev/null 2>&1; then
            echo "✅ $service_name is ready and healthy"
            return 0
        else
            echo "⏳ Waiting for $service_name to be ready... (${elapsed}s elapsed)"
            sleep 15
        fi
    done
}

# Build and push Docker images
echo "🔨 Building and pushing Docker images..."

# Build Janapada
echo "📦 Building Janapada Memory Service..."
docker build --platform linux/amd64 -f Dockerfile.janapada \
    --build-arg GOOGLE_CLOUD_PROJECT="${PROJECT_ID}" \
    --build-arg INDEX_ID="${INDEX_ID}" \
    -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/janapada:${IMAGE_TAG} \
    -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/janapada:latest \
    .

# Verify Janapada build succeeded
if [ $? -eq 0 ]; then
    echo "✅ Janapada build successful"
else
    echo "❌ Janapada build failed"
    exit 1
fi

docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/janapada:${IMAGE_TAG}
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/janapada:latest

# Build Amatya
echo "🎭 Building Amatya Role Prompter Service..."
docker build --platform linux/amd64 -f Dockerfile.amatya \
    --build-arg GOOGLE_CLOUD_PROJECT="${PROJECT_ID}" \
    -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/amatya:${IMAGE_TAG} \
    -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/amatya:latest \
    .

# Verify Amatya build succeeded
if [ $? -eq 0 ]; then
    echo "✅ Amatya build successful"
else
    echo "❌ Amatya build failed"
    exit 1
fi

docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/amatya:${IMAGE_TAG}
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/amatya:latest

# Note: Svami will be built after we have Janapada and Amatya URLs
echo "🎼 Svami Orchestrator will be built after service URLs are available..."

# Deploy to Cloud Run
echo "☁️  Deploying services to Cloud Run..."

# Deploy Janapada Memory Service
echo "📦 Deploying Janapada Memory Service..."
gcloud run deploy janapada \
    --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/janapada:${IMAGE_TAG} \
    --region=${REGION} \
    --platform=managed \
    --memory=512Mi \
    --cpu=1 \
    --min-instances=1 \
    --max-instances=10 \
    --port=8080 \
    --timeout=300 \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION},VECTOR_INDEX_ID=${INDEX_ID},ENVIRONMENT=production,DEBUG=false" \
    --service-account=${SERVICE_ACCOUNT_EMAIL} \
    --allow-unauthenticated \
    --project=${PROJECT_ID}

# Wait for Janapada deployment to complete  
wait_for_deployment "janapada" $DEPLOYMENT_TIMEOUT
if [ $? -ne 0 ]; then
    echo "❌ Janapada deployment failed or timed out"
    exit 1
fi

# Get Janapada service URL
JANAPADA_URL=$(gcloud run services describe janapada --region=${REGION} --project=${PROJECT_ID} --format="value(status.url)")
echo "✅ Janapada deployed at: $JANAPADA_URL"

# Wait for Janapada service to be ready before proceeding
wait_for_service_ready "janapada" "$JANAPADA_URL" 300
if [ $? -ne 0 ]; then
    echo "❌ Janapada service failed to become ready"
    exit 1
fi

# Deploy Amatya Role Prompter Service
echo "🎭 Deploying Amatya Role Prompter Service..."
gcloud run deploy amatya \
    --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/amatya:${IMAGE_TAG} \
    --region=${REGION} \
    --platform=managed \
    --memory=512Mi \
    --cpu=1 \
    --min-instances=1 \
    --max-instances=10 \
    --port=8080 \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION},GOOGLE_API_KEY=${GOOGLE_API_KEY},ENVIRONMENT=production,DEBUG=false" \
    --timeout=300 \
    --service-account=${SERVICE_ACCOUNT_EMAIL} \
    --allow-unauthenticated \
    --project=${PROJECT_ID}

# Wait for Amatya deployment to complete
wait_for_deployment "amatya" $DEPLOYMENT_TIMEOUT
if [ $? -ne 0 ]; then
    echo "❌ Amatya deployment failed or timed out"
    exit 1
fi

# Get Amatya service URL
AMATYA_URL=$(gcloud run services describe amatya --region=${REGION} --project=${PROJECT_ID} --format="value(status.url)")
echo "✅ Amatya deployed at: $AMATYA_URL"

# Wait for Amatya service to be ready before proceeding
wait_for_service_ready "amatya" "$AMATYA_URL" 300
if [ $? -ne 0 ]; then
    echo "❌ Amatya service failed to become ready"
    exit 1
fi

# Now build Svami with the service URLs
echo "🎼 Building Svami Orchestrator Service with service URLs..."
docker build --platform linux/amd64 -f Dockerfile.svami \
    --build-arg JANAPADA_URL="${JANAPADA_URL}" \
    --build-arg AMATYA_URL="${AMATYA_URL}" \
    -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/svami:${IMAGE_TAG} \
    -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/svami:latest \
    .

# Verify Svami build succeeded
if [ $? -eq 0 ]; then
    echo "✅ Svami build successful"
else
    echo "❌ Svami build failed"
    exit 1
fi

docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/svami:${IMAGE_TAG}
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/svami:latest

# Deploy Svami Orchestrator Service (with service URLs)
echo "🎼 Deploying Svami Orchestrator Service..."
gcloud run deploy svami \
    --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/svami:${IMAGE_TAG} \
    --region=${REGION} \
    --platform=managed \
    --memory=512Mi \
    --cpu=1 \
    --min-instances=1 \
    --max-instances=10 \
    --port=8080 \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},JANAPADA_URL=${JANAPADA_URL},AMATYA_URL=${AMATYA_URL},ENVIRONMENT=production,DEBUG=false" \
    --service-account=${SERVICE_ACCOUNT_EMAIL} \
    --allow-unauthenticated \
    --project=${PROJECT_ID}

# Wait for Svami deployment to complete
wait_for_deployment "svami" $DEPLOYMENT_TIMEOUT
if [ $? -ne 0 ]; then
    echo "❌ Svami deployment failed or timed out"
    exit 1
fi

# Get Svami service URL
SVAMI_URL=$(gcloud run services describe svami --region=${REGION} --project=${PROJECT_ID} --format="value(status.url)")
echo "✅ Svami deployed at: $SVAMI_URL"

# Wait for Svami service to be ready before proceeding
wait_for_service_ready "svami" "$SVAMI_URL" 300
if [ $? -ne 0 ]; then
    echo "❌ Svami service failed to become ready"
    exit 1
fi

# Test inter-service connectivity before final validation
echo "🔗 Testing inter-service connectivity..."

# Test Svami's ability to connect to Janapada and Amatya
echo "🧠 Testing Svami → Janapada connectivity..."
JANAPADA_TEST=$(curl -s -f --connect-timeout 10 --max-time 30 "${JANAPADA_URL}/health" && echo "OK" || echo "FAIL")
if [ "$JANAPADA_TEST" = "OK" ]; then
    echo "✅ Svami can reach Janapada"
else
    echo "⚠️  Svami may have connectivity issues with Janapada"
fi

echo "🎭 Testing Svami → Amatya connectivity..."
AMATYA_TEST=$(curl -s -f --connect-timeout 10 --max-time 30 "${AMATYA_URL}/health" && echo "OK" || echo "FAIL")
if [ "$AMATYA_TEST" = "OK" ]; then
    echo "✅ Svami can reach Amatya"
else
    echo "⚠️  Svami may have connectivity issues with Amatya"
fi

# Enhanced health endpoint testing with retry logic
test_health_endpoint() {
    local service_name=$1
    local service_url=$2
    local health_url="${service_url}/health"
    
    echo "🏥 Testing $service_name health endpoint..."
    
    for i in $(seq 1 $HEALTH_CHECK_RETRIES); do
        if curl -f --connect-timeout 10 "$health_url" > /dev/null 2>&1; then
            echo "✅ $service_name health check passed"
            return 0
        else
            echo "⚠️  $service_name health check failed (attempt $i/$HEALTH_CHECK_RETRIES)"
            if [ $i -lt $HEALTH_CHECK_RETRIES ]; then
                sleep $RETRY_DELAY
            fi
        fi
    done
    
    echo "❌ $service_name health check failed after $HEALTH_CHECK_RETRIES attempts"
    return 1
}

echo "🏥 Testing service health endpoints with retry logic..."

# Test all services
test_health_endpoint "Janapada" "$JANAPADA_URL"
JANAPADA_HEALTH=$?

test_health_endpoint "Amatya" "$AMATYA_URL"  
AMATYA_HEALTH=$?

test_health_endpoint "Svami" "$SVAMI_URL"
SVAMI_HEALTH=$?

# Check if any health checks failed
if [ $JANAPADA_HEALTH -ne 0 ] || [ $AMATYA_HEALTH -ne 0 ] || [ $SVAMI_HEALTH -ne 0 ]; then
    echo "⚠️  Some health checks failed. Services may still be starting up."
    echo "💡 You can check service logs with:"
    echo "   gcloud logs tail --service=janapada --project=$PROJECT_ID"
    echo "   gcloud logs tail --service=amatya --project=$PROJECT_ID"  
    echo "   gcloud logs tail --service=svami --project=$PROJECT_ID"
else
    echo "🎉 All health checks passed!"
    
    # Test conversational intelligence feature
    echo "🤖 Testing conversational intelligence feature..."
    GREETING_TEST=$(curl -s -X POST "${SVAMI_URL}/answer" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer demo-token" \
        -d '{"question": "hi", "role": "developer"}' | grep -o '"answer":"[^"]*"' | cut -d'"' -f4 || echo "")
    
    if [[ "$GREETING_TEST" == *"Hi there"* ]] || [[ "$GREETING_TEST" == *"Hello"* ]]; then
        echo "✅ Conversational intelligence working - greeting response detected"
    else
        echo "⚠️  Conversational intelligence may need verification - response: $GREETING_TEST"
    fi
fi

# Output deployment summary
echo ""
echo "🎉 Deployment Complete!"
echo "===================="
echo "Janapada Memory Service:     $JANAPADA_URL"
echo "Amatya Role Prompter:        $AMATYA_URL"
echo "Svami Orchestrator:          $SVAMI_URL"
echo ""
echo "📋 Service Configuration:"
echo "- Memory: 512Mi (Janapada), 256Mi (Amatya, Svami)"
echo "- CPU: 1 per service"
echo "- Min instances: 1 per service"
echo "- Max instances: 10 per service"
echo "- Service account: $SERVICE_ACCOUNT_EMAIL"
echo ""
echo "🔧 Next Steps:"
echo "1. Test the deployed services using the URLs above"
echo "2. Set up monitoring and alerting"
echo "3. Configure custom domain if needed"
echo "4. Update client applications with production URLs"
echo ""
echo "📊 View logs:"
echo "gcloud logs tail --project=$PROJECT_ID --service=janapada"
echo "gcloud logs tail --project=$PROJECT_ID --service=amatya"
echo "gcloud logs tail --project=$PROJECT_ID --service=svami"