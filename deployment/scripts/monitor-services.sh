#\!/bin/bash

# Monitor KonveyN2AI services health and metrics
# Usage: ./monitor-services.sh [check|logs|metrics|test]

PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"konveyn2ai"}
REGION=${GOOGLE_CLOUD_LOCATION:-"us-central1"}

function check_health() {
    echo "🏥 Checking service health..."
    
    JANAPADA_URL=$(gcloud run services describe janapada --region=${REGION} --project=${PROJECT_ID} --format="value(status.url)" 2>/dev/null || echo "")
    AMATYA_URL=$(gcloud run services describe amatya --region=${REGION} --project=${PROJECT_ID} --format="value(status.url)" 2>/dev/null || echo "")
    SVAMI_URL=$(gcloud run services describe svami --region=${REGION} --project=${PROJECT_ID} --format="value(status.url)" 2>/dev/null || echo "")
    
    for service in "Janapada:$JANAPADA_URL" "Amatya:$AMATYA_URL" "Svami:$SVAMI_URL"; do
        name=$(echo $service | cut -d: -f1)
        url=$(echo $service | cut -d: -f2-)
        
        if [ -n "$url" ]; then
            printf "  %s: " "$name"
            if curl -s -f "${url}/health" > /dev/null; then
                echo "✅ Healthy"
            else
                echo "❌ Unhealthy"
            fi
        else
            echo "  $name: ⚠️  Not deployed"
        fi
    done
}

function view_logs() {
    echo "📋 Recent logs from all services..."
    gcloud logging read "resource.type=cloud_run_revision AND (resource.labels.service_name=janapada OR resource.labels.service_name=amatya OR resource.labels.service_name=svami)" --limit=50 --format="table(timestamp,resource.labels.service_name,severity,textPayload)" --project=$PROJECT_ID
}

function view_metrics() {
    echo "📊 Service metrics (last 1 hour)..."
    echo "Request counts:"
    gcloud monitoring metrics list --filter="metric.type:run.googleapis.com/request_count" --project=$PROJECT_ID
    
    echo "Memory utilization:"
    gcloud monitoring metrics list --filter="metric.type:run.googleapis.com/container/memory/utilizations" --project=$PROJECT_ID
}

function test_integration() {
    echo "🔗 Testing inter-service communication..."
    
    JANAPADA_URL=$(gcloud run services describe janapada --region=${REGION} --project=${PROJECT_ID} --format="value(status.url)" 2>/dev/null || echo "")
    AMATYA_URL=$(gcloud run services describe amatya --region=${REGION} --project=${PROJECT_ID} --format="value(status.url)" 2>/dev/null || echo "")
    SVAMI_URL=$(gcloud run services describe svami --region=${REGION} --project=${PROJECT_ID} --format="value(status.url)" 2>/dev/null || echo "")
    
    if [ -n "$JANAPADA_URL" ]; then
        printf "🗄️  Testing Janapada search: "
        if curl -s -f --max-time 15 -X POST "$JANAPADA_URL/" -H "Content-Type: application/json" -d '{"jsonrpc": "2.0", "method": "search", "params": {"query": "test search", "k": 1}, "id": "test"}' > /dev/null; then
            echo "✅ Working"
        else
            echo "❌ Failed"
        fi
    fi
    
    if [ -n "$AMATYA_URL" ]; then
        printf "🎭 Testing Amatya advice: "
        if curl -s -f --max-time 15 -X POST "$AMATYA_URL/" -H "Content-Type: application/json" -d '{"jsonrpc": "2.0", "method": "advise", "params": {"role": "developer", "chunks": []}, "id": "test"}' > /dev/null; then
            echo "✅ Working"
        else
            echo "❌ Failed"
        fi
    fi
    
    if [ -n "$SVAMI_URL" ]; then
        printf "🎼 Testing Svami orchestration: "
        if curl -s -f --max-time 15 "$SVAMI_URL/services" > /dev/null; then
            echo "✅ Working"
        else
            echo "❌ Failed"
        fi
    fi
}

case "${1:-check}" in
    "check")
        check_health
        ;;
    "logs")
        view_logs
        ;;
    "metrics")
        view_metrics
        ;;
    "test")
        test_integration
        ;;
    *)
        echo "Usage: $0 [check|logs|metrics|test]"
        echo "  check   - Check service health endpoints"
        echo "  logs    - View recent logs from all services"
        echo "  metrics - View service metrics"
        echo "  test    - Test inter-service communication"
        ;;
esac