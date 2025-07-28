#!/bin/bash

# Setup monitoring and health checks for KonveyN2AI Cloud Run services
# This script configures Cloud Monitoring, logging, and alerting

set -e

PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"konveyn2ai"}
REGION=${GOOGLE_CLOUD_LOCATION:-"us-central1"}
NOTIFICATION_EMAIL=${NOTIFICATION_EMAIL:-"contact@konveyn2ai.com"}

echo "📊 Setting up monitoring and health checks for KonveyN2AI"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Notification Email: $NOTIFICATION_EMAIL"

# Enable required APIs
echo "📡 Enabling required APIs..."
gcloud services enable \
    monitoring.googleapis.com \
    logging.googleapis.com \
    clouderrorreporting.googleapis.com \
    cloudtrace.googleapis.com \
    --project=$PROJECT_ID

# Create notification channel for email alerts
echo "📧 Creating notification channel..."
NOTIFICATION_CHANNEL_CONFIG=$(cat << EOF
{
  "type": "email",
  "displayName": "KonveyN2AI Contact Email",
  "description": "Email notifications for KonveyN2AI services",
  "labels": {
    "email_address": "$NOTIFICATION_EMAIL"
  }
}
EOF
)

# Create notification channel
NOTIFICATION_CHANNEL_ID=$(gcloud alpha monitoring channels create \
    --channel-content="$NOTIFICATION_CHANNEL_CONFIG" \
    --project=$PROJECT_ID \
    --format="value(name)" | sed 's|.*/||')

echo "Created notification channel: $NOTIFICATION_CHANNEL_ID"

# Create uptime checks for each service
echo "🏥 Creating uptime checks..."

# Get service URLs
JANAPADA_URL=$(gcloud run services describe janapada --region=${REGION} --project=${PROJECT_ID} --format="value(status.url)" 2>/dev/null || echo "")
AMATYA_URL=$(gcloud run services describe amatya --region=${REGION} --project=${PROJECT_ID} --format="value(status.url)" 2>/dev/null || echo "")
SVAMI_URL=$(gcloud run services describe svami --region=${REGION} --project=${PROJECT_ID} --format="value(status.url)" 2>/dev/null || echo "")

if [ -n "$JANAPADA_URL" ]; then
    JANAPADA_HOST=$(echo $JANAPADA_URL | sed 's|https://||' | sed 's|/.*||')
    gcloud alpha monitoring uptime create "Janapada Health Check" \
        --resource-type=uptime-url \
        --resource-labels=host=$JANAPADA_HOST,project_id=$PROJECT_ID \
        --protocol=https \
        --path="/health" \
        --project=$PROJECT_ID
fi

if [ -n "$AMATYA_URL" ]; then
    AMATYA_HOST=$(echo $AMATYA_URL | sed 's|https://||' | sed 's|/.*||')
    gcloud alpha monitoring uptime create "Amatya Health Check" \
        --resource-type=uptime-url \
        --resource-labels=host=$AMATYA_HOST,project_id=$PROJECT_ID \
        --protocol=https \
        --path="/health" \
        --project=$PROJECT_ID
fi

if [ -n "$SVAMI_URL" ]; then
    SVAMI_HOST=$(echo $SVAMI_URL | sed 's|https://||' | sed 's|/.*||')
    gcloud alpha monitoring uptime create "Svami Health Check" \
        --resource-type=uptime-url \
        --resource-labels=host=$SVAMI_HOST,project_id=$PROJECT_ID \
        --protocol=https \
        --path="/health" \
        --project=$PROJECT_ID
fi

# Create alerting policies
echo "🚨 Creating alerting policies..."

# High error rate alert
ERROR_RATE_POLICY=$(cat << 'EOF'
{
  "displayName": "KonveyN2AI High Error Rate",
  "documentation": {
    "content": "Alert when any KonveyN2AI service has high error rate (>5% over 5 minutes)",
    "mimeType": "text/markdown"
  },
  "conditions": [
    {
      "displayName": "High error rate condition",
      "conditionThreshold": {
        "filter": "resource.type=\"cloud_run_revision\" AND resource.label.service_name=~\"janapada|amatya|svami\" AND metric.type=\"run.googleapis.com/request_count\"",
        "comparison": "GREATER_THAN",
        "thresholdValue": 0.05,
        "duration": "300s",
        "aggregations": [
          {
            "alignmentPeriod": "300s",
            "perSeriesAligner": "ALIGN_RATE",
            "crossSeriesReducer": "REDUCE_SUM",
            "groupByFields": ["resource.label.service_name"]
          }
        ]
      }
    }
  ],
  "combiner": "OR",
  "enabled": true,
  "notificationChannels": [
    "projects/PROJECT_ID/notificationChannels/NOTIFICATION_CHANNEL_ID"
  ]
}
EOF
)

# Replace placeholders
ERROR_RATE_POLICY=$(echo "$ERROR_RATE_POLICY" | sed "s/PROJECT_ID/$PROJECT_ID/g" | sed "s/NOTIFICATION_CHANNEL_ID/$NOTIFICATION_CHANNEL_ID/g")

# High memory usage alert
MEMORY_POLICY=$(cat << 'EOF'
{
  "displayName": "KonveyN2AI High Memory Usage",
  "documentation": {
    "content": "Alert when any KonveyN2AI service uses >80% memory for 10 minutes",
    "mimeType": "text/markdown"
  },
  "conditions": [
    {
      "displayName": "High memory usage condition",
      "conditionThreshold": {
        "filter": "resource.type=\"cloud_run_revision\" AND resource.label.service_name=~\"janapada|amatya|svami\" AND metric.type=\"run.googleapis.com/container/memory/utilizations\"",
        "comparison": "GREATER_THAN",
        "thresholdValue": 0.8,
        "duration": "600s",
        "aggregations": [
          {
            "alignmentPeriod": "300s",
            "perSeriesAligner": "ALIGN_MEAN",
            "crossSeriesReducer": "REDUCE_MEAN",
            "groupByFields": ["resource.label.service_name"]
          }
        ]
      }
    }
  ],
  "combiner": "OR",
  "enabled": true,
  "notificationChannels": [
    "projects/PROJECT_ID/notificationChannels/NOTIFICATION_CHANNEL_ID"
  ]
}
EOF
)

# Replace placeholders
MEMORY_POLICY=$(echo "$MEMORY_POLICY" | sed "s/PROJECT_ID/$PROJECT_ID/g" | sed "s/NOTIFICATION_CHANNEL_ID/$NOTIFICATION_CHANNEL_ID/g")

# Service down alert
SERVICE_DOWN_POLICY=$(cat << 'EOF'
{
  "displayName": "KonveyN2AI Service Down",
  "documentation": {
    "content": "Alert when any KonveyN2AI service is down or unresponsive",
    "mimeType": "text/markdown"
  },
  "conditions": [
    {
      "displayName": "Service down condition",
      "conditionThreshold": {
        "filter": "resource.type=\"uptime_check\" AND metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\"",
        "comparison": "EQUAL",
        "thresholdValue": 0,
        "duration": "180s",
        "aggregations": [
          {
            "alignmentPeriod": "60s",
            "perSeriesAligner": "ALIGN_FRACTION_TRUE",
            "crossSeriesReducer": "REDUCE_MEAN"
          }
        ]
      }
    }
  ],
  "combiner": "OR",
  "enabled": true,
  "notificationChannels": [
    "projects/PROJECT_ID/notificationChannels/NOTIFICATION_CHANNEL_ID"
  ]
}
EOF
)

# Replace placeholders
SERVICE_DOWN_POLICY=$(echo "$SERVICE_DOWN_POLICY" | sed "s/PROJECT_ID/$PROJECT_ID/g" | sed "s/NOTIFICATION_CHANNEL_ID/$NOTIFICATION_CHANNEL_ID/g")

# Create the alerting policies
echo "$ERROR_RATE_POLICY" > /tmp/error-rate-policy.json
echo "$MEMORY_POLICY" > /tmp/memory-policy.json
echo "$SERVICE_DOWN_POLICY" > /tmp/service-down-policy.json

gcloud alpha monitoring policies create --policy-from-file=/tmp/error-rate-policy.json --project=$PROJECT_ID
gcloud alpha monitoring policies create --policy-from-file=/tmp/memory-policy.json --project=$PROJECT_ID
gcloud alpha monitoring policies create --policy-from-file=/tmp/service-down-policy.json --project=$PROJECT_ID

# Clean up temporary files
rm -f /tmp/error-rate-policy.json /tmp/memory-policy.json /tmp/service-down-policy.json

# Create custom dashboard
echo "📈 Creating monitoring dashboard..."
DASHBOARD_CONFIG=$(cat << 'EOF'
{
  "displayName": "KonveyN2AI Services Dashboard",
  "mosaicLayout": {
    "columns": 12,
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Request Count by Service",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\" AND resource.label.service_name=~\"janapada|amatya|svami\" AND metric.type=\"run.googleapis.com/request_count\"",
                    "aggregation": {
                      "alignmentPeriod": "300s",
                      "perSeriesAligner": "ALIGN_RATE",
                      "crossSeriesReducer": "REDUCE_SUM",
                      "groupByFields": ["resource.label.service_name"]
                    }
                  }
                },
                "plotType": "LINE"
              }
            ],
            "timeshiftDuration": "0s",
            "yAxis": {
              "label": "Requests/sec",
              "scale": "LINEAR"
            }
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "xPos": 6,
        "widget": {
          "title": "Memory Utilization by Service",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\" AND resource.label.service_name=~\"janapada|amatya|svami\" AND metric.type=\"run.googleapis.com/container/memory/utilizations\"",
                    "aggregation": {
                      "alignmentPeriod": "300s",
                      "perSeriesAligner": "ALIGN_MEAN",
                      "crossSeriesReducer": "REDUCE_MEAN",
                      "groupByFields": ["resource.label.service_name"]
                    }
                  }
                },
                "plotType": "LINE"
              }
            ],
            "timeshiftDuration": "0s",
            "yAxis": {
              "label": "Memory Utilization",
              "scale": "LINEAR"
            }
          }
        }
      },
      {
        "width": 12,
        "height": 4,
        "yPos": 4,
        "widget": {
          "title": "Response Latency by Service",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\" AND resource.label.service_name=~\"janapada|amatya|svami\" AND metric.type=\"run.googleapis.com/request_latencies\"",
                    "aggregation": {
                      "alignmentPeriod": "300s",
                      "perSeriesAligner": "ALIGN_DELTA",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_95",
                      "groupByFields": ["resource.label.service_name"]
                    }
                  }
                },
                "plotType": "LINE"
              }
            ],
            "timeshiftDuration": "0s",
            "yAxis": {
              "label": "Latency (ms)",
              "scale": "LINEAR"
            }
          }
        }
      }
    ]
  }
}
EOF
)

echo "$DASHBOARD_CONFIG" > /tmp/dashboard-config.json
gcloud monitoring dashboards create --config-from-file=/tmp/dashboard-config.json --project=$PROJECT_ID
rm -f /tmp/dashboard-config.json

# Create monitoring script for local use
cat > monitor-services.sh << 'EOF'
#!/bin/bash

# Monitor KonveyN2AI services health and metrics
# Usage: ./monitor-services.sh [check|logs|metrics]

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
            echo -n "  $name: "
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
        echo -n "🗄️  Testing Janapada search: "
        if curl -s -f --max-time 15 -X POST "$JANAPADA_URL/" -H "Content-Type: application/json" -d '{"jsonrpc": "2.0", "method": "search", "params": {"query": "test search", "k": 1}, "id": "test"}' > /dev/null; then
            echo "✅ Working"
        else
            echo "❌ Failed"
        fi
    fi
    
    if [ -n "$AMATYA_URL" ]; then
        echo -n "🎭 Testing Amatya advice: "
        if curl -s -f --max-time 15 -X POST "$AMATYA_URL/" -H "Content-Type: application/json" -d '{"jsonrpc": "2.0", "method": "advise", "params": {"role": "developer", "chunks": []}, "id": "test"}' > /dev/null; then
            echo "✅ Working"
        else
            echo "❌ Failed"
        fi
    fi
    
    if [ -n "$SVAMI_URL" ]; then
        echo -n "🎼 Testing Svami orchestration: "
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
EOF

chmod +x monitor-services.sh

echo "✅ Monitoring setup complete!"
echo ""
echo "📋 Configuration Summary:"
echo "- Uptime checks created for all services"
echo "- Email notifications configured for: $NOTIFICATION_EMAIL"
echo "- Alerting policies created for error rate, memory usage, and service availability"
echo "- Custom dashboard created in Cloud Monitoring"
echo "- Local monitoring script created: monitor-services.sh"
echo ""
echo "🔧 Next Steps:"
echo "1. Visit Cloud Monitoring console to view dashboard"
echo "2. Test alerts by simulating service failures"
echo "3. Run './monitor-services.sh check' to verify health"
echo "4. Set up additional custom metrics as needed"
echo ""
echo "📊 View dashboard:"
echo "https://console.cloud.google.com/monitoring/dashboards?project=$PROJECT_ID"