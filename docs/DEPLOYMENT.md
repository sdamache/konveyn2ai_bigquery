# KonveyN2AI Cloud Run Deployment Guide

This guide provides step-by-step instructions for deploying all KonveyN2AI services to Google Cloud Run.

## Prerequisites

- Google Cloud Project with billing enabled
- Google Cloud SDK (`gcloud`) installed and authenticated
- Docker installed for local building
- Required environment variables configured in `.env` file

## Quick Start

For a complete deployment, run these commands in order:

```bash
# 1. Set up Artifact Registry and prerequisites
./setup-artifact-registry.sh

# 2. Deploy all services to Cloud Run
./deploy-to-cloud-run.sh

# 3. Configure service authentication
./setup-service-auth.sh

# 4. Set up monitoring and alerting
./setup-monitoring.sh

# 5. Test the deployment
./test-deployment.sh
```

## Detailed Steps

### 1. Environment Setup

Ensure your `.env` file contains all required variables:

```bash
# Copy the example and fill in your values
cp .env.example .env

# Required variables:
# GOOGLE_CLOUD_PROJECT=konveyn2ai
# GOOGLE_API_KEY=your_google_api_key
# INDEX_ENDPOINT_ID=your_index_endpoint_id
# INDEX_ID=your_index_id
```

### 2. Google Cloud Setup

```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login

# Set your project
gcloud config set project konveyn2ai
```

### 3. Artifact Registry Setup

Run the setup script to create the Docker repository:

```bash
./setup-artifact-registry.sh
```

This script will:
- Enable required Google Cloud APIs
- Create Artifact Registry repository
- Configure Docker authentication
- Create service account with proper permissions

### 4. Deploy Services

Deploy all three services to Cloud Run:

```bash
./deploy-to-cloud-run.sh
```

This script will:
- Build Docker images for all services
- Push images to Artifact Registry
- Deploy services to Cloud Run with proper configuration
- Configure environment variables and resource limits

### 5. Configure Authentication

Set up service-to-service authentication:

```bash
./setup-service-auth.sh
```

This configures:
- IAM policies for service communication
- Identity token authentication
- Service URL updates

### 6. Set Up Monitoring

Configure monitoring and alerting:

```bash
./setup-monitoring.sh
```

This creates:
- Uptime checks for all services
- Email notification channel
- Alerting policies for errors, memory usage, and downtime
- Custom monitoring dashboard

### 7. Test Deployment

Verify everything is working:

```bash
# Comprehensive test
./test-deployment.sh

# Quick health check
./test-deployment.sh quick

# Load testing
./test-deployment.sh load
```

## Service Configuration

Each service is deployed with:

- **Memory**: 256Mi
- **CPU**: 1 vCPU
- **Min Instances**: 1
- **Max Instances**: 10
- **Service Account**: `konveyn2ai-service@konveyn2ai.iam.gserviceaccount.com`

### Service Ports

- **Janapada Memory Service**: Port 8081
- **Amatya Role Prompter**: Port 8082
- **Svami Orchestrator**: Port 8080

## Environment Variables

### Production Environment

The deployment uses these key environment variables:

```bash
# Google Cloud
GOOGLE_CLOUD_PROJECT=konveyn2ai
GOOGLE_API_KEY=<your-api-key>

# Vector Index
INDEX_ENDPOINT_ID=<your-endpoint-id>
INDEX_ID=<your-index-id>

# Service Configuration
ENVIRONMENT=production
DEBUG=false
```

## Monitoring and Logging

### View Logs

```bash
# View logs for specific service
gcloud logs tail --service=janapada --project=konveyn2ai
gcloud logs tail --service=amatya --project=konveyn2ai
gcloud logs tail --service=svami --project=konveyn2ai

# View all logs
gcloud logs tail --project=konveyn2ai
```

### Monitor Services

```bash
# Check service health
./monitor-services.sh check

# View recent logs
./monitor-services.sh logs

# View metrics
./monitor-services.sh metrics
```

### Cloud Monitoring Dashboard

Access the monitoring dashboard at:
https://console.cloud.google.com/monitoring/dashboards?project=konveyn2ai

## Troubleshooting

### Common Issues

1. **Service deployment fails**
   - Check if all required APIs are enabled
   - Verify service account permissions
   - Ensure environment variables are set correctly

2. **Health checks fail**
   - Services might still be starting (wait 1-2 minutes)
   - Check service logs for errors
   - Verify network connectivity

3. **Authentication errors**
   - Run `./setup-service-auth.sh` again
   - Check service account IAM roles
   - Verify identity token configuration

### Debug Commands

```bash
# Check service status
gcloud run services list --project=konveyn2ai

# Describe specific service
gcloud run services describe <service-name> --region=us-central1 --project=konveyn2ai

# View service revisions
gcloud run revisions list --service=<service-name> --region=us-central1 --project=konveyn2ai

# Check logs
gcloud logs read "resource.type=cloud_run_revision" --limit=50 --project=konveyn2ai
```

## Cleanup

To remove all deployed resources:

```bash
# Delete Cloud Run services
gcloud run services delete janapada --region=us-central1 --project=konveyn2ai --quiet
gcloud run services delete amatya --region=us-central1 --project=konveyn2ai --quiet
gcloud run services delete svami --region=us-central1 --project=konveyn2ai --quiet

# Delete Artifact Registry repository
gcloud artifacts repositories delete konveyn2ai-repo --location=us-central1 --project=konveyn2ai --quiet

# Delete service account
gcloud iam service-accounts delete konveyn2ai-service@konveyn2ai.iam.gserviceaccount.com --project=konveyn2ai --quiet
```

## Security Considerations

- Services use non-root containers with dedicated user accounts
- Service-to-service communication uses Google Cloud IAM authentication
- Environment variables are managed through Cloud Run configuration
- All services run with minimal required permissions

## Performance Optimization

- Multi-stage Docker builds for smaller images
- Health checks configured with appropriate timeouts
- Auto-scaling configured with minimum instances for warm starts
- Resource limits set to balance performance and cost

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review service logs using the provided commands
3. Verify all prerequisites are met
4. Test individual components using the test scripts

## Files Created

This deployment creates the following files:

- `cloudbuild.yaml` - Google Cloud Build configuration
- `setup-artifact-registry.sh` - Artifact Registry setup script
- `deploy-to-cloud-run.sh` - Main deployment script
- `setup-service-auth.sh` - Service authentication configuration
- `setup-monitoring.sh` - Monitoring and alerting setup
- `test-deployment.sh` - Deployment testing script
- `.env.production` - Production environment template
- `monitor-services.sh` - Service monitoring utility
- `service-auth-helper.py` - Authentication helper script