# Regional Configuration - us-central1

This document outlines the regional resource configuration for the KonveyN2AI project, all deployed in the `us-central1` region for optimal latency and cost efficiency.

## Regional Strategy

**Primary Region**: `us-central1`
- **Reasoning**: Low latency for North American users, comprehensive Google Cloud service availability
- **Backup Region**: `us-east1` (for disaster recovery in production)

## Regional Resources Configured

### 1. Artifact Registry
- **Repository**: `konveyn2ai-repo`
- **Location**: `us-central1`
- **Format**: Docker
- **URL**: `us-central1-docker.pkg.dev/konveyn2ai/konveyn2ai-repo/`
- **Purpose**: Container image storage for all three components

### 2. Vertex AI Platform
- **Project**: `konveyn2ai`
- **Location**: `us-central1`
- **Vector Index**: `projects/72021522495/locations/us-central1/indexes/805460437066842112`
- **Configuration**: 3072 dimensions, COSINE_DISTANCE, 150 approximate neighbors
- **Service Account**: `konveyn2ai-vertex-ai@konveyn2ai.iam.gserviceaccount.com`

### 3. Cloud Run Services (Configured)

#### Amatya Role Prompter
- **Service Name**: `amatya-role-prompter`
- **Region**: `us-central1`
- **Image**: `us-central1-docker.pkg.dev/konveyn2ai/konveyn2ai-repo/amatya-role-prompter:latest`
- **Service Account**: `konveyn2ai-vertex-ai@konveyn2ai.iam.gserviceaccount.com`
- **Resources**: 1-2 CPU, 1-2Gi memory
- **URL Pattern**: `https://amatya-role-prompter-72021522495.us-central1.run.app`

#### Janapada Memory
- **Service Name**: `janapada-memory` 
- **Region**: `us-central1`
- **Image**: `us-central1-docker.pkg.dev/konveyn2ai/konveyn2ai-repo/janapada-memory:latest`
- **Service Account**: `konveyn2ai-vertex-ai@konveyn2ai.iam.gserviceaccount.com`
- **Resources**: 1-2 CPU, 2-4Gi memory (higher memory for vector operations)
- **URL Pattern**: `https://janapada-memory-72021522495.us-central1.run.app`

#### Svami Orchestrator
- **Service Name**: `svami-orchestrator`
- **Region**: `us-central1`
- **Image**: `us-central1-docker.pkg.dev/konveyn2ai/konveyn2ai-repo/svami-orchestrator:latest`
- **Service Account**: `konveyn2ai-cloud-run@konveyn2ai.iam.gserviceaccount.com`
- **Resources**: 1-2 CPU, 1-2Gi memory
- **URL Pattern**: `https://svami-orchestrator-72021522495.us-central1.run.app`

## Service Account Mapping

### Workload Identity Configuration
- **Vertex AI Operations**: Both Amatya and Janapada use `konveyn2ai-vertex-ai` service account
- **Cloud Run Management**: Svami Orchestrator uses `konveyn2ai-cloud-run` service account
- **Regional Binding**: All service accounts have regional permissions in `us-central1`

## Environment Variables (Regional)

All services are configured with regional environment variables:

```yaml
env:
- name: PROJECT_ID
  value: "konveyn2ai"
- name: REGION
  value: "us-central1"
- name: VECTOR_INDEX_ID
  value: "805460437066842112"
```

## Service Communication

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│  Svami Orchestrator │───▶│   Amatya Prompter    │───▶│  Janapada Memory    │
│  (Cloud Run Admin)  │    │  (Vertex AI User)    │    │  (Vertex AI User)   │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
         │                           │                           │
         ▼                           ▼                           ▼
    us-central1                us-central1                us-central1
   Cloud Run SA               Vertex AI SA               Vertex AI SA
```

## Performance Benefits

1. **Low Latency**: All components in same region
2. **Cost Optimization**: No cross-region data transfer costs
3. **Simplified Networking**: Regional VPC configuration
4. **Consistent Performance**: Predictable service-to-service communication

## Deployment Commands

```bash
# Deploy Amatya Role Prompter
gcloud run services replace config/cloud-run/amatya-role-prompter.yaml \
  --region=us-central1 --project=konveyn2ai

# Deploy Janapada Memory
gcloud run services replace config/cloud-run/janapada-memory.yaml \
  --region=us-central1 --project=konveyn2ai

# Deploy Svami Orchestrator
gcloud run services replace config/cloud-run/svami-orchestrator.yaml \
  --region=us-central1 --project=konveyn2ai
```

## Monitoring & Logging

- **Cloud Logging**: All services log to us-central1 region
- **Cloud Monitoring**: Regional dashboards configured
- **Health Checks**: HTTP health endpoints on `/health` for all services

## Next Steps

1. Build and push Docker images to Artifact Registry
2. Deploy services using the provided configurations
3. Test inter-service communication
4. Configure load balancing and custom domains if needed

---

**Note**: This regional configuration supports the hackathon's scalable, real-world architecture requirements.