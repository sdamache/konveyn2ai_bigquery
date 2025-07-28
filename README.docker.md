# Docker Deployment Guide for KonveyN2AI

This guide explains how to deploy the KonveyN2AI multi-agent system using Docker containers.

## Architecture Overview

The system consists of three containerized services:

- **Svami Orchestrator** (Port 8080) - Entry point and workflow coordinator
- **Janapada Memory** (Port 8081) - Semantic search and memory management  
- **Amatya Role Prompter** (Port 8082) - Role-based advice generation

## Prerequisites

1. **Docker & Docker Compose** installed
2. **Google Cloud credentials** (credentials.json file)
3. **Environment variables** configured
4. **Vertex AI setup** (for Janapada service)

## Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp .env.docker.template .env

# Edit .env with your actual values
nano .env
```

Required environment variables:
```
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_API_KEY=your-google-api-key
INDEX_ENDPOINT_ID=your-index-endpoint-id
INDEX_ID=your-index-id
```

### 2. Google Cloud Credentials

Place your Google Cloud service account key file as `credentials.json` in the project root:

```bash
# Copy your service account key
cp /path/to/your/service-account-key.json ./credentials.json
```

### 3. Build and Run

```bash
# Build all services
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

### 4. Test Services

```bash
# Test Svami Orchestrator
curl http://localhost:8080/health

# Test Janapada Memory
curl http://localhost:8081/health

# Test Amatya Role Prompter
curl http://localhost:8082/health

# Test full workflow
curl -X POST http://localhost:8080/answer \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer demo-token" \
  -d '{"question": "How do I implement authentication?", "role": "developer"}'
```

## Individual Service Deployment

### Build Individual Images

```bash
# Svami Orchestrator
docker build -f Dockerfile.svami -t konveyn2ai/svami:latest .

# Janapada Memory
docker build -f Dockerfile.janapada -t konveyn2ai/janapada:latest .

# Amatya Role Prompter
docker build -f Dockerfile.amatya -t konveyn2ai/amatya:latest .
```

### Run Individual Containers

```bash
# Svami (depends on other services)
docker run -p 8080:8080 \
  -e JANAPADA_URL=http://host.docker.internal:8081 \
  -e AMATYA_URL=http://host.docker.internal:8082 \
  konveyn2ai/svami:latest

# Janapada
docker run -p 8081:8081 \
  -e GOOGLE_CLOUD_PROJECT=your-project \
  -v ./credentials.json:/app/credentials.json:ro \
  konveyn2ai/janapada:latest

# Amatya
docker run -p 8082:8082 \
  -e GOOGLE_API_KEY=your-api-key \
  -v ./credentials.json:/app/credentials.json:ro \
  konveyn2ai/amatya:latest
```

## Cloud Run Deployment

### 1. Push to Google Container Registry

```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Tag and push images
docker tag konveyn2ai/svami:latest gcr.io/YOUR_PROJECT/svami:latest
docker tag konveyn2ai/janapada:latest gcr.io/YOUR_PROJECT/janapada:latest
docker tag konveyn2ai/amatya:latest gcr.io/YOUR_PROJECT/amatya:latest

docker push gcr.io/YOUR_PROJECT/svami:latest
docker push gcr.io/YOUR_PROJECT/janapada:latest
docker push gcr.io/YOUR_PROJECT/amatya:latest
```

### 2. Deploy to Cloud Run

```bash
# Deploy Janapada (memory service)
gcloud run deploy janapada \
  --image gcr.io/YOUR_PROJECT/janapada:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=YOUR_PROJECT \
  --memory 1Gi \
  --cpu 1

# Deploy Amatya (role prompter)
gcloud run deploy amatya \
  --image gcr.io/YOUR_PROJECT/amatya:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=YOUR_PROJECT \
  --memory 768Mi \
  --cpu 1

# Deploy Svami (orchestrator) - update URLs with actual Cloud Run URLs
gcloud run deploy svami \
  --image gcr.io/YOUR_PROJECT/svami:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars JANAPADA_URL=https://janapada-xxx.run.app \
  --set-env-vars AMATYA_URL=https://amatya-xxx.run.app \
  --memory 512Mi \
  --cpu 1
```

## Resource Configuration

### Container Limits

- **Svami**: 512MB RAM, 1 CPU
- **Janapada**: 1GB RAM, 1.5 CPU (memory-intensive for embeddings)
- **Amatya**: 768MB RAM, 1 CPU

### Health Checks

All services include health check endpoints:
- Interval: 30 seconds
- Timeout: 10 seconds
- Start period: 5 seconds
- Retries: 3

## Troubleshooting

### Common Issues

1. **Service not starting**: Check logs with `docker-compose logs [service]`
2. **Connection refused**: Ensure all services are running and network connectivity
3. **Authentication errors**: Verify credentials.json and environment variables
4. **Memory issues**: Increase Docker memory limits if needed

### Debug Commands

```bash
# View all container status
docker-compose ps

# Check individual service logs
docker-compose logs svami
docker-compose logs janapada
docker-compose logs amatya

# Execute commands inside containers
docker-compose exec svami /bin/bash
docker-compose exec janapada python -c "import vertexai; print('Vertex AI available')"
docker-compose exec amatya python -c "from google import genai; print('Gemini available')"

# Test internal connectivity
docker-compose exec svami curl http://janapada:8081/health
docker-compose exec svami curl http://amatya:8082/health
```

### Performance Monitoring

```bash
# Monitor resource usage
docker stats

# View detailed service information
docker-compose top
```

## Security Considerations

1. **Credentials**: Never commit credentials.json to version control
2. **Environment Variables**: Use Docker secrets for sensitive data in production
3. **Network**: Services communicate within Docker network by default
4. **User**: All containers run as non-root user (appuser)
5. **Updates**: Regularly update base images for security patches

## Development Workflow

```bash
# Development with hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Rebuild after code changes
docker-compose build [service]
docker-compose up -d [service]

# Clean rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```