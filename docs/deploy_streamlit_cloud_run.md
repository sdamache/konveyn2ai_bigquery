# Deploying the Documentation Coverage Dashboard to Cloud Run

This guide walks you through packaging the Streamlit dashboard and running it on Google Cloud Run. If you prefer an automated deployment of the full microservice stack (Janapada, Amatya, Svami, and the Streamlit dashboard), run `deployment/scripts/deploy-to-cloud-run.sh`; it now builds all four images, pushes them to Artifact Registry, and rolls out the Cloud Run services with health checks.

## Prerequisites

- Google Cloud project: `konveyn2ai`
- gcloud CLI authenticated (`gcloud auth login`) and project set (`gcloud config set project konveyn2ai`)
- Artifact Registry or Container Registry enabled
- BigQuery tables (`documentation_progress_snapshots`, etc.) accessible by the service account you deploy with

> These steps assume you are working from the repository root with the Python 3.11 virtual environment (`source .venv/bin/activate`) already set up for local testing.

## 1. Build the container image

A dedicated Dockerfile is available at `Dockerfile.streamlit`.

```bash
# (optional) configure Docker to talk to Google registries
gcloud auth configure-docker

# Build the container
DOCKER_IMAGE=gcr.io/konveyn2ai/coverage-dashboard:$(date +%Y%m%d-%H%M%S)
docker build -f Dockerfile.streamlit -t "$DOCKER_IMAGE" .

# Push to Container / Artifact Registry
docker push "$DOCKER_IMAGE"
# Managed build (all services + dashboard)
./deployment/scripts/deploy-to-cloud-run.sh

# Manual build & push of the dashboard only
```

If you prefer to offload the build to Cloud Build:

```bash
gcloud builds submit --config=- --substitutions=_IMAGE=gcr.io/konveyn2ai/coverage-dashboard <<'YAML'
steps:
  - name: gcr.io/cloud-builders/docker
    args: ['build', '-f', 'Dockerfile.streamlit', '-t', '$_IMAGE', '.']
images:
  - '$_IMAGE'
YAML
```

## 2. Provision a service account (optional but recommended)

```bash
gcloud iam service-accounts create streamlit-dashboard \
  --display-name "Streamlit Dashboard"

gcloud projects add-iam-policy-binding konveyn2ai \
  --member "serviceAccount:streamlit-dashboard@konveyn2ai.iam.gserviceaccount.com" \
  --role roles/bigquery.dataViewer
```

Grant additional roles (e.g. `roles/bigquery.readSessionUser`) if your dashboard needs them.

## 3. Deploy to Cloud Run

```bash
SERVICE=coverage-dashboard
REGION=us-central1
IMAGE=gcr.io/konveyn2ai/coverage-dashboard:latest   # use the tag you pushed

# Deploy
gcloud run deploy "$SERVICE" \
  --platform managed \
  --region "$REGION" \
  --image "$IMAGE" \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=konveyn2ai,DOCUMENTATION_DATASET=documentation_ops \
  --service-account streamlit-dashboard@konveyn2ai.iam.gserviceaccount.com
```

Notes:
- Cloud Run automatically injects the `PORT` environment variable; the Dockerfile maps it to Streamlit (`STREAMLIT_SERVER_PORT`).
- Remove `--allow-unauthenticated` if you want IAP/Auth proxy in front of the app.

## 4. Validate

- After deployment, gcloud prints the service URL. Open it in the browser.
- Refresh the dashboard to ensure BigQuery calls succeed. You may want to run:
  ```bash
  python scripts/compute_progress_snapshot.py --project-id konveyn2ai --dataset documentation_ops --snapshot-date $(date +%F)
  ```
  to ensure the table has fresh data.

## 5. Update and redeploy

When you modify the dashboard:

```bash
# rebuild with a new tag
NEW_IMAGE=gcr.io/konveyn2ai/coverage-dashboard:$(date +%Y%m%d-%H%M)
docker build -f Dockerfile.streamlit -t "$NEW_IMAGE" .
docker push "$NEW_IMAGE"

gcloud run deploy coverage-dashboard \
  --platform managed \
  --region us-central1 \
  --image "$NEW_IMAGE" \
  --service-account streamlit-dashboard@konveyn2ai.iam.gserviceaccount.com
```

## 6. Cleaning up (optional)

- Delete unused image revisions: `gcloud container images delete IMAGE --force-delete-tags`
- Remove the Cloud Run service: `gcloud run services delete coverage-dashboard`
- Tear down the service account if no longer required.

With these steps you can switch from local development to a managed, scalable deployment suitable for demos and production trials.
