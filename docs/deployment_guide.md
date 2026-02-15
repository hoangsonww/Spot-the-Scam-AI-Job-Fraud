# Deployment Guide

This document provides a practical checklist for promoting Spot the Scam from local development to staging or production. It is focused on what this repository already supports.

## Table of Contents

- [Deployment Prerequisites](#deployment-prerequisites)
- [Build and Package Workflow](#build-and-package-workflow)
- [Serving Options](#serving-options)
- [Frontend Deployment Notes](#frontend-deployment-notes)
- [Docker Compose Stack](#docker-compose-stack)
- [Kubernetes Progressive Delivery](#kubernetes-progressive-delivery)
- [Argo CD GitOps (Optional)](#argo-cd-gitops-optional)
- [Observability Checklist](#observability-checklist)
- [Security and Privacy Considerations](#security-and-privacy-considerations)
- [Production Promotion Checklist](#production-promotion-checklist)
- [Rollback Strategy](#rollback-strategy)
- [Related Documentation](#related-documentation)

## Deployment Prerequisites

Recommended baseline:

- Python 3.12+ virtual environment
- Node.js 18+ for the frontend
- Trained artifacts under `artifacts/`
- A configured MLflow tracking directory or server
- Optional GPU support for transformer workloads

If the API fails on startup, the most common root cause is missing or stale artifacts. Re-run training to regenerate them.

## Build and Package Workflow

A standard “local to deployable” flow looks like this.

### 1. Install dependencies

```bash
source .venv/bin/activate
pip install -e '.[dev]'
npm --prefix frontend install
```

### 2. Train models

```bash
PYTHONPATH=src python -m spot_scam.pipeline.train
```

For faster iterations:

```bash
PYTHONPATH=src python -m spot_scam.pipeline.train --skip-transformer
```

### 3. Quantize transformer (optional)

```bash
PYTHONPATH=src python -m spot_scam.pipeline.quantize
```

### 4. Inspect MLflow runs (optional but recommended)

```bash
mlflow ui --backend-store-uri file:./mlruns
```

## Serving Options

The repository supports multiple serving paths. FastAPI is the most direct and is what the frontend expects.

### FastAPI (recommended)

```bash
SPOT_SCAM_ALLOWED_ORIGINS="http://localhost:3000" \
SPOT_SCAM_USE_QUANTIZED=0 \
PYTHONPATH=src uvicorn spot_scam.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### MLflow model server

```bash
mlflow models serve --env-manager local -m runs:/<RUN_ID>/model -p 8080
```

Notes:

- `--env-manager local` ensures MLflow reuses your current virtualenv.
- The pyfunc model is designed to mirror core preprocessing and policy logic.

## Frontend Deployment Notes

The frontend is a Next.js App Router app under `frontend/`.

### Build and run

```bash
cd frontend
npm run build
npm run start
```

### Environment variable

Set the API base URL:

- `NEXT_PUBLIC_API_BASE_URL`

For local deployments, `http://localhost:8000` is typical.

## Docker Compose Stack

`docker-compose.yml` provides a two-service local stack:

```bash
docker compose up --build
```

What it mounts and why:

- `configs/` for config visibility
- `artifacts/` for model persistence
- `experiments/` for analysis outputs
- `data/` as read-only input data
- `mlruns/` for MLflow tracking

Update `SPOT_SCAM_ALLOWED_ORIGINS` in the compose file if you expose the API beyond the bundled frontend.

## Kubernetes Progressive Delivery

Kubernetes scaffolding lives under `ops/k8s/` with base resources and overlays.

- Staging overlay: `ops/k8s/overlays/staging-canary`
- Production overlay: `ops/k8s/overlays/prod-bluegreen`

Bootstrap ingress and Argo Rollouts controller first:

```bash
./ops/ci/bootstrap_cluster_addons.sh
kubectl get crd rollouts.argoproj.io
```

Apply an overlay:

```bash
./scripts/apply_k8s_overlay.sh ops/k8s/overlays/staging-canary spot-scam
```

Promotion and health checks use Argo Rollouts commands such as:

```bash
argo-rollouts get rollout spot-scam-api -n spot-scam
argo-rollouts promote spot-scam-api -n spot-scam
argo-rollouts abort spot-scam-api -n spot-scam
```

## Argo CD GitOps (Optional)

Argo CD assets live under `ops/argo/` and can be bootstrapped with:

```bash
./ops/ci/bootstrap_argocd.sh --env staging --provider aws --repo-url https://github.com/<org>/<repo>.git --revision main
```

After bootstrap, sync and wait:

```bash
./ops/ci/argocd_sync_wait.sh --app spot-scam-staging-aws --timeout-sec 900
```

## Observability Checklist

The repo includes several hooks, but production readiness still depends on your runtime environment.

Recommended checks:

- Logging is enabled and aggregated.
- A metrics endpoint exists and is scraped.
- Alerting rules match your SLOs.
- Latency and error budgets are monitored during rollout.
- Tracking data retention is intentional and documented.

The `ops/` directory includes k6 scripts and monitoring scaffolding you can adapt.

## Security and Privacy Considerations

The system handles user-provided text. Treat it accordingly.

- Avoid logging raw payloads in external systems unless necessary.
- Keep secrets in environment variables or a secret manager.
- Add rate limiting if the API is exposed publicly.
- Review third-party API usage, especially for `/chat`.

## Production Promotion Checklist

A simple, reliable promotion workflow:

1. Train and confirm metrics in `artifacts/metadata.json`.
2. Inspect key plots under `experiments/figs/`.
3. Run sanity checks via the UI or `curl`.
4. Deploy API and frontend.
5. Monitor latency, error rate, and review queue behavior.

## Rollback Strategy

Have a rollback plan before you need it.

- Keep the previous artifact set and image tag available.
- If using MLflow, retain prior run IDs.
- Roll back by reverting to the prior artifact bundle or image tag.

## Related Documentation

- Multi-cloud production runbook: [DEPLOYMENT.md](../DEPLOYMENT.md)
- Jenkins CI/CD runbook: [ops/ci/jenkins/README.md](../ops/ci/jenkins/README.md)
- Deployment validation script: [ops/ci/validate_deployment_assets.sh](../ops/ci/validate_deployment_assets.sh)
- Deployment preflight checks: [ops/ci/preflight_deploy_checks.sh](../ops/ci/preflight_deploy_checks.sh)
- Argo CD GitOps assets: [ops/argo/README.md](../ops/argo/README.md)
- End-to-end setup: [INSTRUCTIONS.md](../INSTRUCTIONS.md)
- MLOps lifecycle: [MLOPS.md](../MLOPS.md)
- System design: [ARCHITECTURE.md](../ARCHITECTURE.md)
- Training strategy: [TRAINING_ANALYSIS.md](../TRAINING_ANALYSIS.md)
- Metrics and diagnostics: [RESULTS.md](../RESULTS.md)
- Pipeline narrative: [docs/pipeline_walkthrough.md](pipeline_walkthrough.md)
- Explainability details: [docs/explainability.md](explainability.md)
- DevOps scaffolding: [docs/DEVOPS_READINESS.md](DEVOPS_READINESS.md)
