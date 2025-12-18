# Deployment Guide

This document provides a practical checklist for promoting Spot the Scam from local dev into staging/production.

## 1. Prerequisites
- ✅ Python 3.12+ virtual environment
- ✅ Node.js 18+ (Next.js dashboard)
- ✅ Downloaded datasets (`scripts/download_data.py`)
- ✅ GPU optional: install `onnxruntime-gpu==<matching version>` inside the virtualenv to avoid ORT CPU warnings.
- ✅ MLflow tracking directory (`mlruns/`) or remote tracking server.

## 2. Build & Package
1. **Install dependencies**
   ```bash
   source .venv/bin/activate
   pip install -e '.[dev]'
   npm --prefix frontend install
   ```
2. **Train models**
   ```bash
   PYTHONPATH=src python -m spot_scam.pipeline.train
   ```
   - Use `--skip-transformer` for quick iterations.
   - Verify `artifacts/metadata.json` for final metrics.
3. **Quantize transformer (optional)**
   ```bash
  PYTHONPATH=src python -m spot_scam.pipeline.quantize
   ```
4. **Inspect MLflow run**
   ```bash
   mlflow ui --backend-store-uri file:./mlruns
   ```

## 3. Serving Options
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
- `--env-manager local` ensures MLflow reuses the project virtualenv (prevents the `pyenv` error on hosts without pyenv).
- The pyfunc model mirrors FastAPI logic (gray-zone, explanations, calibration).

## 4. Frontend deployment
1. Build optimized Next.js bundle:
   ```bash
   cd frontend
   npm run build
   npm run start  # or deploy via Vercel, Netlify, etc.
   ```
2. Set `NEXT_PUBLIC_API_BASE_URL` in `.env.production`.
3. Container option: extend `Dockerfile` or create a separate Dockerfile for the frontend (SSR vs static export).

## 5. Docker Compose (optional local stack)
- Adjust `docker-compose.yml` to align with your environment.
- Typical run:
  ```bash
  docker compose up --build
  ```
- Compose file includes API + frontend, mounts `configs/`, `data/`, `artifacts/`, `experiments/`, and `mlruns/`, and sets `MLFLOW_TRACKING_URI=file:///app/mlruns` so containerised jobs log to the shared volume. Update `SPOT_SCAM_ALLOWED_ORIGINS` in the compose file if you expose the API beyond the bundled frontend.

## 6. Kubernetes progressive delivery (blue/green + canary)
- Manifests live in `ops/k8s/` (base + overlays). Staging defaults to canary (`ops/k8s/overlays/staging-canary`), production defaults to blue/green (`ops/k8s/overlays/prod-bluegreen`).
- Prereqs: Argo Rollouts CRD + CLI, NGINX Ingress, Prometheus for analysis templates, RWX storage class for PVCs, and cert-manager (for TLS secret `spot-scam-api-tls`).
- Deploy:
  ```bash
  ./scripts/apply_k8s_overlay.sh ops/k8s/overlays/staging-canary spot-scam
  # or production
  ./scripts/apply_k8s_overlay.sh ops/k8s/overlays/prod-bluegreen spot-scam
  ```
- Promotion and health checks:
  ```bash
  argo-rollouts get rollout spot-scam-api -n spot-scam
  argo-rollouts promote spot-scam-api -n spot-scam        # after preview validation
  argo-rollouts terminate spot-scam-api -n spot-scam      # fast rollback
  ```
- Base resources include HPA (cpu+mem), PDB, NetworkPolicy, PVCs for artifacts/tracking/mlruns, ingress TLS, and Prometheus-based AnalysisTemplates for p95 latency + error budget.
- Swap strategy by pointing the overlay to `rollout-canary.yaml` or `rollout-bluegreen.yaml` in `ops/k8s/base` and updating the image tag via `kustomize edit set image ghcr.io/your-org/spot-scam-api=<tag>`.
- Optional smoke/load test before promotion: `API_BASE=https://staging-api.yourdomain ./scripts/loadtest_k6.sh` (uses `ops/observability/k6-smoke.js`).

## 7. Observability Checklist
- **Logging:** FastAPI uses standard logging; extend with structured logs (JSON) for production.
- **Metrics:** expose `/metrics` (e.g., `prometheus-fastapi-instrumentator`) and scrape via ServiceMonitor (`ops/k8s/base/servicemonitor.yaml`). Alerts in `ops/k8s/base/prometheus-rules.yaml`.
- **Tracing:** set `OTEL_EXPORTER_OTLP_ENDPOINT` to ship traces to your collector (already wired in ConfigMap).
- **Alerts:** monitor F1 drop or distribution shift by comparing new predictions vs baseline token frequencies.

## 8. Security Considerations
- Sanitize incoming JSON (FastAPI already enforces schema but consider additional checks).
- Rate-limit `/predict` if exposed to public networks.
- Store secrets (API keys, database connections) in environment variables or a secret manager.

## 9. Promoting to Production
1. Run full training with transformer (time-intensive) and verify metrics.
2. Review MLflow run; tag the best run as `Production`.
3. Deploy FastAPI or MLflow server behind load balancer (ASGI app for Uvicorn/Gunicorn).
4. Point Next.js app at the deployed API (update `.env`).
5. Re-run sanity checks (score curated samples, run inference smoke tests).

## 10. Rollback Plan
- Keep previous model version in MLflow; revert by serving the earlier run ID.
- Maintain last stable Docker image (tagged release).
- If using feature flags, enable/disable new model features without redeploying code.

---

For a deeper look at data flow and modules, read `docs/pipeline_walkthrough.md` and `ARCHITECTURE.md`. For explanation specifics, see `docs/explainability.md`.
