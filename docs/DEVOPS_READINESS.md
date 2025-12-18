# DevOps & Delivery Readiness

This repository now includes production-grade deployment assets (progressive delivery, hardening, CI/CD). No Docker or GitHub configuration was changed.

## Progressive delivery on Kubernetes
- **Location:** `ops/k8s/` (base resources + overlays for staging canary and prod blue/green).
- **Workloads:** Argo Rollouts with health probes, HPA (CPU+memory), PDB, NetworkPolicy, TLS Ingress, PVCs for `artifacts/`, `tracking/`, and `mlruns/`.
- **Strategies:**  
  - Staging defaults to **canary** (weighted 20% → 60% → 100% with Prometheus analysis).  
  - Production defaults to **blue/green** (manual promotion, preview replicas, post-promotion analysis).  
- **Promotion controls:** `argo-rollouts promote spot-scam-api -n spot-scam` after validation. Abort with `argo-rollouts terminate`.
- **Health checks:** Prometheus AnalysisTemplate (`latency-error-budget`) enforces <2% errors and p95 latency <800ms before promotion.

## How to deploy
```bash
# staging canary
./scripts/apply_k8s_overlay.sh ops/k8s/overlays/staging-canary

# production blue/green
./scripts/apply_k8s_overlay.sh ops/k8s/overlays/prod-bluegreen
```
Set image tags via `kustomize edit set image ghcr.io/your-org/spot-scam-api=<tag>` inside the chosen overlay.

## CI/CD pipeline (Tekton)
- **Definition:** `ops/ci/tekton-pipeline.yaml`
- **Stages:** git clone → backend lint/type-check/tests → frontend lint/build → Kaniko image builds (API + frontend) → Trivy scans → `kubectl kustomize` apply → k6 smoke → optional Argo Rollouts promotion.
- **Parameters:** `repo-url`, `git-revision`, `image-tag`, `registry`, `overlay`. Defaults point to staging canary overlay and `ghcr.io/your-org`.
- **Promotion hook:** pipeline ends with `argo-rollouts promote` so blue/green can be gated manually (keep `autoPromotionEnabled: false`).

## Operational guardrails
- **NetworkPolicy:** ingress limited to namespace + ingress controller; egress restricted to DNS, monitoring, and TLS.
- **Storage:** RWX PVCs for artifacts and tracking; adjust storage class/size as needed.
- **Security:** secrets pulled from `spot-scam-api-secrets` (e.g., `GEMINI_API_KEY`). Avoid embedding credentials in ConfigMaps.
- **Reliability:** PDB keeps at least one pod running; anti-affinity spreads pods across nodes; HPA scales between 3–10 replicas.
- **Observability:** ServiceMonitor + PrometheusRule in `ops/k8s/base/` for error-rate/latency alerts. Expect a `/metrics` endpoint (enable FastAPI/Prometheus exporter in the image).
- **Synthetic checks:** `ops/observability/k6-smoke.js` + `scripts/loadtest_k6.sh` run a lightweight smoke to catch regressions before promotion.
- **Tracing:** set `OTEL_EXPORTER_OTLP_ENDPOINT` and propagate trace headers via ingress; wire FastAPI middleware to emit spans to your collector.

## Runbook snippets
- Status: `argo-rollouts get rollout spot-scam-api -n spot-scam`
- Pause canary: `argo-rollouts pause spot-scam-api -n spot-scam`
- Promote: `argo-rollouts promote spot-scam-api -n spot-scam`
- Roll back: `argo-rollouts rollback spot-scam-api -n spot-scam --to-revision <rev>`
