# DevOps and Delivery Readiness

This repository includes production-oriented deployment scaffolding (progressive delivery, hardening, and CI/CD examples) under `ops/`.

This document summarizes what is already present and how to use it as a starting point.

## Table of Contents

- [What Is Included](#what-is-included)
- [Kubernetes Progressive Delivery](#kubernetes-progressive-delivery)
- [How to Deploy with Overlays](#how-to-deploy-with-overlays)
- [CI/CD Pipeline Example (Tekton)](#cicd-pipeline-example-tekton)
- [Operational Guardrails Present in Repo](#operational-guardrails-present-in-repo)
- [Runbook Snippets](#runbook-snippets)
- [Related Documentation](#related-documentation)

## What Is Included

Key locations:

- Kubernetes manifests: `ops/k8s/`
- CI/CD example: `ops/ci/tekton-pipeline.yaml`
- Load testing: `ops/observability/` and `scripts/loadtest_k6.sh`

These assets are intended as scaffolding you can adapt to your environment.

## Kubernetes Progressive Delivery

Kubernetes assets are organized as base resources plus overlays:

- Base: `ops/k8s/base/`
- Staging canary: `ops/k8s/overlays/staging-canary/`
- Production blue/green: `ops/k8s/overlays/prod-bluegreen/`

Highlights include:

- Argo Rollouts strategies
- HPA, PDB, and NetworkPolicy
- TLS ingress scaffolding
- PVCs for artifacts, tracking, and MLflow
- Analysis templates and Prometheus rule scaffolding

## How to Deploy with Overlays

Apply an overlay using the helper script:

```bash
# staging canary
./scripts/apply_k8s_overlay.sh ops/k8s/overlays/staging-canary spot-scam

# production blue/green
./scripts/apply_k8s_overlay.sh ops/k8s/overlays/prod-bluegreen spot-scam
```

Set image tags via kustomize inside the overlay directory:

```bash
kustomize edit set image ghcr.io/your-org/spot-scam-api=<tag>
```

## CI/CD Pipeline Example (Tekton)

The repo includes a Tekton pipeline definition at:

- `ops/ci/tekton-pipeline.yaml`

It covers the shape of a full pipeline, including:

- Backend checks
- Frontend checks and builds
- Image builds and scans
- Kubernetes apply
- Smoke testing hooks
- Promotion hooks

Treat it as a reference pipeline that you can adapt.

## Operational Guardrails Present in Repo

Examples of guardrails in the manifest scaffolding:

- Networking constraints via NetworkPolicy
- Availability controls via PDB
- Scaling controls via HPA
- Storage scaffolding for model and tracking state
- Observability hooks via ServiceMonitor and Prometheus rules

These still require you to provide compatible cluster components (ingress, monitoring stack, storage class, and rollouts controller).

## Runbook Snippets

Common Argo Rollouts commands:

```bash
argo-rollouts get rollout spot-scam-api -n spot-scam
argo-rollouts pause spot-scam-api -n spot-scam
argo-rollouts promote spot-scam-api -n spot-scam
argo-rollouts rollback spot-scam-api -n spot-scam --to-revision <rev>
```

## Related Documentation

- Deployment checklist: [docs/deployment_guide.md](deployment_guide.md)
- MLOps lifecycle: [MLOPS.md](../MLOPS.md)
- System design: [ARCHITECTURE.md](../ARCHITECTURE.md)
- Setup and operations: [INSTRUCTIONS.md](../INSTRUCTIONS.md)
