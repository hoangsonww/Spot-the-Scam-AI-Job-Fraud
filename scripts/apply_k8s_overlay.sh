#!/usr/bin/env bash
set -euo pipefail

OVERLAY="${1:-ops/k8s/overlays/staging-canary}"
NAMESPACE="${2:-spot-scam}"

if [[ "${NAMESPACE}" != "spot-scam" ]]; then
  echo "This repository currently uses a fixed Kubernetes namespace: spot-scam."
  echo "Received namespace '${NAMESPACE}'. Update kustomization namespace fields before using a custom namespace."
  exit 1
fi

echo "Applying kustomize overlay ${OVERLAY} to namespace ${NAMESPACE}..."
kubectl kustomize "${OVERLAY}" | kubectl apply -n "${NAMESPACE}" -f -

if command -v argo-rollouts >/dev/null 2>&1; then
  echo "Current rollout status:"
  argo-rollouts get rollout spot-scam-api -n "${NAMESPACE}"
else
  echo "argo-rollouts CLI not found; skipped status display."
fi
