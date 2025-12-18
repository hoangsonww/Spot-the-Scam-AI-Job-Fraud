#!/usr/bin/env bash
set -euo pipefail

OVERLAY="${1:-ops/k8s/overlays/staging-canary}"
NAMESPACE="${2:-spot-scam}"

echo "Applying kustomize overlay ${OVERLAY} to namespace ${NAMESPACE}..."
kubectl kustomize "${OVERLAY}" | kubectl apply -n "${NAMESPACE}" -f -

if command -v argo-rollouts >/dev/null 2>&1; then
  echo "Current rollout status:"
  argo-rollouts get rollout spot-scam-api -n "${NAMESPACE}"
else
  echo "argo-rollouts CLI not found; skipped status display."
fi
