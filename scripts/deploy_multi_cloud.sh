#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $0 --provider <aws|azure|gcp|oci> --strategy <canary|bluegreen> [--namespace spot-scam]

Options:
  --provider    Cloud provider deployment pack to use.
  --strategy    Rollout strategy: canary or bluegreen.
  --namespace   Kubernetes namespace. Currently only 'spot-scam' is supported.
USAGE
}

PROVIDER=""
STRATEGY=""
NAMESPACE="spot-scam"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider)
      PROVIDER="${2:-}"
      shift 2
      ;;
    --strategy)
      STRATEGY="${2:-}"
      shift 2
      ;;
    --namespace)
      NAMESPACE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$PROVIDER" || -z "$STRATEGY" ]]; then
  usage
  exit 1
fi

if [[ "$PROVIDER" != "aws" && "$PROVIDER" != "azure" && "$PROVIDER" != "gcp" && "$PROVIDER" != "oci" ]]; then
  echo "Invalid provider: $PROVIDER" >&2
  exit 1
fi

if [[ "$STRATEGY" != "canary" && "$STRATEGY" != "bluegreen" ]]; then
  echo "Invalid strategy: $STRATEGY" >&2
  exit 1
fi

if [[ "$NAMESPACE" != "spot-scam" ]]; then
  echo "This repository currently uses a fixed Kubernetes namespace: spot-scam" >&2
  echo "Received --namespace ${NAMESPACE}. Update kustomization namespace fields before using a custom namespace." >&2
  exit 1
fi

for cmd in kubectl; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

OVERLAY_PATH="${PROVIDER}/k8s/${STRATEGY}"
if [[ ! -f "${OVERLAY_PATH}/kustomization.yaml" ]]; then
  echo "Overlay not found: ${OVERLAY_PATH}" >&2
  exit 1
fi

if [[ ! -x "ops/ci/preflight_deploy_checks.sh" ]]; then
  echo "Missing preflight checker: ops/ci/preflight_deploy_checks.sh" >&2
  exit 1
fi

echo "Deploying Spot the Scam"
echo "  provider:  ${PROVIDER}"
echo "  strategy:  ${STRATEGY}"
echo "  namespace: ${NAMESPACE}"

./ops/ci/preflight_deploy_checks.sh \
  --provider "${PROVIDER}" \
  --strategy "${STRATEGY}" \
  --namespace "${NAMESPACE}"

kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
kubectl kustomize "${OVERLAY_PATH}" | kubectl apply -n "${NAMESPACE}" -f -

if command -v argo-rollouts >/dev/null 2>&1; then
  argo-rollouts get rollout spot-scam-api -n "${NAMESPACE}"
else
  echo "argo-rollouts CLI not installed; skipping rollout status command."
fi
