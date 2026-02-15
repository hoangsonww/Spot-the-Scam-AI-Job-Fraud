#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $0 --provider <aws|azure|gcp|oci> --strategy <canary|bluegreen> [--namespace spot-scam]
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

OVERLAY_PATH="${PROVIDER}/k8s/${STRATEGY}"
if [[ ! -f "${OVERLAY_PATH}/kustomization.yaml" ]]; then
  echo "Overlay not found: ${OVERLAY_PATH}" >&2
  exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required" >&2
  exit 1
fi

if ! command -v rg >/dev/null 2>&1; then
  echo "rg (ripgrep) is required" >&2
  exit 1
fi

rendered_file="$(mktemp)"
trap 'rm -f "${rendered_file}"' EXIT
kubectl kustomize "${OVERLAY_PATH}" > "${rendered_file}"

# Ensure Argo Rollouts CRD exists before attempting rollout deployment.
if ! kubectl get crd rollouts.argoproj.io >/dev/null 2>&1; then
  echo "Preflight failed: Argo Rollouts CRD (rollouts.argoproj.io) is not installed in this cluster." >&2
  echo "Install Argo Rollouts controller before deployment." >&2
  echo "Recommended bootstrap: ./ops/ci/bootstrap_cluster_addons.sh --skip-ingress" >&2
  exit 1
fi

# Block deployment when placeholder domains remain in ingress/origin config.
if rg -n "example\.com" "${rendered_file}" >/dev/null 2>&1; then
  echo "Preflight failed: placeholder domain 'example.com' found in rendered manifests." >&2
  echo "Update ingress hostnames and allowed origins in ${OVERLAY_PATH} before production deploy." >&2
  exit 1
fi

# Detect common placeholder repository/org tokens.
if rg -n "your-org|my-project|mytenancy|<set-in-secret-manager>" "${rendered_file}" >/dev/null 2>&1; then
  echo "Preflight failed: placeholder values found in rendered manifests." >&2
  echo "Replace placeholder image/domain/secret tokens before deployment." >&2
  exit 1
fi

# Ensure runtime secret exists and includes required key.
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null

if ! kubectl -n "${NAMESPACE}" get secret spot-scam-api-secrets >/dev/null 2>&1; then
  echo "Preflight failed: missing secret 'spot-scam-api-secrets' in namespace '${NAMESPACE}'." >&2
  echo "Create it before deployment." >&2
  exit 1
fi

secret_value="$(kubectl -n "${NAMESPACE}" get secret spot-scam-api-secrets -o jsonpath='{.data.GEMINI_API_KEY}')"
if [[ -z "${secret_value}" ]]; then
  echo "Preflight failed: secret 'spot-scam-api-secrets' is missing key GEMINI_API_KEY." >&2
  exit 1
fi

echo "Preflight checks passed for ${PROVIDER}/${STRATEGY} in namespace ${NAMESPACE}."
