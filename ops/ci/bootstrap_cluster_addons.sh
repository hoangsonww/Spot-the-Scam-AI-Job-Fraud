#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $0 [--ingress-namespace ingress-nginx] [--rollouts-namespace argo-rollouts] [--skip-ingress] [--skip-rollouts]
USAGE
}

INGRESS_NAMESPACE="ingress-nginx"
ROLLOUTS_NAMESPACE="argo-rollouts"
INSTALL_INGRESS=true
INSTALL_ROLLOUTS=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ingress-namespace)
      INGRESS_NAMESPACE="${2:-}"
      shift 2
      ;;
    --rollouts-namespace)
      ROLLOUTS_NAMESPACE="${2:-}"
      shift 2
      ;;
    --skip-ingress)
      INSTALL_INGRESS=false
      shift
      ;;
    --skip-rollouts)
      INSTALL_ROLLOUTS=false
      shift
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

if [[ "${INSTALL_INGRESS}" != "true" && "${INSTALL_ROLLOUTS}" != "true" ]]; then
  echo "Nothing to install: both ingress and rollouts are disabled." >&2
  exit 1
fi

for cmd in kubectl helm; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
done

kubectl get nodes >/dev/null

helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx --force-update >/dev/null
helm repo add argo https://argoproj.github.io/argo-helm --force-update >/dev/null
helm repo update >/dev/null

if [[ "${INSTALL_INGRESS}" == "true" ]]; then
  kubectl create namespace "${INGRESS_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
    --namespace "${INGRESS_NAMESPACE}" \
    --wait
fi

if [[ "${INSTALL_ROLLOUTS}" == "true" ]]; then
  kubectl create namespace "${ROLLOUTS_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  helm upgrade --install argo-rollouts argo/argo-rollouts \
    --namespace "${ROLLOUTS_NAMESPACE}" \
    --wait

  if ! kubectl get crd rollouts.argoproj.io >/dev/null 2>&1; then
    echo "Argo Rollouts CRD was not detected after installation." >&2
    exit 1
  fi
fi

echo "Cluster add-on bootstrap completed."
echo "  ingress installed:  ${INSTALL_INGRESS}"
echo "  rollouts installed: ${INSTALL_ROLLOUTS}"
echo "  ingress namespace:  ${INGRESS_NAMESPACE}"
echo "  rollouts namespace: ${ROLLOUTS_NAMESPACE}"
