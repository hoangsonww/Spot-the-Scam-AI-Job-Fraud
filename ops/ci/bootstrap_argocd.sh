#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $0 --env <staging|prod> --repo-url <git-url> [--revision <git-ref>] [--provider <core|aws|azure|gcp|oci>] [--namespace argocd]
USAGE
}

ENVIRONMENT=""
REPO_URL=""
REVISION="main"
PROVIDER="core"
NAMESPACE="argocd"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENVIRONMENT="${2:-}"
      shift 2
      ;;
    --repo-url)
      REPO_URL="${2:-}"
      shift 2
      ;;
    --revision)
      REVISION="${2:-}"
      shift 2
      ;;
    --provider)
      PROVIDER="${2:-}"
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

if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "prod" ]]; then
  echo "Invalid or missing --env value. Use staging or prod." >&2
  exit 1
fi

if [[ -z "$REPO_URL" ]]; then
  echo "--repo-url is required" >&2
  exit 1
fi

if [[ "$PROVIDER" != "core" && "$PROVIDER" != "aws" && "$PROVIDER" != "azure" && "$PROVIDER" != "gcp" && "$PROVIDER" != "oci" ]]; then
  echo "Invalid --provider value. Use core|aws|azure|gcp|oci." >&2
  exit 1
fi

if [[ "$NAMESPACE" != "argocd" ]]; then
  echo "This repository currently uses a fixed Argo CD namespace: argocd." >&2
  echo "Received --namespace ${NAMESPACE}. Update Argo manifest namespaces before using a custom namespace." >&2
  exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required" >&2
  exit 1
fi

if ! command -v helm >/dev/null 2>&1; then
  echo "helm is required" >&2
  exit 1
fi

if [[ "$PROVIDER" == "core" ]]; then
  if [[ "$ENVIRONMENT" == "staging" ]]; then
    APP_PATH="ops/k8s/overlays/staging-canary"
  else
    APP_PATH="ops/k8s/overlays/prod-bluegreen"
  fi
else
  if [[ "$ENVIRONMENT" == "staging" ]]; then
    APP_PATH="${PROVIDER}/k8s/canary"
  else
    APP_PATH="${PROVIDER}/k8s/bluegreen"
  fi
fi

APP_NAME="spot-scam-${ENVIRONMENT}-${PROVIDER}"

kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

helm repo add argo https://argoproj.github.io/argo-helm --force-update >/dev/null
helm repo update >/dev/null
helm upgrade --install argocd argo/argo-cd --namespace "${NAMESPACE}" --wait

cat <<EOF_PROJECT | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: spot-scam
  namespace: ${NAMESPACE}
spec:
  description: Spot the Scam deployment project
  sourceRepos:
    - ${REPO_URL}
  destinations:
    - namespace: spot-scam
      server: https://kubernetes.default.svc
  clusterResourceWhitelist:
    - group: ""
      kind: Namespace
  namespaceResourceWhitelist:
    - group: '*'
      kind: '*'
  orphanedResources:
    warn: true
EOF_PROJECT

cat <<EOF_APP | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ${APP_NAME}
  namespace: ${NAMESPACE}
spec:
  project: spot-scam
  source:
    repoURL: ${REPO_URL}
    targetRevision: ${REVISION}
    path: ${APP_PATH}
  destination:
    server: https://kubernetes.default.svc
    namespace: spot-scam
  syncPolicy:
    syncOptions:
      - CreateNamespace=true
      - PruneLast=true
      - ApplyOutOfSyncOnly=true
  ignoreDifferences:
    - group: argoproj.io
      kind: Rollout
      jsonPointers:
        - /spec/template/spec/containers
  revisionHistoryLimit: 20
EOF_APP

echo "Argo CD bootstrap completed for ${ENVIRONMENT}/${PROVIDER}."
echo "Application created: ${APP_NAME}"
