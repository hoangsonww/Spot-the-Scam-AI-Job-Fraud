#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $0 --app <app-name> [--server <argocd-server>] [--timeout-sec 600]
USAGE
}

APP_NAME=""
ARGOCD_SERVER=""
TIMEOUT_SEC=600

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app)
      APP_NAME="${2:-}"
      shift 2
      ;;
    --server)
      ARGOCD_SERVER="${2:-}"
      shift 2
      ;;
    --timeout-sec)
      TIMEOUT_SEC="${2:-}"
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

if [[ -z "${APP_NAME}" ]]; then
  echo "--app is required" >&2
  exit 1
fi

if ! command -v argocd >/dev/null 2>&1; then
  echo "argocd CLI is required" >&2
  exit 1
fi

if [[ -n "${ARGOCD_SERVER}" ]]; then
  argocd app sync "${APP_NAME}" --server "${ARGOCD_SERVER}" --timeout "${TIMEOUT_SEC}"
  argocd app wait "${APP_NAME}" --server "${ARGOCD_SERVER}" --health --sync --timeout "${TIMEOUT_SEC}"
else
  argocd app sync "${APP_NAME}" --timeout "${TIMEOUT_SEC}"
  argocd app wait "${APP_NAME}" --health --sync --timeout "${TIMEOUT_SEC}"
fi

echo "Argo CD app ${APP_NAME} is synced and healthy."
