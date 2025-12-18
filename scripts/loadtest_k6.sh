#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v k6 >/dev/null 2>&1; then
  echo "k6 is required. Install from https://k6.io/docs/getting-started/installation/"
  exit 1
fi

echo "Running k6 smoke against ${API_BASE} ..."
API_BASE="${API_BASE}" k6 run "${ROOT_DIR}/ops/observability/k6-smoke.js"
