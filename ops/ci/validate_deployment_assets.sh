#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

required_files=(
  "DEPLOYMENT.md"
  "Jenkinsfile"
  "scripts/deploy_multi_cloud.sh"
  "scripts/apply_k8s_overlay.sh"
  "ops/ci/preflight_deploy_checks.sh"
  "ops/k8s/base/secret-api.example.yaml"
  "aws/README.md"
  "azure/README.md"
  "gcp/README.md"
  "oci/README.md"
)

for f in "${required_files[@]}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing required deployment asset: ${f}" >&2
    exit 1
  fi
done

bash -n scripts/deploy_multi_cloud.sh
bash -n scripts/apply_k8s_overlay.sh
bash -n ops/ci/preflight_deploy_checks.sh

# Validate base overlays
for overlay in \
  ops/k8s/overlays/staging-canary \
  ops/k8s/overlays/prod-bluegreen; do
  kubectl kustomize "${overlay}" >/dev/null
  echo "Validated overlay render: ${overlay}"
done

# Validate provider overlays
for provider in aws azure gcp oci; do
  for strategy in canary bluegreen; do
    overlay="${provider}/k8s/${strategy}"
    kubectl kustomize "${overlay}" >/dev/null
    echo "Validated overlay render: ${overlay}"
  done
done

# Ensure old cross-directory file refs do not come back.
if rg -n "\.\./\.\./base/rollout-(canary|bluegreen)\.yaml" ops/k8s/overlays >/dev/null 2>&1; then
  echo "Found unsupported kustomize rollout file references in ops overlays." >&2
  exit 1
fi

# Prevent accidental reintroduction of placeholder secret into auto-applied base.
if rg -n "secret-api.yaml" ops/k8s/base/kustomization.yaml >/dev/null 2>&1; then
  echo "Base kustomization must not auto-apply secret-api.yaml. Use secret-api.example.yaml + out-of-band secret creation." >&2
  exit 1
fi

echo "Deployment asset validation passed."
