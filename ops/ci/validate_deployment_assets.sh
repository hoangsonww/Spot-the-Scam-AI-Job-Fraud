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
  "ops/ci/bootstrap_cluster_addons.sh"
  "ops/ci/bootstrap_argocd.sh"
  "ops/ci/argocd_sync_wait.sh"
  "ops/k8s/base/secret-api.example.yaml"
  "ops/argo/README.md"
  "ops/argo/base/kustomization.yaml"
  "ops/argo/projects/spot-scam-project.yaml"
  "ops/argo/apps/staging/application.example.yaml"
  "ops/argo/apps/prod/application.example.yaml"
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
bash -n ops/ci/bootstrap_cluster_addons.sh
bash -n ops/ci/bootstrap_argocd.sh
bash -n ops/ci/argocd_sync_wait.sh

for provider in aws azure gcp oci; do
  for tf_file in main.tf variables.tf versions.tf outputs.tf; do
    if [[ ! -f "${provider}/terraform/${tf_file}" ]]; then
      echo "Missing Terraform file: ${provider}/terraform/${tf_file}" >&2
      exit 1
    fi
  done

  if ! rg -n 'output "configure_kubectl"' "${provider}/terraform/outputs.tf" >/dev/null 2>&1; then
    echo "Missing configure_kubectl output in ${provider}/terraform/outputs.tf" >&2
    exit 1
  fi
done

if command -v terraform >/dev/null 2>&1; then
  terraform fmt -check -recursive aws/terraform azure/terraform gcp/terraform oci/terraform >/dev/null
  echo "Terraform formatting check passed."
else
  echo "terraform not found; skipping terraform fmt check."
fi

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

# Validate Argo CD project overlays
for overlay in \
  ops/argo/base \
  ops/argo/overlays/staging \
  ops/argo/overlays/prod; do
  kubectl kustomize "${overlay}" >/dev/null
  echo "Validated Argo overlay render: ${overlay}"
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

# Ensure no placeholder GitOps values leak into non-example Argo manifests.
if rg -n "github.com/<org>/<repo>" ops/argo --glob '*.yaml' --glob '!**/*.example.yaml' >/dev/null 2>&1; then
  echo "Found placeholder values in non-example Argo manifests." >&2
  exit 1
fi

# Keep Argo project manifest copies in sync.
if ! cmp -s ops/argo/base/spot-scam-project.yaml ops/argo/projects/spot-scam-project.yaml; then
  echo "Argo project manifests are out of sync between base/ and projects/." >&2
  exit 1
fi

echo "Deployment asset validation passed."
