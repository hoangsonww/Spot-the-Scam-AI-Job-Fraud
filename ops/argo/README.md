# Argo CD GitOps Assets

This directory contains Argo CD project and application examples for Spot the Scam.

## Contents

- `projects/spot-scam-project.yaml`: Argo CD AppProject for repository deployments.
- `apps/staging/application.example.yaml`: staging application example.
- `apps/prod/application.example.yaml`: production application example.
- `base/` and `overlays/`: kustomize layers for Argo project bootstrap.

## Bootstrap

Use the bootstrap script to install Argo CD and create an environment/provider-specific app:

```bash
./ops/ci/bootstrap_argocd.sh \
  --env staging \
  --provider aws \
  --repo-url https://github.com/<org>/<repo>.git \
  --revision main
```

Note: this repository currently uses a fixed Argo CD namespace (`argocd`) in manifests and bootstrap flow.
The bootstrap script injects the provided `--repo-url` into both the AppProject and Application.

## Sync and Wait

```bash
./ops/ci/argocd_sync_wait.sh --app spot-scam-staging-aws --timeout-sec 900
```

## Important

- Example application manifests are templates and contain placeholder repo values.
- Runtime rollout image patching via Jenkins remains compatible through `ignoreDifferences` on rollout container image fields.
