# Spot the Scam - Setup, Training, and Operations Guide

This guide is the “do everything” manual for the repository. It covers local setup, training, serving, the frontend dashboard, review workflows, tuning, export, and containerized execution.

If you want a shorter entry point, start in `README.md` and come back here when you need deeper control.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Repository Setup (Local)](#repository-setup-local)
- [Data Acquisition](#data-acquisition)
- [Environment Variables and Configuration](#environment-variables-and-configuration)
- [Training Workflows](#training-workflows)
- [Artifacts and Outputs](#artifacts-and-outputs)
- [Serving the FastAPI Backend](#serving-the-fastapi-backend)
- [Frontend Dashboard (Next.js)](#frontend-dashboard-nextjs)
- [Human-in-the-Loop Review and Feedback](#human-in-the-loop-review-and-feedback)
- [AI Chat Assistant (Gemini)](#ai-chat-assistant-gemini)
- [Hyperparameter Tuning (Optuna)](#hyperparameter-tuning-optuna)
- [ONNX and MLflow Export](#onnx-and-mlflow-export)
- [Docker and Devcontainer](#docker-and-devcontainer)
- [Ops and Load Testing Hooks](#ops-and-load-testing-hooks)
- [Testing and Code Quality](#testing-and-code-quality)
- [Troubleshooting](#troubleshooting)
- [Cleanup and Regeneration](#cleanup-and-regeneration)
- [High-Value Make Targets](#high-value-make-targets)

## Prerequisites

The project spans Python ML, a FastAPI backend, and a Next.js frontend.

### Required tooling

- Python 3.12+
- Node.js 18+ and npm 9+
- Git

### Strongly recommended

- A virtual environment (`venv` or similar)
- Docker and Docker Compose
- A GPU for transformer fine-tuning (optional but helpful)

### Notes on versions

- `scikit-learn` is pinned to `1.7.2` in `pyproject.toml`.
- The training and inference stack expects dependencies compatible with the versions in `pyproject.toml`.

## Repository Setup (Local)

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

This installs both runtime and development tooling (pytest, black, ruff, mypy).

## Data Acquisition

The repository already includes the Kaggle dataset files under `data/`. You can re-download them if needed.

### Optional: re-download with Kaggle CLI

```bash
./scripts/download_data.py
```

This script:

- Verifies the Kaggle CLI is installed
- Downloads the dataset archive
- Extracts CSVs into `data/`

## Environment Variables and Configuration

Configuration is split between YAML defaults and environment variables.

### YAML configuration

The canonical configuration lives at `configs/defaults.yaml`. Key settings include:

- Data fields and drop columns
- TF-IDF and tabular feature settings
- Classical and transformer model parameters
- Calibration and evaluation settings
- Gray-zone policy width and labels

### Backend environment variables (`.env`)

Create a root-level `.env` file for backend configuration:

```bash
# Required only for the /chat endpoint
GEMINI_API_KEY=your_key_here

# Comma-separated CORS allowlist
SPOT_SCAM_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Use quantized transformer weights when available
SPOT_SCAM_USE_QUANTIZED=0

# MLflow tracking directory/URI
MLFLOW_TRACKING_URI=file:///app/mlruns

# Feedback integration switch for training
# USE_FEEDBACK=1
```

### Frontend environment variables (`frontend/.env.local`)

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

The frontend will fall back to demo/mock behavior when it cannot connect to the backend.

## Training Workflows

All training flows are orchestrated by `src/spot_scam/pipeline/train.py`.

### Fastest path: classical-only training

```bash
PYTHONPATH=src python -m spot_scam.pipeline.train --skip-transformer
```

### Full training: classical + transformer

```bash
PYTHONPATH=src python -m spot_scam.pipeline.train
```

### What the training pipeline actually does

The pipeline is configuration-driven and performs the following steps:

1. Load and merge raw CSVs (`data/ingest.py`).
2. Clean and normalize text fields (`data/preprocess.py`).
3. Stratify into train/validation/test splits and persist indices.
4. Persist parquet splits to `data/processed/`.
5. Build TF-IDF and tabular feature bundles (`features/builders.py`).
6. Train classical candidates and run XGBoost variant sweeps.
7. Optionally fine-tune DistilBERT (`models/transformer.py`).
8. Calibrate probabilities and optimize thresholds.
9. Build ensemble candidates when beneficial.
10. Evaluate on the held-out test set.
11. Persist artifacts, figures, tables, and `experiments/report.md`.
12. Benchmark latency/throughput and append run records.
13. Attempt ONNX and MLflow export when enabled.

## Artifacts and Outputs

After a training run, the repository is populated with model and analysis assets that the API and frontend consume.

### Primary artifact locations

| Location | Purpose |
|----------|---------|
| `artifacts/model.joblib` | Calibrated runtime model |
| `artifacts/base_model.joblib` | Uncalibrated base estimator (when available) |
| `artifacts/metadata.json` | Metrics, threshold, gray-zone policy, model identity |
| `artifacts/config_used.yaml` | Frozen configuration snapshot used for training |
| `artifacts/features/` | TF-IDF vectorizer, tabular scaler, tabular feature names |
| `artifacts/test_predictions.csv` | Test split probabilities, labels, and decisions |
| `artifacts/xgboost_variants/` | Per-variant XGBoost artifacts |
| `artifacts/transformer/` | Transformer checkpoints, tokenizer, and quantized weights |

### Analysis outputs

| Location | Purpose |
|----------|---------|
| `experiments/figs/` | Curves, confusion matrix, distributions, and benchmark plots |
| `experiments/tables/` | Metrics summaries, slices, thresholds, token analytics, benchmarks |
| `experiments/report.md` | Auto-generated training summary |
| `tracking/runs.csv` | Append-only run metadata log |

## Serving the FastAPI Backend

The backend API is defined in `src/spot_scam/api/app.py` and loads artifacts via a cached `FraudPredictor`.

### Run locally

```bash
source .venv/bin/activate
PYTHONPATH=src uvicorn spot_scam.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Core endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness check and basic model info |
| `GET /metadata` | Thresholds, gray-zone policy, and metrics |
| `GET /models` | Recent tracked model candidates |
| `POST /predict` | Batch scoring |
| `POST /predict/single` | Single scoring |
| `GET /insights/token-importance` | Token coefficient summary |
| `GET /insights/token-frequency` | Token frequency deltas |
| `GET /insights/threshold-metrics` | Validation threshold sweep |
| `GET /insights/latency` | Latency and throughput summaries |
| `GET /insights/slice-metrics` | Slice-based performance summaries |
| `GET /cases` | Review queue sampling |
| `POST /feedback` | Human review feedback |
| `POST /chat` | Streaming Gemini responses (requires API key) |

### Quick API check

```bash
curl http://localhost:8000/health
```

## Frontend Dashboard (Next.js)

The frontend lives in `frontend/` and is built with Next.js App Router, Tailwind CSS, and shadcn/ui components.

### Install and run

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000`.

### Key pages

- `/`: scoring and model rationale
- `/review`: review queue triage and feedback submission
- `/chat`: streaming AI assistant

## Human-in-the-Loop Review and Feedback

The review loop is implemented in the backend tracking modules and exposed via `/cases` and `/feedback`.

### 1. Serve the review-capable API

```bash
make serve-queue
```

### 2. Generate predictions

Use the dashboard or API endpoints. Each prediction is logged under `tracking/predictions/` with masked payloads.

### 3. Triage cases in the UI

Go to `http://localhost:3000/review`.

### 4. (Optional) sample uncertain cases

```bash
PYTHONPATH=src python scripts/sample_uncertain.py --policy entropy --limit 200
```

This writes `experiments/tables/active_sample.csv`, which the review queue can include.

### 5. Retrain with reviewer overrides

```bash
USE_FEEDBACK=1 PYTHONPATH=src python -m spot_scam.pipeline.train
```

Or:

```bash
make retrain-with-feedback
```

When enabled, feedback overrides labels by matching on `text_hash`.

## AI Chat Assistant (Gemini)

The `/chat` endpoint streams answers from Gemini and can optionally auto-run the fraud predictor when it detects a job posting.

### Requirements

- Install dependencies via `pip install -e '.[dev]'`
- Set `GEMINI_API_KEY` in `.env`
- Restart the backend after setting the key

### Behavior summary

- The backend first classifies the message as a potential job posting.
- If it looks like a job post and no explicit context is supplied, it auto-scores the text.
- The fraud prediction and explanation are injected into the assistant prompt.
- The response streams back as Server-Sent Events (SSE).

## Hyperparameter Tuning (Optuna)

Optuna tuning is supported via `scripts/tune_with_optuna.py` and the tuning module in `src/spot_scam/tuning/optuna_tuner.py`.

### Tune logistic regression

```bash
PYTHONPATH=src python scripts/tune_with_optuna.py --model-type logistic --n-trials 20
```

### Tune linear SVM

```bash
PYTHONPATH=src python scripts/tune_with_optuna.py --model-type svm --n-trials 30
```

### Visualize studies

```bash
OMP_NUM_THREADS=1 optuna-dashboard sqlite:///optuna_study.db --server wsgiref --host 127.0.0.1 --port 8080
```

Then select the desired study in the dashboard.

## ONNX and MLflow Export

The training pipeline attempts to export models to ONNX and log an MLflow pyfunc model when MLflow is enabled in config.
For the full MLOps lifecycle and artifact contract, see [MLOPS.md](MLOPS.md).

### MLflow notes

- MLflow output defaults to `mlruns/`
- Export is best-effort; failures log warnings without stopping training
- The pyfunc packaging includes preprocessing and gray-zone policy logic

### Serve an MLflow model locally

```bash
mlflow models serve --env-manager local -m runs:/<RUN_ID>/model -p 8080
```

## Docker and Devcontainer

Docker support is first-class and useful for end-to-end local demos.

### Docker Compose runtime

```bash
docker compose build
docker compose up -d
```

- API: `http://localhost:8000`
- Dashboard: `http://localhost:3000`

The Compose file mounts `configs/`, `artifacts/`, `experiments/`, `mlruns/`, and `data/` for persistence.

### Run training inside the API container

```bash
docker compose exec api bash
PYTHONPATH=src python -m spot_scam.pipeline.train --skip-transformer
```

### Devcontainer (VS Code)

A devcontainer workflow is documented in `DOCKER.md` and is a good option when you want a standardized environment.

## Ops and Load Testing Hooks

The repository includes additional operational assets:

- Kubernetes manifests under `ops/k8s/` with base and overlay structures.
- A Tekton pipeline example under `ops/ci/tekton-pipeline.yaml`.
- Load testing helpers:
  - `ops/observability/k6-smoke.js`
  - `scripts/loadtest_k6.sh`

These are intended as deployment scaffolding and can be adapted to your environment.

## Testing and Code Quality

The repository includes both Python and frontend quality checks.

### Python checks

```bash
make check-all
make test
```

### Frontend checks

```bash
make frontend-check
```

### Formatters and linters

- Python: Black and Ruff
- TypeScript: Prettier and ESLint

## Troubleshooting

Common issues and their fastest fixes.

### Backend fails to start due to missing artifacts

The API expects trained artifacts in `artifacts/`. Fix by training:

```bash
PYTHONPATH=src python -m spot_scam.pipeline.train --skip-transformer
```

### `ModuleNotFoundError: spot_scam`

Ensure you installed the package and set `PYTHONPATH=src` when running modules directly.

### Frontend shows demo data or “backend offline”

- Confirm the backend is running on port 8000
- Confirm `NEXT_PUBLIC_API_BASE_URL` points to the correct backend
- Check `http://localhost:8000/health`

### Chat endpoint errors

- Ensure `GEMINI_API_KEY` is set in `.env`
- Restart the backend after setting the key
- Verify `google-generativeai` is installed via the project dependencies

### Training is slow

- Use `--skip-transformer` for fast iteration
- Reduce XGBoost variants via config (`models.xgboost.max_variants`)

## Cleanup and Regeneration

Prefer safe cleanup via `make clean` for cache files.

If you intentionally want to clear generated artifacts and reports, remove contents under these directories:

- `artifacts/`
- `experiments/`
- `data/processed/`
- `tracking/`
- `mlruns/`

Then run training again.

## High-Value Make Targets

The Makefile is the easiest operational interface for day-to-day use.

### Training and serving

| Command | Purpose |
|--------|---------|
| `make install` | Install Python dependencies |
| `make train` | Full training pipeline |
| `make train-fast` | Classical-only training |
| `make serve` | Run the API |
| `make serve-queue` | Run the API with review workflows |
| `make review-sample` | Sample uncertain predictions |
| `make retrain-with-feedback` | Train with feedback overrides |
| `make quantize-transformer` | Quantize transformer weights |

### Quality and testing

| Command | Purpose |
|--------|---------|
| `make check-all` | Format check, lint, and type-check (Python) |
| `make test` | Run pytest |
| `make frontend-check` | Format check, lint, and type-check (frontend) |
| `make clean` | Remove cache and build artifacts |

### What to read next

For architectural context and “why things are the way they are,” see [ARCHITECTURE.md](ARCHITECTURE.md) and [TRAINING_ANALYSIS.md](TRAINING_ANALYSIS.md).

Feel free to ask questions or open issues if anything is unclear!
