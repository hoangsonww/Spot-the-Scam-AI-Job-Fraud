# Spot the Scam - Job Posting Fraud Detector

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.121-009688?logo=fastapi&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.57-FF6F61?logo=huggingface&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?logo=pandas&logoColor=white)
![Google Generative AI](https://img.shields.io/badge/Google_Generative_AI-0.13-4285F4?logo=google&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.15-3F4F75?logo=plotly&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4-00A0E9?logo=lightgbm&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-FF9900?logo=xgboost&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-3-2E2E2E?logo=optuna&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.12-13B6FF?logo=mlflow&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-1.15-000000?logo=onnx&logoColor=white)
![PyTest](https://img.shields.io/badge/PyTest-7-ED8B00?logo=pytest&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-20.10-2496ED?logo=docker&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=next.js&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3-38B2AC?logo=tailwind-css&logoColor=white)
![shadcn](https://img.shields.io/badge/shadcn-ui-000000?logo=shadcnui&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2.304.0-2088FF?logo=githubactions&logoColor=white)

Spot the Scam delivers an uncertainty-aware job-posting fraud detector with calibrated probabilities, a gray-zone review policy, and an interactive dashboard for analysis.

## Table of Contents

- [At a Glance](#at-a-glance)
- [What Makes It Practical](#what-makes-it-practical)
- [Training System Summary](#training-system-summary)
- [Serving and Review Summary](#serving-and-review-summary)
- [Explainability Summary](#explainability-summary)
- [Packaging and Deployment Summary](#packaging-and-deployment-summary)
- [Where to Go Next](#where-to-go-next)

## At a Glance

This repository is full-stack and artifact-driven:

- Offline training produces a clear artifact contract under `artifacts/`.
- Online serving loads those artifacts through `FraudPredictor`.
- The frontend consumes predictions, insights, and review workflows directly from the API.

## What Makes It Practical

The design prioritizes operational reliability:

- Precision-first defaults so alerts are actionable.
- Calibration and threshold optimization as first-class steps.
- A gray-zone policy so ambiguous cases route to review.
- Explanations attached to every prediction.
- Tracking utilities that support auditability and feedback loops.

## Training System Summary

The training pipeline (`src/spot_scam/pipeline/train.py`) provides a reproducible, config-driven workflow that includes:

- Automated data ingestion and normalization.
- Stratified train/validation/test splits with persisted parquet snapshots.
- TF-IDF plus engineered tabular features.
- Classical baselines (Logistic Regression, Linear SVM, LightGBM).
- XGBoost variant sweeps capped by configuration.
- Optional DistilBERT fine-tuning.
- Calibration, threshold optimization, and ensemble candidates.
- Strict artifact persistence and report generation.

Training runs emit artifacts and reports that the API and frontend read at runtime.

## Serving and Review Summary

The backend (`src/spot_scam/api/app.py`) exposes:

- Prediction endpoints (`/predict`, `/predict/single`).
- Insights endpoints (`/insights/*`).
- Review workflows (`/cases`, `/feedback`).
- A streaming chat assistant (`/chat`, requires `GEMINI_API_KEY`).

Prediction logs and reviewer feedback are stored under `tracking/` as partitioned parquet files.

## Explainability Summary

Every prediction includes an explanation payload:

- Classical models compute signed contributions per feature.
- Transformer models use gradient × input token attribution with an attention fallback.
- The frontend renders both positive and negative signals plus a summary sentence.

Interpretability artifacts are also generated under `experiments/tables/`.

## Packaging and Deployment Summary

The repository includes multiple packaging and deployment surfaces:

- ONNX export and MLflow pyfunc packaging.
- Docker and Docker Compose for local stacks.
- Kubernetes progressive delivery scaffolding under `ops/k8s/`.
- Load-testing and observability hooks under `ops/` and `scripts/`.

Transformer quantization is supported via `spot_scam.pipeline.quantize`.

## Where to Go Next

For deeper detail, use this map:

- Setup and operations: [INSTRUCTIONS.md](INSTRUCTIONS.md)
- Architecture and data flow: [ARCHITECTURE.md](ARCHITECTURE.md)
- Training strategy and trade-offs: [TRAINING_ANALYSIS.md](TRAINING_ANALYSIS.md)
- Metrics, plots, and diagnostics: [RESULTS.md](RESULTS.md)
- Model extension guide: [ADD_MODELS.md](ADD_MODELS.md)
- Deployment checklist: [docs/deployment_guide.md](docs/deployment_guide.md)
- Pipeline narrative: [docs/pipeline_walkthrough.md](docs/pipeline_walkthrough.md)
- Explainability details: [docs/explainability.md](docs/explainability.md)
- Optuna usage: [docs/optuna_quickstart.md](docs/optuna_quickstart.md) and [docs/optuna_tuning.md](docs/optuna_tuning.md)
