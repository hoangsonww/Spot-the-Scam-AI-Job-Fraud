# Spot the Scam - Job Posting Fraud Detector

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.121-009688?logo=fastapi&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.57-FF6F61?logo=huggingface&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.15-3F4F75?logo=plotly&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4-00A0E9?logo=lightgbm&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.12-13B6FF?logo=mlflow&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-1.15-000000?logo=onnx&logoColor=white)
![PyTest](https://img.shields.io/badge/PyTest-7-ED8B00?logo=pytest&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-20.10-2496ED?logo=docker&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=next.js&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3-38B2AC?logo=tailwind-css&logoColor=white)

Spot the Scam delivers an uncertainty-aware job-posting fraud detector with calibrated probabilities, a gray-zone review policy, and an interactive dashboard for analysis.

> [!NOTE]
> Intelligent fraud triage for job postings - built for transparency and speed! 

## Features
- **Reproducible pipeline**: Config-driven ingestion (merges both Kaggle CSV snapshots), stratified splitting, TF-IDF + tabular features, classical baselines, DistilBERT fine-tuning, and strict artifact persistence.
- **Uncertainty-aware decisions**: Validation-driven calibration (Platt/isotonic), gray-zone banding, slice metrics, and reliability plots.
- **Explainability & monitoring**: Per-prediction natural-language rationales with top contributing signals, token importances and frequency gaps, SHAP summaries, threshold sweeps, probability regressions, and latency benchmarks.
- **Serving + UX**: FastAPI service exposing prediction/metadata/insights endpoints and a Next.js + shadcn UI for triaging and reporting.
- **Human-in-the-loop feedback**: Review queue for gray-zone predictions, feedback logging, and retraining hooks so human judgements continuously improve calibration.
- **Container-ready**: Dockerfile, docker compose, and VS Code devcontainer for reproducible local or cloud environments (see [DOCKER.md](DOCKER.md) for local commands and CI that publishes model/API/frontend images to GHCR).

## Outputs
- `artifacts/` - models, vectorizers, calibration metadata, final predictions.
- `experiments/figs/` - PR & calibration curves, confusion matrices, score distributions, threshold vs metric sweeps, latency plots.
- `experiments/tables/` - metrics summary, slice reports, token analyses, threshold metrics, latency benchmarks.
- `INSTRUCTIONS.md` - step-by-step setup, training, serving, and frontend guidance.
- `ARCHITECTURE.md` - detailed system architecture with flow diagrams.

## Explainability & Model Packaging

- Every prediction includes a **local explanation**: the API surfaces the top supporting/opposing features (tokens and tabular signals) as well as the intercept so reviewers can understand the decision instantly. The Next.js dashboard renders these insights in the “Decision rationale” card.
- Classical winners (logistic regression, etc.) export linear contributions directly; transformer winners currently emit a high-level summary placeholder.

### ONNX + MLflow

- The training pipeline automatically converts the selected model to ONNX and logs a ready-to-serve MLflow pyfunc package (vectorizer, scaler, ONNX graph, metadata, and decision policy).
- Check `mlruns/` for runs and registered models. Use `mlflow models serve -m runs:/<RUN_ID>/model` to spin up a local pyfunc service with the same behavior as the FastAPI endpoint.
- Running on GPU? Install `onnxruntime-gpu` inside your virtualenv (matching the CPU package version) so ONNX Runtime registers CUDA devices. Otherwise set `ORT_DISABLE_DEVICE_DISCOVERY=1` to silence the CPU-only warning.

### Quantization (optional)
- To create an int8 dynamic-quantized transformer checkpoint: `make quantize-transformer`
- Serve the quantized model by setting `SPOT_SCAM_USE_QUANTIZED=1` before starting the API.
- Non-quantized models remain the default.

### Human-in-the-Loop Review (HITL)

- Run the API and review queue: `make serve-queue` (FastAPI on port 8000) and open `http://localhost:3000/review`.
- Populate the queue with high-uncertainty cases via `make review-sample` (writes `experiments/tables/active_sample.csv`) or by using the live dashboard.
- Reviewers submit confirmations/overrides; feedback is appended under `tracking/feedback/date=*/`.
- Retrain with feedback applied by running `make retrain-with-feedback` (or `USE_FEEDBACK=1 PYTHONPATH=src python -m spot_scam.pipeline.train`). The pipeline produces comparative tables:
  - `experiments/tables/metrics_with_feedback.csv`
  - `experiments/tables/slice_metrics_baseline.csv`
  - `experiments/tables/slice_metrics_feedback_delta.csv`
- The review UI and nav badge automatically reflect queue size via `/cases` and `/feedback` endpoints.

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for setup and usage details. Visit [ARCHITECTURE.md](ARCHITECTURE.md) for system design and data flow diagrams.

Project licensed under the MIT License. See [LICENSE](LICENSE) for details.
