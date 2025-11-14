# Spot the Scam - Job Posting Fraud Detector

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.121-009688?logo=fastapi&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.57-FF6F61?logo=huggingface&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?logo=pandas&logoColor=white)
![Google Generative AI](https://img.shields.io/badge/Google_Generative_AI-0.13-4285F4?logo=google&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.15-3F4F75?logo=plotly&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4-00A0E9?logo=lightgbm&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.12-13B6FF?logo=mlflow&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-1.15-000000?logo=onnx&logoColor=white)
![PyTest](https://img.shields.io/badge/PyTest-7-ED8B00?logo=pytest&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-20.10-2496ED?logo=docker&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=next.js&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3-38B2AC?logo=tailwind-css&logoColor=white)
![shadcn](https://img.shields.io/badge/shadcn-ui-000000?logo=shadcnui&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2.304.0-2088FF?logo=githubactions&logoColor=white)

Spot the Scam delivers an uncertainty-aware job-posting fraud detector with calibrated probabilities, a gray-zone review policy, and an interactive dashboard for analysis.

> [!NOTE]
> Intelligent fraud triage for job postings - built for transparency and speed! 

## Features
- **Reproducible pipeline**: Config-driven ingestion (merges both Kaggle CSV snapshots), stratified splitting, TF-IDF + tabular features, classical baselines, DistilBERT fine-tuning, and strict artifact persistence.
- **Uncertainty-aware decisions**: Validation-driven calibration (Platt/isotonic), gray-zone banding, slice metrics, and reliability plots.
- **Explainability & monitoring**: Per-prediction natural-language rationales with top contributing signals, gradient-based transformer token importances (with attention fallback), token frequency gaps, SHAP summaries, threshold sweeps, probability regressions, and latency benchmarks.
- **Serving + UX**: FastAPI service exposing prediction/metadata/insights endpoints and a Next.js + shadcn UI for triaging and reporting.
- **Human-in-the-loop feedback**: Review queue for gray-zone predictions, feedback logging, and retraining hooks so human judgements continuously improve calibration.
- **Container-ready**: Dockerfile, docker compose, and VS Code devcontainer for reproducible local or cloud environments (see [DOCKER.md](DOCKER.md) for local commands and CI that publishes model/API/frontend images to GHCR).

## Outputs & File Structure
- `artifacts/` - models, vectorizers, calibration metadata, final predictions.
- `experiments/figs/` - PR & calibration curves, confusion matrices, score distributions, threshold vs metric sweeps, latency plots.
- `experiments/tables/` - metrics summary, slice reports, token analyses, threshold metrics, latency benchmarks.
- `src/` - source code for pipeline, model, API, and frontend.
- `tests/` - unit and integration tests with PyTest.
- `tracking/` - MLflow experiment runs and artifacts.
- `experiments/report.md` - markdown report summarizing key results and insights.
- `frontend/` - Next.js + shadcn UI + TailwindCSS source code.
- `data/` - raw and processed datasets.
- `INSTRUCTIONS.md` - step-by-step setup, training, serving, and frontend guidance.
- `ARCHITECTURE.md` - detailed system architecture with flow diagrams.

## Explainability & Model Packaging

- Every prediction includes a **local explanation**: the API surfaces the top supporting/opposing features (tokens and tabular signals) as well as the intercept so reviewers can understand the decision instantly. The Next.js dashboard renders these insights in the “Decision rationale” card.
- Classical winners (logistic regression, etc.) export linear contributions directly; transformer winners surface gradient-derived token contributions (falling back to attention scores when gradients are unavailable, e.g., quantized mode).

### ONNX + MLflow

- The training pipeline automatically converts the selected model to ONNX and logs a ready-to-serve MLflow pyfunc package (vectorizer, scaler, ONNX graph, metadata, and decision policy).

### Quantization (optional)
- Quantization is supported for classical models via `ONNXRuntime` optimizations. Enable with `QUANTIZE_MODEL=1 make train` or `QUANTIZE_MODEL=1 PYTHONPATH=src python -m spot_scam.pipeline.train`.
- Quantization helps reduce model size and inference latency with minimal accuracy loss, suitable for deployment scenarios with resource constraints.
- All reported benchmarks were produced on a workstation with an RTX 3070 Ti (8 GB) running CUDA-enabled PyTorch; expect longer transformer fine-tuning times on smaller GPUs or CPU-only boxes.

## Human-in-the-Loop Review (HITL)

- Cases are automatically added to the review queue. User can submit feedback via the API or frontend.
- Feedback is logged to `artifacts/hitl_feedback.csv` for retraining and calibration updates. 
- Subsequent pipeline runs can incorporate this feedback to refine model performance and decision thresholds, ensuring continuous improvement based on real-world inputs.

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for setup and usage details. Visit [ARCHITECTURE.md](ARCHITECTURE.md) for system design and data flow diagrams.

Project licensed under the MIT License. See [LICENSE](LICENSE) for details.
