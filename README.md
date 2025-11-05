# Spot the Scam - Job Posting Fraud Detector

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.121-009688?logo=fastapi&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.57-FF6F61?logo=huggingface&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.15-3F4F75?logo=plotly&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4-00A0E9?logo=lightgbm&logoColor=white)
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
- **Explainability & monitoring**: Token importances and frequency gaps, SHAP summaries, threshold sweeps, probability regressions, and latency benchmarks.
- **Serving + UX**: FastAPI service exposing prediction/metadata/insights endpoints and a Next.js + shadcn UI for triaging and reporting.
- **Container-ready**: Dockerfile, docker compose, and VS Code devcontainer for reproducible local or cloud environments.

## Outputs
- `artifacts/` — models, vectorizers, calibration metadata, final predictions.
- `experiments/figs/` — PR & calibration curves, confusion matrices, score distributions, threshold vs metric sweeps, latency plots.
- `experiments/tables/` — metrics summary, slice reports, token analyses, threshold metrics, latency benchmarks.
- `INSTRUCTIONS.md` — step-by-step setup, training, serving, and frontend guidance.
- `ARCHITECTURE.md` — detailed system architecture with flow diagrams.

## Quantization (optional)
- To create an int8 dynamic-quantized transformer checkpoint: `make quantize-transformer`
- Serve the quantized model by setting `SPOT_SCAM_USE_QUANTIZED=1` before starting the API.
- Configure CORS (if hosting the UI elsewhere) with `SPOT_SCAM_ALLOWED_ORIGINS="https://your-domain.com,http://localhost:3000"`.
- Non-quantized models remain the default.

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for setup and usage details.

Project licensed under the MIT License. See [LICENSE](LICENSE) for details.
