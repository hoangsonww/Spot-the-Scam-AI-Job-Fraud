# Spot the Scam

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.121-009688?logo=fastapi&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.57-FF6F61?logo=huggingface&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=next.js&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3-38B2AC?logo=tailwind-css&logoColor=white)

Spot the Scam delivers an uncertainty-aware job-posting fraud detector with calibrated probabilities, a gray-zone review policy, and an interactive dashboard for analysis.

> [!NOTE]
> Intelligent fraud triage for job postings - built for transparency and speed! 

## Features
- **Reproducible pipeline**: Config-driven splitting, TF-IDF + tabular features, classical baselines, DistilBERT fine-tuning, and strict artifact persistence.
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
- Non-quantized models remain the default.

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for setup and usage details.
