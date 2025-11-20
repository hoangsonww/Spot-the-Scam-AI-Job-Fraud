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

> [!NOTE]
> Intelligent fraud triage for job postings - built for transparency and speed!

## Training Pipeline

The training pipeline provides a reproducible, config-driven workflow that includes:
- Automated data ingestion with stratified splitting
- Feature engineering: TF-IDF vectorization + tabular features
- Classical baselines (Logistic Regression, Linear SVM, etc.)
- XGBoost hyperparameter sweeps
- Weighted ensemble models
- DistilBERT fine-tuning for advanced text classification
- Strict artifact persistence

All training runs are fully configurable and produce versioned artifacts for downstream deployment.

## Hyperparameter Optimization

Optuna support that enables Bayesian hyperparameter tuning with:
- Intelligent search strategies
- Early stopping and pruning
- Multi-objective optimization capabilities

## Uncertainty-Aware Predictions

- Validation-driven calibration using Platt scaling/isotonic regression
- Predictions within an uncertainty range (gray-zone) are flagged for human review
- Slice-based metrics to analyze performance across different data segments
- Reliability plots to visualize calibration quality

## Human-in-the-Loop Review (HITL)

- Cases that the user submits are automatically added to the review queue. User can submit feedback via the API or frontend.
- Feedback is persisted for retraining and calibration updates.
- Subsequent pipeline runs can incorporate this feedback to refine model performance and decision thresholds, ensuring continuous improvement based on real-world inputs.

## Explainability

Every prediction includes an **explanation** that helps reviewers understand the decision:
- The FastAPI server surfaces the top supporting/opposing features (tokens and tabular signals) as well as the intercept
- The Next.js dashboard renders these insights in the "Decision rationale" card
- Classical models (logistic regression, etc.) export linear contributions directly
- Transformer models surface gradient-derived token contributions (falling back to attention scores when gradients are unavailable, e.g. in quantized mode)

## Model Packaging

The training pipeline automatically packages models for production deployment:
- Models are converted to **ONNX** format for efficient, cross-platform inference
- An **MLflow pyfunc package** is created containing the vectorizer, scaler, ONNX graph, metadata, and decision policy
- All artifacts are versioned and ready for serving

This packaging approach ensures consistent, reproducible deployments across different environments.

## Quantization

Quantization is supported for classical models via `ONNXRuntime` optimizations:
- Reduces model size and inference latency with minimal accuracy loss
- Suitable for deployment scenarios with resource constraints

> [!NOTE]
> All reported benchmarks were produced on a workstation with an RTX 3070 Ti (8 GB) running CUDA-enabled PyTorch. Expect longer transformer fine-tuning times on smaller GPUs or CPU-only boxes.

> [!IMPORTANT]
> See [INSTRUCTIONS.md](INSTRUCTIONS.md) for setup and usage details. 
> 
> Visit [ARCHITECTURE.md](ARCHITECTURE.md) for system design and data flow diagrams. 
> 
> For training results and model diagnostics, refer to [RESULTS.md](RESULTS.md). 
> 
> To learn how to extend the model suite, check out [ADD_MODELS.md](ADD_MODELS.md).
