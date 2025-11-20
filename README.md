# Spot the Scam - AI Job Fraud Detection

## Overview

Spot The Scam protects job seekers from fraudulent postings using an ensemble ML pipeline. The system achieves 85.4% precision and 77.2% F1 score on test data, effectively identifying scams while minimizing false alarms. Well-calibrated probabilities (ECE: 0.0066) ensure reliable confidence estimates for decision-making.

## Key Features

- **Ensemble Architecture**: Combines calibrated Linear SVMs and Logistic Regression with TF-IDF and tabular features
- **Classical ML Models**: Logistic Regression, Linear SVM, XGBoost, and LightGBM with isotonic calibration
- **Deep Learning Support**: DistilBERT fine-tuning option for transformer-based classification
- **Explainable Predictions**: Token-level importance analysis with SHAP-style contribution rankings
- **Interactive Dashboard**: Real-time prediction interface with AI-powered chat assistant for fraud analysis
- **Hyperparameter Optimization**: Bayesian search via Optuna with automated tuning workflows
- **Smart Uncertainty Handling**: Gray-zone policy routes low-confidence predictions to human review
- **Production Deployment**: FastAPI REST API with Docker containerization and MLflow experiment tracking

## Performance

| Metric    | Validation | Test  |
|-----------|------------|-------|
| F1        | 0.856      | 0.772 |
| Precision | 0.930      | 0.854 |
| ROC-AUC   | 0.989      | 0.986 |
| Brier     | 0.010      | 0.014 |

## Technology Stack

- **Backend**: Python, FastAPI, scikit-learn, MLflow, Optuna, XGBoost, LightGBM, Transformers
- **Frontend**: Next.js, TypeScript, Tailwind CSS, shadcn/ui
- **Deployment**: Docker, Vercel

## Documentation

- [INFO.md](docs/INFO.md) - Project overview and feature summary
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design and component breakdown
- [INSTRUCTIONS.md](docs/INSTRUCTIONS.md) - Setup and usage instructions
- [TRAINING_ANALYSIS.md](docs/TRAINING_ANALYSIS.md) - Training pipeline and data analysis
- [RESULTS.md](docs/RESULTS.md) - Detailed performance metrics and visualizations
- [ADD_MODELS.md](docs/ADD_MODELS.md) - Instructions for adding new models
- [docs/](docs/) - More documentation
