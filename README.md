# Spot the Scam - AI Job Fraud Detection

Machine learning system to detect fraudulent job postings using ensemble models, TF-IDF features, and calibrated probability scoring.

## Overview

Spot The Scam protects job seekers from fraudulent postings using an ensemble ML pipeline. The system achieves 85.4% precision and 77.2% F1 score on test data, effectively identifying scams while minimizing false alarms. Well-calibrated probabilities (ECE: 0.0066) ensure reliable confidence estimates for decision-making.

## Key Features

- **Ensemble Model**: Multiple calibrated SVMs for robust predictions
- **Interactive Dashboard**: Next.js frontend with real-time scoring and AI chat assistant
- **Explainable AI**: Token-level analysis and SHAP-style contribution rankings
- **Production-Ready**: FastAPI backend with Docker deployment and MLflow tracking
- **Hyperparameter Tuning**: Optuna integration for Bayesian optimization
- **Gray-Zone Policy**: Configurable uncertainty bands for human review
- **Human-in-the-Loop**: Feedback incorporation for continuous improvement

## Performance

| Metric    | Validation | Test  |
|-----------|------------|-------|
| F1        | 0.856      | 0.772 |
| Precision | 0.930      | 0.854 |
| ROC-AUC   | 0.989      | 0.986 |
| Brier     | 0.010      | 0.014 |

## Technology Stack

**Backend**: Python, FastAPI, scikit-learn, MLflow, Optuna, XGBoost, LightGBM, Transformers
**Frontend**: Next.js, TypeScript, Tailwind CSS, shadcn/ui, Plotly
**Deployment**: Docker, Vercel

## Documentation

- [INFO.md](docs/INFO.md) - Project overview and feature summary
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design and component breakdown
- [INSTRUCTIONS.md](docs/INSTRUCTIONS.md) - Setup and usage instructions
- [TRAINING_ANALYSIS.md](docs/TRAINING_ANALYSIS.md) - Training pipeline and data analysis
- [RESULTS.md](docs/RESULTS.md) - Detailed performance metrics and visualizations
- [ADD_MODELS.md](docs/ADD_MODELS.md) - Instructions for adding new models
- [docs/](docs/) - More documentation

## License

MIT License. See [LICENSE](LICENSE) for details.
