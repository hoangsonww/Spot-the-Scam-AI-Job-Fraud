# Problem & Data
- Detect fraudulent job postings using Kaggle "Real or Fake" dataset (17 columns, text-heavy)
- Constraints: single RTX 3070 Ti GPU, mixed precision allowed, raw data kept local
- Outcomes: calibrated probabilities, gray-zone triage, explainability + bias review

---
# Approach & Results
- Preprocess text (HTML/URL stripping, normalization) + engineered tabular signals
- Baselines: TF-IDF + Logistic/SVM, LightGBM tabular; Transformer: DistilBERT (AMP, early stop)
- Validation selection via F1 (fraud), calibration (Platt/Isotonic) + gray-zone optimization
- Artifacts: vectorizer/scaler/models, metrics, PR & calibration curves, slice tables, report.md

---
# Explainability & Operations
- Top TF-IDF tokens for flagged frauds + SHAP summary (tabular track)
- Token frequency deltas + optional LIME HTML snippets for qualitative review
- Gray-zone recommendations exported with threshold band + API service `/predict`
- Next: optional quantization & deployment hardening once devcontainer finalized
