# Spot the Scam Slide Deck Brief

---

## 💼 Slide 1 — Title
- Title: **Spot the Scam — Fraud Screening Platform**
- Subtitle: Calibrated job-posting detector with explainable decisions.
- Presenter and date.

## 🧭 Slide 2 — Executive Summary
- Problem: fraudulent job posts waste money + erode trust.
- Solution: merged Kaggle datasets, classical + transformer ensemble, uncertainty-aware policy, interactive UI.
- Impact: highlights (precision/recall, automatic explanations, MLflow packaging).

## 📊 Slide 3 — Data & Cleansing
- Bullet points:
  1. Two Kaggle CSV sources merged, deduplicated via checksum (`~32%` duplicates removed).
  2. Leak-prone columns dropped (`salary_range`, `job_id`, etc.).
  3. Text cleaning (HTML strip, URL removal, normalized whitespace, `<missing>` fill).
- Visual: `[ADD_CHART]` histogram of post lengths (use `experiments/figs/score_distribution_test.png` if helpful).

## 🧪 Slide 4 — Feature Engineering
- TF-IDF (1–2 grams, 50K vocab), tabular features (length, uppercase ratio, digit count, URL count, scam term counts, binary metadata).
- Diagram: `[INSERT_FLOW]` vectorizer + scaler pipeline (refer to `ARCHITECTURE.md` section 3).

## 🧠 Slide 5 — Model Training
- Classical models: Logistic Regression, Linear SVM, LightGBM (grid search).
- Transformer: DistilBERT fine-tuned (AMP, early stopping).
- Validation strategy: stratified train/val/test splits (70/15/15).
- Calibration: Platt + isotonic, pick best by F1.
- Include summary table `[ADD_CHART]` (use `experiments/tables/metrics_summary.csv`).

## 🧮 Slide 6 — Final Metrics
- Present F1 / Precision / Recall / ROC-AUC (val + test) for best classical + best transformer.
- Gray-zone policy width (0.1) to triage borderline predictions.
- Chart: PR curve (`experiments/figs/pr_curve_test.png`) and calibration curve (`experiments/figs/calibration_curve_test.png`).

## 🧾 Slide 7 — Explainability
- Per-prediction explanations:
  - Ranked token + tabular contributions.
  - Natural language summary: e.g., “Immediate start… pushed score toward fraud; Quickbooks… reinforced legit.”
- Display the UI snippet of “Decision rationale” (`frontend screenshot or recreate layout`).
- Mention token frequency / importance (tables in `experiments/tables/top_terms_*.csv`).

## 🚀 Slide 8 — Serving & Packaging
- FastAPI endpoints (`/predict`, `/metadata`, `/insights/...`).
- MLflow integration:
  - Automatic ONNX export + pyfunc package.
  - Serve with `mlflow models serve --env-manager local -m runs:/<RUN_ID>/model -p 8080`.
- Flowchart: `[INSERT_FLOW]` training → artifacts → MLflow → FastAPI → Next.js.

## 📈 Slide 9 — Performance & Benchmarks
- Inference latency benchmarks (batch 1/8/32/128) from `experiments/tables/benchmark_latency.csv`.
- Include `[ADD_CHART]` latency plot (`experiments/figs/latency_throughput.png`).
- Quantization option: dynamic INT8 for transformer (optional slide note).

## 🛠 Slide 10 — Architecture Diagram
- Embed diagram from `ARCHITECTURE.md` section 1 (or redraw in PPT).
- Highlight data pipeline, modeling, MLflow registry, FastAPI, Next.js.

## 🛡 Slide 11 — Risk & Mitigation
- Future work: add context-specific quantization, transformer explanations, GPU deployment.
- Risks: dataset drift, adversarial posts, latency spikes → mitigation (retrain cadence, alerting, caching).

## ✅ Slide 12 — Call to Action
- Next steps for stakeholders: run full training w/ transformer, deploy MLflow model, integrate into moderation workflow.
- Contact info / repository link (`github.com/.../spot-the-scam-project`).

---
