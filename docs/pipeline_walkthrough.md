# Spot the Scam Pipeline Walkthrough

This document supplements `INSTRUCTIONS.md` and `ARCHITECTURE.md` with an end-to-end narrative of the training pipeline and the artifact contract it produces.

## Table of Contents

- [Training Entrypoint and Orchestration](#training-entrypoint-and-orchestration)
- [Data Sources and Ingestion](#data-sources-and-ingestion)
- [Preprocessing Stages](#preprocessing-stages)
- [Splitting and Persistence](#splitting-and-persistence)
- [Feature Bundle Construction](#feature-bundle-construction)
- [Model Training Tracks](#model-training-tracks)
- [Calibration, Thresholding, and Gray Zone](#calibration-thresholding-and-gray-zone)
- [Ensembles and Winner Selection](#ensembles-and-winner-selection)
- [Artifacts, Reports, and Benchmarks](#artifacts-reports-and-benchmarks)
- [Automation Hooks and Practical Checks](#automation-hooks-and-practical-checks)
- [Related Documentation](#related-documentation)

## Training Entrypoint and Orchestration

- Entrypoint: `src/spot_scam/pipeline/train.py`
- CLI:

```bash
PYTHONPATH=src python -m spot_scam.pipeline.train
```

The orchestrator coordinates the entire pipeline, including data prep, feature engineering, model candidates, calibration, selection, reporting, benchmarking, tracking, and export.

## Data Sources and Ingestion

### Data files

- `data/fake_job_postings.csv`
- `data/Fake_Real_Job_Posting.csv`

### Ingestion behavior

Ingestion happens in `src/spot_scam/data/ingest.py`:

- Normalize column names to lowercase snake-case.
- Merge multiple CSVs.
- Coerce the `fraudulent` label to numeric.
- Drop duplicates using configured key columns.

Optional download helper:

- `scripts/download_data.py`

## Preprocessing Stages

Preprocessing lives in `src/spot_scam/data/preprocess.py` and uses defaults from `configs/defaults.yaml`.

Key behaviors:

- Fill missing values with `<missing>`.
- Clean text via HTML stripping, URL stripping, lowercasing, and whitespace normalization.
- Construct `text_all` by concatenating configured text fields.
- Drop configured columns (including known leakage risks).
- Cast configured categorical fields to category dtype.

## Splitting and Persistence

Splitting happens in `src/spot_scam/data/split.py`:

- Stratified train/validation/test splits.
- A checksum column helps detect near-duplicates before splitting.
- Split indices are persisted to `data/processed/split_indices.npz`.

In addition, the training pipeline persists full split snapshots:

- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`

This enables notebooks and downstream analysis to consume the exact rows used in training.

## Feature Bundle Construction

Feature construction happens in `src/spot_scam/features/builders.py` and returns a `FeatureBundle`.

### Text features (TF-IDF)

- Builder: `src/spot_scam/features/text.py`
- Defaults (from `configs/defaults.yaml`):
  - N-grams: 1-2
  - `min_df`: 3
  - `max_df`: 0.9
  - Sublinear term frequency: enabled
  - Max vocabulary size: driven by `preprocessing.max_vocabulary_size` (default 100,000)

### Tabular features

- Builder: `src/spot_scam/features/tabular.py`
- Examples of signals:
  - Text length, uppercase ratio, digit and punctuation counts
  - Currency and URL counts
  - Scam-term counters from `features.scamming_terms`
  - Binary metadata fields such as `telecommuting`, `has_company_logo`, `has_questions`
  - Missingness flags for key categorical fields

### Persisted feature artifacts

Training persists the feature contract under `artifacts/features/`:

- `tfidf_vectorizer.joblib`
- `tabular_scaler.joblib`
- `tabular_feature_names.joblib`

## Model Training Tracks

The pipeline trains multiple model families under a shared evaluation and selection framework.

### Classical track

Implemented in `src/spot_scam/models/classical.py`:

- Logistic Regression (L2 and optional L1)
- Linear SVM
- LightGBM (tabular-only)

### XGBoost variants

Implemented in `src/spot_scam/models/xgboost_model.py` and orchestrated in `pipeline/train.py`:

- Generates multiple XGBoost variants
- Caps the number of variants via configuration
- Persists per-variant artifacts under `artifacts/xgboost_variants/`

### Transformer track (optional)

Implemented in `src/spot_scam/models/transformer.py`:

- Fine-tunes `distilbert-base-uncased` by default
- Early stopping and FP16 are supported when available

## Calibration, Thresholding, and Gray Zone

These are core parts of the system, not optional extras.

### Calibration

Implemented in `src/spot_scam/evaluation/calibration.py`:

- Platt scaling (`sigmoid`)
- Isotonic regression

The pipeline evaluates methods on validation data and chooses the best calibration outcome.

### Threshold optimization

Implemented in `src/spot_scam/evaluation/metrics.py` via `optimal_threshold(...)`:

- Thresholds are optimized to maximize validation F1.

### Gray-zone policy

Implemented in `src/spot_scam/policy/gray_zone.py`:

- A band around the threshold routes uncertain cases to `review`.

## Ensembles and Winner Selection

After candidate training, the pipeline can build ensemble candidates over top TF-IDF+tabular models.

- Uniform averaging produces `ensemble_top3`.
- Coarse grid-searched weights can produce `ensemble_weighted_top3` when beneficial.

Selection is then straightforward:

- Winner = highest validation F1 across all candidates.

## Artifacts, Reports, and Benchmarks

The pipeline produces artifacts that power serving and documentation.

### Artifacts (serving contract)

- `artifacts/model.joblib`
- `artifacts/base_model.joblib` (when available)
- `artifacts/metadata.json`
- `artifacts/config_used.yaml`
- `artifacts/test_predictions.csv`

### Reports and diagnostics

- Figures: `experiments/figs/*`
- Tables: `experiments/tables/*`
- Markdown report: `experiments/report.md`

### Latency benchmarks

The pipeline benchmarks the actual inference runtime (`FraudPredictor.predict`) and writes:

- `experiments/tables/benchmark_latency.csv`
- `experiments/tables/benchmark_summary.csv`
- `experiments/figs/latency_throughput.png`

## Automation Hooks and Practical Checks

### Useful Make targets

- `make train`
- `make train-fast`
- `make retrain-with-feedback`
- `make review-sample`

### Recommended checks before serving

- Confirm `artifacts/metadata.json` matches expectations.
- Inspect `experiments/figs/calibration_curve_test.png`.
- Start the API and run a quick health check:

```bash
PYTHONPATH=src uvicorn spot_scam.api.app:app --reload
curl http://localhost:8000/health
```

## Related Documentation

- MLOps lifecycle: [MLOPS.md](../MLOPS.md)
- System design: [ARCHITECTURE.md](../ARCHITECTURE.md)
- Training strategy: [TRAINING_ANALYSIS.md](../TRAINING_ANALYSIS.md)
- Setup and operations: [INSTRUCTIONS.md](../INSTRUCTIONS.md)
- Metrics and diagnostics: [RESULTS.md](../RESULTS.md)
- Explainability details: [docs/explainability.md](explainability.md)
- Deployment checklist: [docs/deployment_guide.md](deployment_guide.md)
