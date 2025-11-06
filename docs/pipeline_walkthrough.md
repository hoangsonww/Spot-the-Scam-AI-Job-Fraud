# Spot the Scam Pipeline Walkthrough

This document supplements `INSTRUCTIONS.md` and `ARCHITECTURE.md` with an end-to-end narrative of the training pipeline. Use it when onboarding engineers or prepping a design review.

## 1. Data Sources & Versioning
- **Primary raw files:** `data/fake_job_postings.csv`, `data/Fake_Real_Job_Posting.csv`
- **Download script:** `scripts/download_data.py`
- **Dedup strategy:** combine datasets → compute SHA256 checksum of normalized `text_all` → drop duplicates (~32 %).
- **Versioning tips:**
  - Store Kaggle download date in git tag or `tracking/runs.csv`
  - Keep raw archives in object storage (S3/GCS) if the repo size must stay small.

## 2. Preprocessing Stages
| Stage | Key operations | Code |
|-------|----------------|------|
| Fill missing | Replace NaNs with `<missing>` | `data/preprocess.py::fill_missing_values` |
| Clean text | Strip HTML/URLs, normalize whitespace, enforce lower-case | `data/preprocess.py::clean_text` |
| Combine text | Build `text_all` from title + body fields | `data/preprocess.py::concatenate_text_fields` |
| Drop leaks | Remove `job_id`, `salary_range`, `department`, etc. | `configs/defaults.yaml[data.drop_columns]` |
| Type coercion | Cast binary columns to `int` after masking `<missing>` | `features/tabular.py` |

## 3. Feature Bundle
- **Text:** TF-IDF `(1,2)` grams, 50k vocab, `sublinear_tf`.
- **Tabular:** text length, uppercase ratio, digit count, currency count, scam term counts, binary metadata, missing flags.
- **Scaler:** `StandardScaler` fitted on train tabular features.
- **Artifacts:**
  - `artifacts/features/tfidf_vectorizer.joblib`
  - `artifacts/features/tabular_scaler.joblib`
  - `artifacts/features/tabular_feature_names.joblib`

## 4. Modeling
### Classical
- Logistic Regression (C ∈ {0.1, 1.0, 10.0})
- Linear SVM (same C grid; pass through sigmoid to estimate probabilities)
- LightGBM small grid (`num_leaves`, `learning_rate`, `n_estimators`, etc.)

### Transformer
- DistilBERT (max length 128, 3 epochs, AdamW, FP16 optional)
- Early stopping (patience 2), gradient accumulation configurable.

### Calibration
- For classical: Platt + Isotonic; pick lowest Brier then ECE.
- Transformer uses native probability head (no extra calibration by default).

## 5. Selection & Artifacts
1. Compare candidates on validation F1.
2. Evaluate winning model on hold-out test set.
3. Persist:
   - `artifacts/model.joblib` (calibrated)
   - `artifacts/base_model.joblib` (pre-calibration)
   - `artifacts/metadata.json`
   - `artifacts/test_predictions.csv`
   - `experiments/figs/*`, `experiments/tables/*`, `experiments/report.md`

## 6. Explainability
- Classical: multiply vectorizer/scaler outputs by logistic coefficients to produce top positive/negative contributions. Summaries are natural-language sentence(s).
- Transformer: stub summary until token-level attributions are implemented.
- Outputs attached to every `/predict` response under `explanation`.

## 7. MLflow + ONNX (Optional)
1. Convert winner to ONNX (captures vectorizer/scaler or transformer graph).
2. Save MLflow pyfunc bundle under `mlruns/<exp>/<run_id>/model`.
3. Serve locally:
   ```bash
   mlflow models serve --env-manager local -m runs:/<RUN_ID>/model -p 8080
   ```
4. Tracking metadata includes threshold, gray-zone widths, calibration method.

## 8. Automation Hooks
- `make train` → full run (classical + transformer).
- `make train-fast` → classical only (useful in CI).
- Add `make export-onnx` (if desired) to force regeneration.
- `tracking/runs.csv` logs metrics + config hash; use it to audit experiments.

## 9. Recommended Checks Before Deployment
- Confirm latest metrics in `artifacts/metadata.json`.
- Inspect `experiments/figs/calibration_curve_test.png`.
- Manually score known-good/known-bad postings via UI or `curl`.
- Run notebook `notebooks/spot_the_scam_overview.ipynb` to recreate the pipeline.

## 10. Next Enhancements
- Add token-level attributions for transformer.
- Implement pipeline caching (TF-IDF reuse) to shorten train time.
- Integrate automated alerts when MLflow metrics regress beyond threshold.

---

**See also:**
- `docs/explainability.md`
- `docs/deployment_guide.md`
- `docs/architecture_diagrams/` (create as needed)
