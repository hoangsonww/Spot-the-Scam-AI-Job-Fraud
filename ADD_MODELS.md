# Adding New Models to Spot the Scam

This guide shows how to add new model candidates while preserving the existing artifact contract, evaluation outputs, and serving parity.

The core idea: new models must flow through the same training orchestrator (`src/spot_scam/pipeline/train.py`) and produce artifacts that `FraudPredictor` can safely load.

## Table of Contents

- [How the Selection System Works](#how-the-selection-system-works)
- [Step 1: Decide Which Track to Extend](#step-1-decide-which-track-to-extend)
- [Step 2: Add Configuration First](#step-2-add-configuration-first)
- [Step 3: Add a Classical Model Candidate](#step-3-add-a-classical-model-candidate)
- [Step 4: Add or Swap a Transformer](#step-4-add-or-swap-a-transformer)
- [Step 5: Ensure It Enters Candidate Selection](#step-5-ensure-it-enters-candidate-selection)
- [Step 6: Validate Artifacts and Serving Parity](#step-6-validate-artifacts-and-serving-parity)
- [Common Pitfalls to Avoid](#common-pitfalls-to-avoid)
- [Helpful References](#helpful-references)

## How the Selection System Works

The training pipeline trains multiple candidates, evaluates them on validation data, and selects the best validation F1 performer. Only after that selection does it compute the final test-set metrics and persist the winner to `artifacts/`.

As long as your new model ends up in the candidate list, the existing selection, reporting, and export steps will continue to work.

## Step 1: Decide Which Track to Extend

There are two main tracks:

- Classical models (scikit-learn / boosting style): `src/spot_scam/models/classical.py` and `src/spot_scam/models/xgboost_model.py`
- Transformer models (Hugging Face): `src/spot_scam/models/transformer.py`

Both tracks feed into the same selection and persistence logic in `src/spot_scam/pipeline/train.py`.

## Step 2: Add Configuration First

Start by declaring your intent in YAML. Copy `configs/defaults.yaml` to a new config file and add your model section there.

Example:

```yaml
models:
  classical:
    logistic_regression:
      Cs: [0.1, 1.0, 10.0, 100.0]
    random_forest:
      n_estimators: [200, 400]
      max_depth: [12, 24]
  transformer:
    model_name: "roberta-base"
    max_length: 256
    batch_size: 8
```

Run with your config:

```bash
PYTHONPATH=src python -m spot_scam.pipeline.train -c configs/roberta.yaml
```

This keeps experiments reproducible and makes it easy to compare runs later.

## Step 3: Add a Classical Model Candidate

Classical candidates are created in `train_classical_models(...)` inside `src/spot_scam/models/classical.py`.

Follow the existing pattern:

1. Reuse the prepared feature matrices.
2. Fit the estimator on train data.
3. Score on validation data.
4. Optimize the threshold via `optimal_threshold(...)`.
5. Compute validation metrics via `compute_metrics(...)`.
6. Append a `ModelRun`.

Shape template:

```python
X_train_linear = sparse.hstack([bundle.tfidf_train, bundle.tabular_train]).tocsr()
X_val_linear = sparse.hstack([bundle.tfidf_val, bundle.tabular_val]).tocsr()

clf.fit(X_train_linear, y_train)
val_scores = clf.predict_proba(X_val_linear)[:, 1]
threshold = optimal_threshold(y_val, val_scores, metric=config["evaluation"]["thresholds"]["optimize_metric"])
metric_results = compute_metrics(...)
runs.append(ModelRun(...))
```

Anything you add to `runs` will later be calibrated (when applicable), evaluated on the test set, logged to `tracking/runs.csv`, and considered for selection.

## Step 4: Add or Swap a Transformer

The easiest transformer change is purely configuration-driven:

- Update `models.transformer.model_name` in your config.
- Adjust `max_length`, `batch_size`, and `fp16` as needed.

Then run:

```bash
PYTHONPATH=src python -m spot_scam.pipeline.train -c configs/roberta.yaml
```

If you want multiple transformer candidates in one run, extend `train_transformer_model(...)` to return multiple results and append them to the candidate list in `pipeline/train.py`.

## Step 5: Ensure It Enters Candidate Selection

The selection rule is simple: best validation F1 wins.

In practice, this means:

- Classical models must appear in the list returned by `train_classical_models(...)`.
- Transformer models must be appended to the candidate list in `pipeline/train.py`.
- Ensembles are already built automatically from top TF-IDF+tabular candidates.

If your model does not appear in `tracking/runs.csv` after training, it likely never entered the candidate set.

## Step 6: Validate Artifacts and Serving Parity

After training, validate that the full contract is still intact.

Checklist:

- `artifacts/metadata.json` names the expected winner and includes metrics.
- `artifacts/config_used.yaml` reflects your custom config.
- `artifacts/features/*.joblib` exists and matches your features.
- `experiments/report.md` and `experiments/figs/*` update without errors.
- The API starts and serves predictions:

```bash
PYTHONPATH=src uvicorn spot_scam.api.app:app --reload
curl http://localhost:8000/health
```

## Common Pitfalls to Avoid

- Changing features without retraining and regenerating artifacts.
- Introducing models that do not expose probabilities (or a usable decision function) without handling that case.
- Editing API schemas without updating the frontend types in `frontend/src/lib/api.ts`.
- Assuming a model won without checking `artifacts/metadata.json`.

## Helpful References

- Setup and workflows: `INSTRUCTIONS.md`
- Architecture and contract boundaries: `ARCHITECTURE.md`
- Training strategy and trade-offs: `TRAINING_ANALYSIS.md`
- Current metrics and diagnostics: `RESULTS.md`
- Pipeline narrative: `docs/pipeline_walkthrough.md`
- Explainability details: `docs/explainability.md`
