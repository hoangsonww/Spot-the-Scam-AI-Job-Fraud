# Adding New Models to Spot the Scam

If you want to experiment with new model architectures in Spot the Scam, this guide walks you through the steps to add classical estimators or transformer-based models. The pipeline is designed to be modular, so adding new candidates is straightforward.

---

## 1. Decide Which Track to Extend

- **Classical (scikit-learn / LightGBM style)**: Lives in `src/spot_scam/models/classical.py`. Models train on TF‑IDF + tabular bundles and are automatically calibrated + evaluated.
- **Transformers (Hugging Face)**: Implemented in `src/spot_scam/models/transformer.py`. Fine-tunes any sequence-classification checkpoint on `text_all`.

Both tracks funnel into `_select_best_artifact` (`src/spot_scam/pipeline/train.py:217-231`), which keeps the winner-selection logic intact-just add more candidates.

---

## 2. Configure Hyperparameters

Create a custom YAML under `configs/` (copy `configs/defaults.yaml` as a starting point) and tweak the `models` section:

```yaml
models:
  classical:
    logistic_regression:
      Cs: [0.1, 1.0, 10.0, 100.0]  # add more C's
    random_forest:
      n_estimators: [200, 400]
      max_depth: [12, 24]
  transformer:
    model_name: "roberta-base"
    max_length: 256
    batch_size: 8
```

Reference the config during training:

```bash
PYTHONPATH=src python -m spot_scam.pipeline.train -c configs/roberta.yaml
```

---

## 3. Add a Classical Estimator

1. Open `src/spot_scam/models/classical.py`.
2. Inside `train_classical_models`, build your estimator using the already-prepared matrices:
   ```python
   X_train_linear = sparse.hstack([bundle.tfidf_train, bundle.tabular_train]).tocsr()
   ```
3. Fit the model, score on `X_val_linear`, and compute a decision threshold via `optimal_threshold`.
4. Wrap the results in a `ModelRun` and append to `runs`.

Example snippet (pseudo-code):
```python
for depth in classical_conf["random_forest"]["max_depth"]:
    clf = RandomForestClassifier(max_depth=depth, n_estimators=...)
    clf.fit(X_train_linear, y_train)
    val_scores = clf.predict_proba(X_val_linear)[:, 1]
    threshold = optimal_threshold(y_val, val_scores, metric=config["evaluation"]["thresholds"]["optimize_metric"])
    metric_results = compute_metrics(...)
    runs.append(ModelRun(...))
```

Everything returned in `runs` is automatically calibrated (`train.py:234-275`) and evaluated on the test set-no extra wiring needed.

---

## 4. Add Another Transformer

Easiest path: change `models.transformer.model_name` in your config to the checkpoint you want (e.g., `bert-base-uncased`, `microsoft/deberta-v3-small`). Adjust `max_length`, `batch_size`, or `fp16` flags as required. Re-run the trainer; the rest of the pipeline (thresholding, reporting, MLflow export) keeps working.

If you need to fine-tune multiple transformers in a single run, extend `train_transformer_model` to loop over a list of model names and return multiple `TransformerRun` objects, or call the trainer multiple times with different configs.

---

## 5. Keeping Winner Selection

`src/spot_scam/pipeline/train.py` collects all classical + transformer candidates, evaluates them on the validation set, and feeds them into `_select_best_artifact`, which picks the highest F1. As long as your new model ends up in `candidate_artifacts`, the “winner takes all” logic and downstream artifacts (`artifacts/metadata.json`, MLflow export, gray-zone policy) remain unchanged.

---

## 6. Run & Inspect

```bash
# Classical + new transformer
PYTHONPATH=src python -m spot_scam.pipeline.train -c configs/roberta.yaml

# Classical only (skip expensive transformer sweeps)
PYTHONPATH=src python -m spot_scam.pipeline.train --skip-transformer -c configs/rf_vs_lgbm.yaml
```

After training, check:

- `artifacts/metadata.json` for the winning model name + metrics.
- `experiments/report.md` and figures/tables for comparisons.
- `tracking/runs.csv` (if enabled) for a chronological record.

Need to compare multiple winners? Repeat with different configs and diff the resulting artifacts.

---
