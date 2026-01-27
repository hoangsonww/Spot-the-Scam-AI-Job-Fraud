# Optuna Hyperparameter Tuning

This document explains how Optuna tuning is integrated into this repository and how to use it without breaking reproducibility or serving parity.

## Table of Contents

- [Scope of the Current Integration](#scope-of-the-current-integration)
- [How to Run Tuning](#how-to-run-tuning)
- [Command Line Interface](#command-line-interface)
- [Hyperparameter Search Spaces](#hyperparameter-search-spaces)
- [Recommended Workflow](#recommended-workflow)
- [Study Storage and Visualization](#study-storage-and-visualization)
- [Grid Search vs Optuna](#grid-search-vs-optuna)
- [Best Practices](#best-practices)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Related Documentation](#related-documentation)

## Scope of the Current Integration

Optuna tuning is currently implemented for two classical models:

- Logistic Regression
- Linear SVM

Implementation locations:

- Tuning logic: `src/spot_scam/tuning/optuna_tuner.py`
- CLI wrapper: `scripts/tune_with_optuna.py`

Transformer tuning is intentionally not automated here due to the much higher per-trial cost.

## How to Run Tuning

### Logistic Regression

```bash
PYTHONPATH=src python scripts/tune_with_optuna.py --model-type logistic --n-trials 20
```

### Linear SVM

```bash
PYTHONPATH=src python scripts/tune_with_optuna.py --model-type svm --n-trials 20
```

## Command Line Interface

The tuning script supports these options:

- `--model-type`: `logistic` or `svm`
- `--n-trials`: number of Optuna trials
- `-c` or `--config-name`: config filename to load (defaults to `defaults.yaml`)

Example:

```bash
PYTHONPATH=src python scripts/tune_with_optuna.py --model-type logistic --n-trials 30 -c defaults.yaml
```

## Hyperparameter Search Spaces

### Logistic Regression

- `C`: log-uniform from 0.01 to 100.0
- `max_iter`: integer from 300 to 1000 (step 100)

### Linear SVM

- `C`: log-uniform from 0.01 to 100.0
- `max_iter`: integer from 1000 to 3000 (step 500)

Both objectives optimize validation F1 using the same threshold optimization logic as the main pipeline.

## Recommended Workflow

A safe and reproducible workflow:

1. Run a baseline training pass.
2. Run Optuna tuning for the model you want to improve.
3. Copy the best parameters into your config.
4. Re-run the standard training pipeline.
5. Confirm the winner and metrics in `artifacts/metadata.json`.

Example flow:

```bash
# Baseline
make train-fast

# Tune
PYTHONPATH=src python scripts/tune_with_optuna.py --model-type logistic --n-trials 30

# Retrain with pinned params
make train-fast
```

## Study Storage and Visualization

Optuna studies are stored in a local SQLite DB by default:

- `optuna_study.db`

Launch the dashboard:

```bash
OMP_NUM_THREADS=1 optuna-dashboard sqlite:///optuna_study.db --server wsgiref --host 127.0.0.1 --port 8080
```

You can run multiple tuning processes against the same study DB if needed.

## Grid Search vs Optuna

Both are useful in this repository, but they serve different goals.

| Aspect | Grid Search | Optuna |
|--------|-------------|--------|
| Search strategy | Exhaustive over fixed points | Bayesian over continuous spaces |
| Runtime | Short and predictable | Longer but more flexible |
| Best for | Fast baselines | Refinement and sensitivity exploration |

Grid search remains the default inside the main training pipeline.

## Best Practices

- Use Optuna after you have a stable baseline.
- Pin best parameters into config to keep runs reproducible.
- Validate improvements on the held-out test split.
- Track changes via `tracking/runs.csv` and `artifacts/metadata.json`.

## Known Limitations

- Transformer tuning is not automated here.
- Optuna can still overfit to the validation split if you chase very small gains.
- The best parameters for validation F1 may not be the best for your operational objective.

## Troubleshooting

### Tuning is too slow

- Reduce `--n-trials`.
- Use `--skip-transformer` in follow-up training runs.

### Results differ from grid search

- This is normal; Optuna explores intermediate values.

### Results are not reproducible

- Confirm the config seed is stable.
- Re-run tuning with the same study DB.

## Related Documentation

- Quick start: [docs/optuna_quickstart.md](optuna_quickstart.md)
- Training strategy: [TRAINING_ANALYSIS.md](../TRAINING_ANALYSIS.md)
- Pipeline narrative: [docs/pipeline_walkthrough.md](pipeline_walkthrough.md)
- Setup and workflows: [INSTRUCTIONS.md](../INSTRUCTIONS.md)
