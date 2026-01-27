# Optuna Quick Start Guide

This is the shortest path to running Optuna tuning in this repository. It is intentionally focused on what the included tuning script supports today.

## Table of Contents

- [What Optuna Is Doing Here](#what-optuna-is-doing-here)
- [Prerequisites](#prerequisites)
- [Run Tuning Commands](#run-tuning-commands)
- [What Gets Optimized](#what-gets-optimized)
- [How to Use the Results](#how-to-use-the-results)
- [Optional: Visualize Studies](#optional-visualize-studies)
- [When to Use Optuna vs Grid Search](#when-to-use-optuna-vs-grid-search)
- [Related Documentation](#related-documentation)

## What Optuna Is Doing Here

Optuna is an automatic hyperparameter optimization framework that uses Bayesian optimization (TPE sampling) to explore continuous hyperparameter spaces more efficiently than grid search.

In this repo, Optuna tuning is implemented in:

- `src/spot_scam/tuning/optuna_tuner.py`
- `scripts/tune_with_optuna.py`

## Prerequisites

Optuna is already included in the project dependencies.

Install dependencies:

```bash
pip install -e '.[dev]'
```

## Run Tuning Commands

### Tune Logistic Regression

```bash
PYTHONPATH=src python scripts/tune_with_optuna.py --model-type logistic --n-trials 20
```

### Tune Linear SVM

```bash
PYTHONPATH=src python scripts/tune_with_optuna.py --model-type svm --n-trials 20
```

### Use a custom config name

The tuning script accepts a config filename via `-c` or `--config-name`:

```bash
PYTHONPATH=src python scripts/tune_with_optuna.py --model-type logistic --n-trials 30 -c defaults.yaml
```

## What Gets Optimized

### Logistic Regression

- `C`: regularization strength (log scale from 0.01 to 100.0)
- `max_iter`: maximum iterations (300 to 1000)

### Linear SVM

- `C`: regularization strength (log scale from 0.01 to 100.0)
- `max_iter`: maximum iterations (1000 to 3000)

## How to Use the Results

A simple, reliable workflow:

1. Run Optuna tuning.
2. Copy the best parameters from the logs.
3. Update `configs/defaults.yaml` (or your custom config).
4. Re-run training normally.

Example config update:

```yaml
models:
  classical:
    logistic_regression:
      Cs: [2.34]
      max_iter: 400
```

Then retrain:

```bash
make train-fast
```

## Optional: Visualize Studies

Optuna studies are stored in `optuna_study.db` by default.

Run the dashboard locally:

```bash
OMP_NUM_THREADS=1 optuna-dashboard sqlite:///optuna_study.db --server wsgiref --host 127.0.0.1 --port 8080
```

## When to Use Optuna vs Grid Search

Grid search is already built into the training pipeline and is faster for quick iteration.

Use Optuna when you want to:

- Explore continuous spaces beyond fixed grid values.
- Squeeze out additional performance after a baseline run.
- Investigate sensitivity around a known good region.

## Related Documentation

- Full tuning guide: [docs/optuna_tuning.md](optuna_tuning.md)
- Training strategy: [TRAINING_ANALYSIS.md](../TRAINING_ANALYSIS.md)
- Setup and workflows: [INSTRUCTIONS.md](../INSTRUCTIONS.md)
