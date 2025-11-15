# Optuna Quick Start Guide

## What is Optuna?

Optuna is an automatic hyperparameter optimization framework that uses Bayesian optimization to intelligently search for the best hyperparameters. Unlike grid search which tries all combinations, Optuna learns from previous trials to focus on promising regions of the search space.

## Installation

Already included in project dependencies:
```bash
pip install -e .
```

## Basic Usage

### 1. Tune Logistic Regression
```bash
PYTHONPATH=src python scripts/tune_with_optuna.py --model-type logistic --n-trials 20
```

### 2. Tune Linear SVM
```bash
PYTHONPATH=src python scripts/tune_with_optuna.py --model-type svm --n-trials 20
```

## Example Output

```
[INFO] Starting Optuna optimization for Logistic Regression (20 trials)
[I 2024-11-15] Trial 0: F1=0.8234 (C=1.23, max_iter=500)
[I 2024-11-15] Trial 1: F1=0.8156 (C=0.45, max_iter=700)
...
[I 2024-11-15] Trial 19: F1=0.8345 (C=2.34, max_iter=400)

============================================================
OPTUNA OPTIMIZATION RESULTS
============================================================
Model: logistic
Best F1: 0.8345
Best params: {'C': 2.34, 'max_iter': 400}
Trials: 20
Time: 45.2s
============================================================
```

## What Gets Optimized?

### Logistic Regression
- `C`: Regularization strength (0.01 to 100.0, log scale)
- `max_iter`: Maximum iterations (300 to 1000)

### Linear SVM
- `C`: Regularization strength (0.01 to 100.0, log scale)
- `max_iter`: Maximum iterations (1000 to 3000)

## Why Use Optuna?

**Grid Search (current default):**
- Tests fixed values: C = [0.1, 1.0, 10.0]
- 3 trials only
- Fast but limited

**Optuna:**
- Tests continuous range: C = 0.01 to 100.0
- Can find optimal values like C=2.34 or C=0.78
- 20+ trials, intelligently sampled
- Takes longer but finds better hyperparameters

## Using Results

After finding best hyperparameters:

1. Copy the best params from Optuna output
2. Update `configs/defaults.yaml`:
   ```yaml
   models:
     classical:
       logistic_regression:
         Cs: [2.34]  # Use Optuna's best value
         max_iter: 400
   ```
3. Run full training: `make train`

## When to Use

- **Use Optuna if:** You want to squeeze out extra 1-2% F1 performance
- **Use Grid Search if:** You need quick results (<1 minute)
- **Use Both:** Grid search for baseline, Optuna for refinement

## See Also

- Full documentation: [docs/optuna_tuning.md](optuna_tuning.md)
- Optuna website: https://optuna.org/
