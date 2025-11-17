# Optuna Hyperparameter Tuning

This document explains how to use Optuna for automated hyperparameter optimization in the Spot the Scam project.

## Overview

Optuna is a hyperparameter optimization framework that uses Bayesian optimization (TPE sampler) to efficiently search the hyperparameter space. Unlike grid search, Optuna intelligently samples promising regions based on previous trial results.

## Installation

Optuna is included in the project dependencies. If you need to install it separately:

```bash
pip install optuna>=3.5.0
```

## Usage

### Quick Start

Run hyperparameter tuning for Logistic Regression:

```bash
PYTHONPATH=src python scripts/tune_with_optuna.py --model-type logistic --n-trials 20
```

Run hyperparameter tuning for Linear SVM:

```bash
PYTHONPATH=src python scripts/tune_with_optuna.py --model-type svm --n-trials 20
```

### Command Line Options

- `--model-type`: Model to optimize (`logistic` or `svm`)
- `--n-trials`: Number of optimization trials (default: 20)
- `--config-path`: Path to configuration file (default: `configs/defaults.yaml`)

### Example Output

```
[INFO] Starting Optuna optimization for Logistic Regression (20 trials)
[I 2024-11-15 00:00:01] Trial 0 finished with value: 0.8234
[I 2024-11-15 00:00:03] Trial 1 finished with value: 0.8156
...
[INFO] Optuna optimization complete. Best F1=0.8345 with params: {'C': 2.34, 'max_iter': 500}

============================================================
OPTUNA OPTIMIZATION RESULTS
============================================================
Model: logistic
Best F1: 0.8345
Best params: {'C': 2.34, 'max_iter': 500}
Trials: 20
Time: 45.2s
============================================================
```

## Hyperparameter Search Spaces

### Logistic Regression

- **C**: Log-uniform distribution from 0.01 to 100.0 (regularization strength)
- **max_iter**: Integer from 300 to 1000 in steps of 100 (maximum iterations)

### Linear SVM

- **C**: Log-uniform distribution from 0.01 to 100.0 (regularization strength)
- **max_iter**: Integer from 1000 to 3000 in steps of 500 (maximum iterations)

## Integration with Training Pipeline

To use Optuna-optimized hyperparameters in your training pipeline:

1. Run Optuna tuning to find best hyperparameters
2. Update `configs/defaults.yaml` with the discovered values
3. Run the full training pipeline

Example:

```bash
# Step 1: Find optimal hyperparameters
PYTHONPATH=src python scripts/tune_with_optuna.py --model-type logistic --n-trials 30

# Step 2: Update configs/defaults.yaml with best params
# (manually edit the file)

# Step 3: Run full training
make train
```

## Advanced Features

### Pruning

Optuna supports early stopping of unpromising trials. You can add pruning to speed up optimization:

```python
from optuna.pruners import MedianPruner

study = optuna.create_study(
    direction="maximize",
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
)
```

### Parallel Optimization

Run multiple trials in parallel using multiple processes:

```bash
# Terminal 1
PYTHONPATH=src python scripts/tune_with_optuna.py --model-type logistic --n-trials 10

# Terminal 2 (shares the same study)
PYTHONPATH=src python scripts/tune_with_optuna.py --model-type logistic --n-trials 10
```

### Visualization

Optuna provides visualization tools. Install the dashboard:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna_study.db
```

Or use built-in plotting:

```python
import optuna.visualization as vis

fig = vis.plot_optimization_history(study)
fig.show()

fig = vis.plot_param_importances(study)
fig.show()
```

## Comparison: Grid Search vs Optuna

| Aspect | Grid Search | Optuna |
|--------|-------------|--------|
| **Search Strategy** | Exhaustive | Bayesian (intelligent) |
| **Efficiency** | Low (tries all combinations) | High (focuses on promising regions) |
| **Trials Needed** | C1 × C2 × ... × Cn | 10-50 typically sufficient |
| **Best For** | Small search spaces | Large/continuous search spaces |
| **Example** | 3 × 3 × 2 = 18 trials | 20 trials (adaptive) |

### Example Comparison

**Grid Search** (current implementation):
- Logistic Regression: 3 C values = 3 trials
- Linear SVM: 3 C values = 3 trials
- Total: 6 trials, ~15 seconds

**Optuna** (with this integration):
- Logistic Regression: 20 trials exploring continuous space
- Can discover intermediate values like C=2.34, max_iter=450
- Total: 20 trials, ~45 seconds, potentially better results

## When to Use Optuna

**Use Optuna when:**
- You want to explore continuous hyperparameter spaces
- You have computational budget for 20+ trials
- You want to discover non-obvious hyperparameter combinations
- You're tuning complex models with many hyperparameters

**Use Grid Search when:**
- You have a small, discrete search space
- You want deterministic, exhaustive results
- You have very limited compute time (<1 minute)
- You're doing quick experimentation

## Best Practices

1. **Start with Grid Search:** Use the existing grid search to get baseline hyperparameters
2. **Refine with Optuna:** Use Optuna to explore around the grid search winners
3. **Validate Results:** Always validate Optuna's suggestions on the test set
4. **Save Studies:** Keep Optuna study objects for later analysis and visualization
5. **Monitor Progress:** Use `show_progress_bar=True` to track optimization

## Limitations

- **Transformer Tuning:** The current implementation has a placeholder for transformer optimization due to computational cost. Each trial would take 3-5 minutes.
- **Overfitting Risk:** More trials can lead to overfitting on the validation set. Use cross-validation or a separate tuning set if concerned.
- **Randomness:** Results may vary slightly between runs due to the stochastic nature of TPE sampling.

## Further Reading

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [TPE Sampler Paper](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization)
- [Hyperparameter Tuning Best Practices](https://arxiv.org/abs/1502.02127)

## Troubleshooting

**Issue:** Optuna trials are very slow
- **Solution:** Reduce `n_trials` or use faster models (grid search may be better)

**Issue:** All trials have similar performance
- **Solution:** Expand the search space or the hyperparameters don't matter much for this dataset

**Issue:** Best params differ significantly from grid search
- **Solution:** This is expected! Optuna can discover better intermediate values

**Issue:** Results not reproducible
- **Solution:** Set `random_seed` in config to ensure TPE sampler is deterministic
