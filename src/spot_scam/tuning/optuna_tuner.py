"""Optuna-based hyperparameter optimization for fraud detection models."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from spot_scam.evaluation.metrics import compute_metrics, optimal_threshold
from spot_scam.features.builders import FeatureBundle
from spot_scam.utils.logging import configure_logging

logger = configure_logging(__name__)


def optimize_logistic_regression(
    bundle: FeatureBundle,
    y_train: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
    n_trials: int = 20,
    storage_url: Optional[str] = "sqlite:///optuna_study.db",
    study_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Use Optuna to find optimal hyperparameters for Logistic Regression.

    Args:
        bundle: Feature bundle containing train/val features
        y_train: Training labels
        y_val: Validation labels
        config: Configuration dictionary
        n_trials: Number of Optuna trials to run

    Returns:
        Dictionary with best hyperparameters and validation metrics
    """
    X_train = sparse.hstack([bundle.tfidf_train, bundle.tabular_train]).tocsr()
    X_val = sparse.hstack([bundle.tfidf_val, bundle.tabular_val]).tocsr()

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        C = trial.suggest_float("C", 0.01, 100.0, log=True)
        max_iter = trial.suggest_int("max_iter", 300, 1000, step=100)

        # Train model
        model = LogisticRegression(
            C=C,
            penalty="l2",
            class_weight="balanced",
            max_iter=max_iter,
            solver="lbfgs",
            random_state=config["project"]["random_seed"],
        )
        model.fit(X_train, y_train)

        # Evaluate on validation set
        val_probs = model.predict_proba(X_val)[:, 1]
        threshold = optimal_threshold(
            y_val, val_probs, metric=config["evaluation"]["thresholds"]["optimize_metric"]
        )
        metrics = compute_metrics(
            y_val,
            val_probs,
            metrics_list=config["evaluation"]["metrics"],
            threshold=threshold,
            positive_label=1,
        )

        # Return F1 score for optimization
        return metrics.values.get("f1", 0.0)

    # Create study and optimize
    resolved_study_name = study_name or "logistic_regression_tuning"
    logger.info(
        "Starting Optuna optimization for Logistic Regression (%d trials) [study=%s]",
        n_trials,
        resolved_study_name,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config["project"]["random_seed"]),
        study_name=resolved_study_name,
        storage=storage_url,
        load_if_exists=True,
    )

    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    optimization_time = time.time() - start_time

    best_params = study.best_params
    best_value = study.best_value

    logger.info(
        "Optuna optimization complete. Best F1=%.4f with params: %s (%.1fs)",
        best_value,
        best_params,
        optimization_time,
    )

    return {
        "best_params": best_params,
        "best_f1": best_value,
        "n_trials": n_trials,
        "optimization_time": optimization_time,
        "study": study,
    }


def optimize_linear_svm(
    bundle: FeatureBundle,
    y_train: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
    n_trials: int = 20,
    storage_url: Optional[str] = "sqlite:///optuna_study.db",
    study_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Use Optuna to find optimal hyperparameters for Linear SVM.

    Args:
        bundle: Feature bundle containing train/val features
        y_train: Training labels
        y_val: Validation labels
        config: Configuration dictionary
        n_trials: Number of Optuna trials to run

    Returns:
        Dictionary with best hyperparameters and validation metrics
    """
    X_train = sparse.hstack([bundle.tfidf_train, bundle.tabular_train]).tocsr()
    X_val = sparse.hstack([bundle.tfidf_val, bundle.tabular_val]).tocsr()

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        C = trial.suggest_float("C", 0.01, 100.0, log=True)
        max_iter = trial.suggest_int("max_iter", 1000, 3000, step=500)

        # Train model
        model = LinearSVC(
            C=C,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=config["project"]["random_seed"],
        )
        model.fit(X_train, y_train)

        # Get decision scores and convert to probabilities
        decision_scores = model.decision_function(X_val)
        val_probs = 1 / (1 + np.exp(-decision_scores))

        # Evaluate on validation set
        threshold = optimal_threshold(
            y_val, val_probs, metric=config["evaluation"]["thresholds"]["optimize_metric"]
        )
        metrics = compute_metrics(
            y_val,
            val_probs,
            metrics_list=config["evaluation"]["metrics"],
            threshold=threshold,
            positive_label=1,
        )

        # Return F1 score for optimization
        return metrics.values.get("f1", 0.0)

    # Create study and optimize
    resolved_study_name = study_name or "linear_svm_tuning"
    logger.info(
        "Starting Optuna optimization for Linear SVM (%d trials) [study=%s]",
        n_trials,
        resolved_study_name,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config["project"]["random_seed"]),
        study_name=resolved_study_name,
        storage=storage_url,
        load_if_exists=True,
    )

    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    optimization_time = time.time() - start_time

    best_params = study.best_params
    best_value = study.best_value

    logger.info(
        "Optuna optimization complete. Best F1=%.4f with params: %s (%.1fs)",
        best_value,
        best_params,
        optimization_time,
    )

    return {
        "best_params": best_params,
        "best_f1": best_value,
        "n_trials": n_trials,
        "optimization_time": optimization_time,
        "study": study,
    }
