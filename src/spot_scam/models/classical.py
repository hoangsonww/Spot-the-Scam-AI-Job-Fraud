from __future__ import annotations

import time
from dataclasses import dataclass, field
import warnings
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from spot_scam.evaluation.metrics import MetricResults, compute_metrics, optimal_threshold
from spot_scam.features.builders import FeatureBundle
from spot_scam.utils.logging import configure_logging

logger = configure_logging(__name__)

warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


@dataclass
class ModelRun:
    name: str
    estimator: object
    val_scores: np.ndarray
    val_metrics: MetricResults
    train_time: float
    config: Dict
    threshold: float
    feature_type: str


class ProbabilityEnsemble:
    """
    Lightweight wrapper that averages probabilities from multiple pre-fit estimators.
    """

    def __init__(self, estimators: Sequence[object], weights: Optional[Sequence[float]] = None):
        self.estimators = [est for est in estimators if est is not None]
        if not self.estimators:
            raise ValueError("ProbabilityEnsemble requires at least one estimator.")
        if weights is None:
            weight_array = np.ones(len(self.estimators), dtype=np.float64)
        else:
            weight_array = np.asarray(list(weights), dtype=np.float64)
            if weight_array.shape[0] != len(self.estimators):
                raise ValueError("Number of weights must match the number of estimators.")
        total = float(weight_array.sum())
        if not np.isfinite(total) or total == 0.0:
            raise ValueError("Ensemble weights must sum to a non-zero finite value.")
        self.weights = weight_array / total
        ref = self.estimators[0]
        self.classes_ = getattr(ref, "classes_", np.array([0, 1]))
        self.n_features_in_ = getattr(ref, "n_features_in_", None)

    def predict_proba(self, X):
        prob_list: List[np.ndarray] = []
        for estimator in self.estimators:
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X)
                if proba.ndim == 1:
                    proba = np.vstack([1 - proba, proba]).T
            else:
                scores = estimator.decision_function(X)
                pos = _sigmoid(scores)
                proba = np.vstack([1 - pos, pos]).T
            prob_list.append(proba)
        stacked = np.stack(prob_list, axis=0)
        averaged = np.tensordot(self.weights, stacked, axes=1)
        return averaged

    def decision_function(self, X):
        probs = self.predict_proba(X)[:, 1]
        eps = 1e-6
        probs = np.clip(probs, eps, 1 - eps)
        return np.log(probs / (1 - probs))

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


def train_classical_models(
    bundle: FeatureBundle,
    y_train: np.ndarray,
    y_val: np.ndarray,
    config: Dict,
) -> List[ModelRun]:
    runs: List[ModelRun] = []
    classical_conf = config["models"]["classical"]

    X_train_linear = sparse.hstack([bundle.tfidf_train, bundle.tabular_train]).tocsr()
    X_val_linear = sparse.hstack([bundle.tfidf_val, bundle.tabular_val]).tocsr()

    num_lr = len(classical_conf["logistic_regression"]["Cs"])
    logger.info("Training %d Logistic Regression variants", num_lr)
    for C in classical_conf["logistic_regression"]["Cs"]:
        params = dict(classical_conf["logistic_regression"])
        params["C"] = C
        start = time.time()
        logger.info("Starting Logistic Regression fit (C=%s)", C)
        clf = LogisticRegression(
            C=C,
            penalty=params.get("penalty", "l2"),
            class_weight=params.get("class_weight"),
            max_iter=params.get("max_iter", 400),
            solver="lbfgs",
        )
        clf.fit(X_train_linear, y_train)
        train_time = time.time() - start
        val_scores = clf.predict_proba(X_val_linear)[:, 1]
        threshold = optimal_threshold(
            y_val, val_scores, metric=config["evaluation"]["thresholds"]["optimize_metric"]
        )
        metric_results = compute_metrics(
            y_val,
            val_scores,
            metrics_list=config["evaluation"]["metrics"],
            threshold=threshold,
            positive_label=1,
        )
        runs.append(
            ModelRun(
                name=f"logistic_regression_C{C}",
                estimator=clf,
                val_scores=val_scores,
                val_metrics=metric_results,
                train_time=train_time,
                config=params,
                threshold=threshold,
                feature_type="tfidf+tabular",
            )
        )
        logger.info(
            "Logistic Regression (C=%s) F1=%.3f Precision=%.3f Recall=%.3f",
            C,
            metric_results.values.get("f1", np.nan),
            metric_results.values.get("precision", np.nan),
            metric_results.values.get("recall", np.nan),
        )

    if "logistic_regression_l1" in classical_conf:
        l1_conf = classical_conf["logistic_regression_l1"]
        logger.info(
            "Training %d Logistic Regression L1 variants", len(l1_conf.get("Cs", [0.1, 1.0, 10.0]))
        )
        for C in l1_conf.get("Cs", [0.1, 1.0, 10.0]):
            params = dict(l1_conf)
            params["C"] = C
            start = time.time()
            logger.info("Starting Logistic Regression L1 fit (C=%s)", C)
            clf = LogisticRegression(
                C=C,
                penalty="l1",
                class_weight=params.get("class_weight", "balanced"),
                max_iter=params.get("max_iter", 3000),
                solver="saga",
                n_jobs=-1,
            )
            clf.fit(X_train_linear, y_train)
            train_time = time.time() - start
            val_scores = clf.predict_proba(X_val_linear)[:, 1]
            threshold = optimal_threshold(
                y_val, val_scores, metric=config["evaluation"]["thresholds"]["optimize_metric"]
            )
            metric_results = compute_metrics(
                y_val,
                val_scores,
                metrics_list=config["evaluation"]["metrics"],
                threshold=threshold,
                positive_label=1,
            )
            runs.append(
                ModelRun(
                    name=f"logreg_l1_C{C}",
                    estimator=clf,
                    val_scores=val_scores,
                    val_metrics=metric_results,
                    train_time=train_time,
                    config=params,
                    threshold=threshold,
                    feature_type="tfidf+tabular",
                )
            )
            logger.info(
                "Logistic Regression L1 (C=%s) F1=%.3f Precision=%.3f Recall=%.3f",
                C,
                metric_results.values.get("f1", np.nan),
                metric_results.values.get("precision", np.nan),
                metric_results.values.get("recall", np.nan),
            )

    logger.info("Training %d Linear SVM variants", len(classical_conf["linear_svm"]["Cs"]))
    for C in classical_conf["linear_svm"]["Cs"]:
        params = dict(classical_conf["linear_svm"])
        params["C"] = C
        start = time.time()
        logger.info("Starting Linear SVM fit (C=%s)", C)
        svm = LinearSVC(
            C=C,
            class_weight=params.get("class_weight"),
            max_iter=params.get("max_iter", 1000),
        )
        svm.fit(X_train_linear, y_train)
        train_time = time.time() - start
        decision_scores = svm.decision_function(X_val_linear)
        val_scores = _sigmoid(decision_scores)
        threshold = optimal_threshold(
            y_val, val_scores, metric=config["evaluation"]["thresholds"]["optimize_metric"]
        )
        metric_results = compute_metrics(
            y_val,
            val_scores,
            metrics_list=config["evaluation"]["metrics"],
            threshold=threshold,
            positive_label=1,
        )
        runs.append(
            ModelRun(
                name=f"linear_svm_C{C}",
                estimator=svm,
                val_scores=val_scores,
                val_metrics=metric_results,
                train_time=train_time,
                config=params,
                threshold=threshold,
                feature_type="tfidf+tabular",
            )
        )
        logger.info(
            "Linear SVM (C=%s) F1=%.3f Precision=%.3f Recall=%.3f",
            C,
            metric_results.values.get("f1", np.nan),
            metric_results.values.get("precision", np.nan),
            metric_results.values.get("recall", np.nan),
        )

    tab_train = bundle.tabular_train
    tab_val = bundle.tabular_val
    lightgbm_conf = classical_conf["lightgbm"]
    grid = _expand_grid(lightgbm_conf)
    logger.info("Training %d LightGBM variants", len(grid))
    for params in grid:
        base_params = {"objective": "binary"}
        if "class_weight" not in params and lightgbm_conf.get("class_weight") is not None:
            base_params["class_weight"] = lightgbm_conf.get("class_weight")
        start = time.time()
        logger.info("Starting LightGBM fit with params: %s", params)
        clf = LGBMClassifier(**base_params, **params)
        clf.fit(tab_train.toarray(), y_train)
        train_time = time.time() - start
        val_scores = clf.predict_proba(tab_val.toarray())[:, 1]
        threshold = optimal_threshold(
            y_val, val_scores, metric=config["evaluation"]["thresholds"]["optimize_metric"]
        )
        metric_results = compute_metrics(
            y_val,
            val_scores,
            metrics_list=config["evaluation"]["metrics"],
            threshold=threshold,
            positive_label=1,
        )
        runs.append(
            ModelRun(
                name="lightgbm_" + "_".join(f"{k}{v}" for k, v in params.items()),
                estimator=clf,
                val_scores=val_scores,
                val_metrics=metric_results,
                train_time=train_time,
                config=params,
                threshold=threshold,
                feature_type="tabular",
            )
        )
        logger.info(
            "LightGBM %s F1=%.3f Precision=%.3f Recall=%.3f",
            params,
            metric_results.values.get("f1", np.nan),
            metric_results.values.get("precision", np.nan),
            metric_results.values.get("recall", np.nan),
        )

    return runs


def _expand_grid(param_grid: Dict[str, Iterable]) -> List[Dict]:
    from itertools import product

    keys = [key for key, value in param_grid.items() if isinstance(value, (list, tuple))]
    constant_params = {
        key: value for key, value in param_grid.items() if not isinstance(value, (list, tuple))
    }
    grid = []
    if not keys:
        grid.append(constant_params)
        return grid
    for values in product(*[param_grid[key] for key in keys]):
        params = dict(zip(keys, values))
        params.update(constant_params)
        grid.append(params)
    return grid


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-scores))


__all__ = ["train_classical_models", "ModelRun", "ProbabilityEnsemble"]
