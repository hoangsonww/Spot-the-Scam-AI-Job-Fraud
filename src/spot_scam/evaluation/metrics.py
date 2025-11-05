from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
from sklearn import metrics


@dataclass
class MetricResults:
    values: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.5
    confusion_matrix: Optional[np.ndarray] = None


def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    metrics_list: Iterable[str],
    threshold: float,
    positive_label: int = 1,
) -> MetricResults:
    """
    Compute a suite of classification metrics given continuous scores and a decision threshold.
    """
    y_pred = (y_scores >= threshold).astype(int)
    results: Dict[str, float] = {}

    unique_labels = np.unique(y_true)

    for metric_name in metrics_list:
        if metric_name == "f1":
            results["f1"] = metrics.f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        elif metric_name == "precision":
            results["precision"] = metrics.precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        elif metric_name == "recall":
            results["recall"] = metrics.recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        elif metric_name == "roc_auc":
            try:
                results["roc_auc"] = metrics.roc_auc_score(y_true, y_scores)
            except ValueError:
                results["roc_auc"] = float("nan")
        elif metric_name == "average_precision":
            results["pr_auc"] = metrics.average_precision_score(y_true, y_scores) if len(unique_labels) > 1 else 0.0
        elif metric_name == "brier":
            results["brier"] = metrics.brier_score_loss(y_true, y_scores, pos_label=positive_label)
        else:
            raise ValueError(f"Unsupported metric requested: {metric_name}")

    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    return MetricResults(values=results, threshold=threshold, confusion_matrix=cm)


def optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray, metric: str = "f1") -> float:
    """
    Determine the optimal threshold on validation data for a specified metric.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_scores)

    if metric == "f1":
        denominator = precisions + recalls
        denominator[denominator == 0] = 1
        f1_scores = 2 * precisions * recalls / denominator
        best_idx = np.argmax(f1_scores)
        if best_idx >= len(thresholds):
            best_idx = len(thresholds) - 1
        best_threshold = thresholds[best_idx]
    else:
        raise ValueError(f"Unsupported optimization metric: {metric}")
    return float(np.clip(best_threshold, 1e-6, 1 - 1e-6))


def expected_calibration_error(y_true: np.ndarray, y_scores: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute a simple Expected Calibration Error (ECE).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_scores, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if not np.any(mask):
            continue
        avg_conf = y_scores[mask].mean()
        avg_acc = y_true[mask].mean()
        bin_prob = np.mean(mask)
        ece += np.abs(avg_conf - avg_acc) * bin_prob
    return float(ece)


__all__ = ["compute_metrics", "optimal_threshold", "expected_calibration_error", "MetricResults"]
