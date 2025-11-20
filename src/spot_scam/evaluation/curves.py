from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.calibration import calibration_curve

from spot_scam.utils.logging import configure_logging

logger = configure_logging(__name__)


def plot_pr_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    label: str,
    path: Path,
) -> None:
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_scores)
    ap = metrics.average_precision_score(y_true, y_scores)
    plt.figure(figsize=(6, 4))
    plt.step(recall, precision, where="post", label=f"{label} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall Curve")
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info("Saved PR curve to %s", path)


def plot_calibration_curve(y_true: np.ndarray, y_scores: np.ndarray, *, path: Path) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=15)
    plt.figure(figsize=(6, 4))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Calibration Curve")
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info("Saved calibration curve to %s", path)


def plot_confusion_matrix(
    cm: np.ndarray, labels: Sequence[str], *, path: Path, normalize: bool = False
) -> None:
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info("Saved confusion matrix plot to %s", path)


def plot_score_distribution(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    path: Path,
    bins: int = 30,
) -> None:
    plt.figure(figsize=(6, 4))
    negatives = y_scores[y_true == 0]
    positives = y_scores[y_true == 1]
    plt.hist(
        negatives,
        bins=bins,
        alpha=0.6,
        label="Legit (0)",
        color="#94a3b8",
        density=True,
    )
    plt.hist(
        positives,
        bins=bins,
        alpha=0.6,
        label="Fraud (1)",
        color="#f97316",
        density=True,
    )
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title("Score Distribution by Class")
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info("Saved score distribution plot to %s", path)


def plot_threshold_sweep(
    thresholds: np.ndarray,
    metrics_map: Dict[str, np.ndarray],
    *,
    path: Path,
) -> None:
    plt.figure(figsize=(6, 4))
    for metric_name, values in metrics_map.items():
        plt.plot(thresholds, values, label=metric_name.upper())
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("Threshold Sweep (Validation)")
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.0)
    plt.grid(alpha=0.2)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info("Saved threshold sweep plot to %s", path)


def plot_probability_vs_feature(
    feature_values: np.ndarray,
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    feature_name: str,
    path: Path,
) -> None:
    feature_values = np.asarray(feature_values, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    plt.figure(figsize=(6, 4))
    plt.scatter(
        feature_values[y_true == 0],
        probabilities[y_true == 0],
        alpha=0.35,
        label="Legit (0)",
        color="#94a3b8",
        s=16,
    )
    plt.scatter(
        feature_values[y_true == 1],
        probabilities[y_true == 1],
        alpha=0.55,
        label="Fraud (1)",
        color="#f97316",
        s=20,
    )
    if feature_values.size > 1 and not np.allclose(feature_values, feature_values[0]):
        slope, intercept = np.polyfit(feature_values, probabilities, 1)
        xs = np.linspace(feature_values.min(), feature_values.max(), 200)
        plt.plot(xs, intercept + slope * xs, color="#0f172a", linewidth=2, label="Linear fit")
    plt.xlabel(feature_name)
    plt.ylabel("Predicted probability")
    plt.title(f"Probability vs {feature_name}")
    plt.grid(alpha=0.2)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info("Saved probability regression plot to %s", path)


def plot_latency_curve(
    batch_sizes: np.ndarray,
    latency_ms: np.ndarray,
    throughput: np.ndarray,
    *,
    path: Path,
) -> None:
    plt.figure(figsize=(6, 4))
    ax1 = plt.gca()
    ax1.plot(batch_sizes, latency_ms, marker="o", color="#2563eb", label="Latency (ms)")
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Latency (ms)", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")

    ax2 = ax1.twinx()
    ax2.plot(batch_sizes, throughput, marker="s", color="#f97316", label="Throughput (req/s)")
    ax2.set_ylabel("Throughput (req/s)", color="#f97316")
    ax2.tick_params(axis="y", labelcolor="#f97316")

    ax1.set_title("Inference Benchmark")
    ax1.grid(alpha=0.2)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info("Saved inference benchmark plot to %s", path)


__all__ = [
    "plot_pr_curve",
    "plot_calibration_curve",
    "plot_confusion_matrix",
    "plot_score_distribution",
    "plot_threshold_sweep",
    "plot_probability_vs_feature",
    "plot_latency_curve",
]
