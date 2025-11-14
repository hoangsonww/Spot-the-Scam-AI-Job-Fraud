from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from spot_scam.evaluation.metrics import compute_metrics
from spot_scam.utils.logging import configure_logging

logger = configure_logging(__name__)


@dataclass
class SliceMetric:
    slice_name: str
    category: str
    count: int
    metrics: Dict[str, float]


def compute_slice_metrics(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
    slice_columns: Iterable[str],
    metrics_list: Iterable[str],
    min_count: int = 30,
) -> List[SliceMetric]:
    """
    Compute metrics on specified categorical slices to assess bias and robustness.
    """
    slices: List[SliceMetric] = []
    for column in slice_columns:
        if column not in df.columns:
            logger.warning("Slice column '%s' not found in dataframe; skipping.", column)
            continue

    for category, group in df.groupby(column, dropna=False, observed=False):
        mask = group.index
        subset_labels = labels[df.index.isin(mask)]
        subset_probs = probabilities[df.index.isin(mask)]
        count = subset_labels.size
        if count < min_count:
            continue

        metric_result = compute_metrics(
            subset_labels,
            subset_probs,
            metrics_list=metrics_list,
            threshold=threshold,
            positive_label=1,
        )

        slices.append(
            SliceMetric(
                slice_name=column,
                category=str(category),
                count=count,
                metrics=metric_result.values,
            )
        )
    return slices


def slices_to_dataframe(slices: List[SliceMetric]) -> pd.DataFrame:
    records = []
    for slice_metric in slices:
        record = {
            "slice": slice_metric.slice_name,
            "category": slice_metric.category,
            "count": slice_metric.count,
        }
        record.update(slice_metric.metrics)
        records.append(record)
    return pd.DataFrame(records)


__all__ = ["compute_slice_metrics", "slices_to_dataframe", "SliceMetric"]
