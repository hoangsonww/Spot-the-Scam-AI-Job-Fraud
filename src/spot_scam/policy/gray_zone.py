from __future__ import annotations

import numpy as np
from typing import Dict, Iterable, List


def classify_probability(
    probability: float,
    *,
    threshold: float,
    width: float,
    positive_label: str,
    negative_label: str,
    review_label: str,
) -> str:
    lower = max(0.0, threshold - width / 2)
    upper = min(1.0, threshold + width / 2)
    if probability >= upper:
        return positive_label
    if probability <= lower:
        return negative_label
    return review_label


def apply_gray_zone(probabilities: Iterable[float], policy: Dict[str, float]) -> List[str]:
    return [
        classify_probability(
            prob,
            threshold=policy["threshold"],
            width=policy["width"],
            positive_label=policy["positive_label"],
            negative_label=policy["negative_label"],
            review_label=policy["review_label"],
        )
        for prob in probabilities
    ]


__all__ = ["classify_probability", "apply_gray_zone"]
