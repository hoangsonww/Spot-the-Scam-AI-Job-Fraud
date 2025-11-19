from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from spot_scam.utils.logging import configure_logging

logger = configure_logging(__name__)


def generate_lime_explanations(
    texts: List[str],
    predict_proba: Callable[[List[str]], np.ndarray],
    *,
    num_samples: int = 10,
    num_features: int = 8,
) -> Optional[List[str]]:
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        logger.warning("LIME is not installed; skipping local text explanations.")
        return None

    class_names = ["legit", "fraud"]
    explainer = LimeTextExplainer(class_names=class_names)
    explanations = []
    subset = texts[:num_samples]
    for text in subset:
        exp = explainer.explain_instance(text, predict_proba, num_features=num_features)
        explanations.append(exp.as_html())
    return explanations


__all__ = ["generate_lime_explanations"]
