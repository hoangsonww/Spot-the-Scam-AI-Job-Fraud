from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import shap

from spot_scam.utils.logging import configure_logging

logger = configure_logging(__name__)


def compute_tabular_shap(
    estimator,
    data: np.ndarray,
    feature_names: Iterable[str],
    *,
    sample_size: int = 1024,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    feature_names = list(feature_names)
    background = data
    if data.shape[0] > sample_size:
        idx = np.random.choice(data.shape[0], size=sample_size, replace=False)
        background = data[idx]

    if hasattr(estimator, "booster_") or estimator.__class__.__name__.lower().startswith("lgbm"):
        explainer = shap.TreeExplainer(estimator)
    elif hasattr(estimator, "coef_"):
        explainer = shap.LinearExplainer(estimator, background)
    else:
        explainer = shap.KernelExplainer(estimator.predict_proba, background)  # type: ignore[attr-defined]

    shap_values = explainer(background)
    values = (
        shap_values.values
        if isinstance(shap_values, shap._explanation.Explanation)
        else shap_values
    )
    if isinstance(values, list):
        values = values[1]
    mean_abs_shap = np.mean(np.abs(values), axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    df = df.sort_values(by="mean_abs_shap", ascending=False)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved SHAP summary to %s", output_path)

    return df


__all__ = ["compute_tabular_shap"]
