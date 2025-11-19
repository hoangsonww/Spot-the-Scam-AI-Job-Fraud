from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

from spot_scam.evaluation.metrics import expected_calibration_error


class IsotonicCalibratedModel:
    def __init__(self, base_estimator, iso_model):
        self.base_estimator = base_estimator
        self.iso_model = iso_model

    def predict_proba(self, X):
        scores = _get_raw_scores(self.base_estimator, X)
        calibrated_scores = self.iso_model.predict(scores)
        return np.vstack([1 - calibrated_scores, calibrated_scores]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def decision_function(self, X):
        return _get_raw_scores(self.base_estimator, X)


@dataclass
class CalibrationResult:
    method: str
    estimator: object
    val_probabilities: np.ndarray
    brier: float
    ece: float


def calibrate_prefit_model(
    estimator,
    X_val,
    y_val,
    methods: Iterable[str],
) -> List[CalibrationResult]:
    results: List[CalibrationResult] = []
    for method in methods:
        if method in {"platt", "sigmoid"}:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="The `cv='prefit'` option is deprecated in 1.6 and will be removed in 1.8.*",
                    category=FutureWarning,
                )
                calibrated = CalibratedClassifierCV(estimator, cv="prefit", method="sigmoid")
                calibrated.fit(X_val, y_val)
                probs = calibrated.predict_proba(X_val)[:, 1]
        elif method == "isotonic":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                    category=UserWarning,
                )
                scores = _get_raw_scores(estimator, X_val)
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(scores, y_val)
                calibrated = IsotonicCalibratedModel(estimator, iso)
                probs = calibrated.predict_proba(X_val)[:, 1]
        else:
            raise ValueError(f"Unsupported calibration method: {method}")

        brier = np.mean((probs - y_val) ** 2)
        ece = expected_calibration_error(y_val, probs)
        results.append(
            CalibrationResult(
                method=method,
                estimator=calibrated,
                val_probabilities=probs,
                brier=brier,
                ece=ece,
            )
        )

    results.sort(key=lambda r: (r.brier, r.ece))
    return results


def _get_raw_scores(estimator, X):
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 1:
            return proba
        return proba[:, 1]
    raise AttributeError("Estimator does not provide decision_function or predict_proba.")


__all__ = ["CalibrationResult", "calibrate_prefit_model", "IsotonicCalibratedModel"]
