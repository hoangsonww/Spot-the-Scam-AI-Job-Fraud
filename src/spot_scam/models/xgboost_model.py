from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Dict

import numpy as np
from scipy import sparse
from xgboost import XGBClassifier
from spot_scam.utils.logging import configure_logging
from sklearn.calibration import CalibratedClassifierCV

from spot_scam.features.builders import FeatureBundle
from spot_scam.evaluation.metrics import compute_metrics, optimal_threshold, MetricResults


@dataclass
class XGBoostTrainingResult:
    name: str
    calibrated_estimator: CalibratedClassifierCV
    base_estimator: XGBClassifier
    val_probabilities: np.ndarray
    val_metrics: MetricResults
    threshold: float
    config: Dict


class XGBoostModel:
    """Wrapper for training a calibrated XGBoost classifier on TF-IDF + tabular features.

    This class mirrors the interface expectations of the existing classical models
    so it can be integrated into the training pipeline seamlessly.
    """

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.logger = configure_logging(__name__ + ".xgb")
        xgb_cfg = self.config.get("models", {}).get("xgboost", {})
        self.calibration_method = str(xgb_cfg.get("calibration_method", "sigmoid"))
        self.early_stopping_rounds = int(xgb_cfg.get("early_stopping_rounds", 0))
        self.base_estimator = XGBClassifier(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.2,
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=3,
            use_label_encoder=False,
        )
        self.calibrated_estimator: CalibratedClassifierCV | None = None
        self.threshold: float | None = None

    def fit(self, bundle: FeatureBundle, y_train: np.ndarray, y_val: np.ndarray) -> XGBoostTrainingResult:
        # Build combined sparse matrices (TF-IDF + tabular) identical to other classical models.
        X_train = sparse.hstack([bundle.tfidf_train, bundle.tabular_train]).tocsr()
        X_val = sparse.hstack([bundle.tfidf_val, bundle.tabular_val]).tocsr()

        self.logger.info("Starting XGBoost fit: n=%d depth=%d lr=%.3f", self.base_estimator.n_estimators, self.base_estimator.max_depth, self.base_estimator.learning_rate)
        fit_kwargs: Dict = {"verbose": False}
        if self.early_stopping_rounds and self.early_stopping_rounds > 0:
            eval_set = [(X_val, y_val)]
            fit_signature = inspect.signature(self.base_estimator.fit)
            fit_params = fit_signature.parameters
            if "callbacks" in fit_params:
                try:
                    from xgboost.callback import EarlyStopping as _EarlyStopping
                    fit_kwargs.update({
                        "eval_set": eval_set,
                        "callbacks": [_EarlyStopping(rounds=self.early_stopping_rounds, save_best=True)],
                    })
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.warning("XGBoost callbacks unavailable (%s); falling back to early_stopping_rounds.", exc)
                    if "early_stopping_rounds" in fit_params:
                        fit_kwargs.update({"eval_set": eval_set, "early_stopping_rounds": self.early_stopping_rounds})
            elif "early_stopping_rounds" in fit_params:
                fit_kwargs.update({"eval_set": eval_set, "early_stopping_rounds": self.early_stopping_rounds})
            else:  # pragma: no cover - very old versions
                self.logger.warning("Early stopping requested but this xgboost version does not expose callbacks or early_stopping_rounds; continuing without it.")
        self.base_estimator.fit(X_train, y_train, **fit_kwargs)
        self.logger.info("Completed XGBoost fit")

        # Calibrate using validation set (prefit mode). Method can be 'sigmoid' (Platt) or 'isotonic'.
        method = self.calibration_method if self.calibration_method in {"sigmoid", "isotonic"} else "sigmoid"
        calibrator = CalibratedClassifierCV(self.base_estimator, cv="prefit", method=method)
        calibrator.fit(X_val, y_val)
        self.calibrated_estimator = calibrator

        val_probs = calibrator.predict_proba(X_val)[:, 1]
        self.threshold = optimal_threshold(
            y_val,
            val_probs,
            metric=self.config["evaluation"]["thresholds"]["optimize_metric"],
        )

        val_metrics = compute_metrics(
            y_val,
            val_probs,
            metrics_list=self.config["evaluation"]["metrics"],
            threshold=self.threshold,
            positive_label=1,
        )

        return XGBoostTrainingResult(
            name="XGBOOST",
            calibrated_estimator=calibrator,
            base_estimator=self.base_estimator,
            val_probabilities=val_probs,
            val_metrics=val_metrics,
            threshold=self.threshold,
            config={},
        )

    def predict_proba(self, X):  # pragma: no cover - thin wrapper
        if self.calibrated_estimator is None:
            raise RuntimeError("Model not fitted; call fit() first.")
        return self.calibrated_estimator.predict_proba(X)


__all__ = ["XGBoostModel", "XGBoostTrainingResult"]
