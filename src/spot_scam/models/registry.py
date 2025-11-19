from __future__ import annotations

from .xgboost_model import XGBoostModel

MODEL_REGISTRY = {
    "XGBOOST": XGBoostModel,
}

__all__ = ["MODEL_REGISTRY", "XGBoostModel"]