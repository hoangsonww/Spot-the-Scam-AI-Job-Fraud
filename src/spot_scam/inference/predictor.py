from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import sparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from spot_scam.data.preprocess import preprocess_dataframe
from spot_scam.features.tabular import create_tabular_features
from spot_scam.policy.gray_zone import apply_gray_zone
from spot_scam.utils.logging import configure_logging
from spot_scam.utils.paths import ARTIFACTS_DIR, EXPERIMENTS_DIR

logger = configure_logging(__name__)


class FraudPredictor:
    """
    Wrapper around the trained model artifacts to provide batch prediction utilities.
    """

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.metadata = self._load_metadata()
        self.config = self._load_config()
        self.threshold = float(self.metadata["threshold"])
        self.policy = {
            "threshold": self.threshold,
            "width": self.metadata.get("gray_zone", {}).get("width", 0.1),
            "positive_label": self.metadata.get("gray_zone", {}).get("positive_label", "fraud"),
            "negative_label": self.metadata.get("gray_zone", {}).get("negative_label", "legit"),
            "review_label": self.metadata.get("gray_zone", {}).get("review_label", "review"),
        }
        self.model_type = self.metadata["model_type"]
        self.feature_type = self.metadata.get("feature_type", "tfidf+tabular")
        self.use_quantized = os.getenv("SPOT_SCAM_USE_QUANTIZED", "").lower() in {"1", "true", "yes"}

        if self.model_type == "classical":
            self._load_classical_artifacts()
            self.device = None
        elif self.model_type == "transformer":
            self._load_transformer_artifacts()
        else:
            raise ValueError(f"Unsupported model type in metadata: {self.model_type}")

    def get_gray_zone_band(self) -> Dict[str, float]:
        lower = max(0.0, self.threshold - self.policy["width"] / 2)
        upper = min(1.0, self.threshold + self.policy["width"] / 2)
        return {
            "width": float(self.policy["width"]),
            "lower": float(lower),
            "upper": float(upper),
            "positive_label": self.policy["positive_label"],
            "negative_label": self.policy["negative_label"],
            "review_label": self.policy["review_label"],
        }

    def get_model_metadata(self) -> Dict[str, Any]:
        band = self.get_gray_zone_band()
        metadata = {
            "model_name": self.metadata.get("model_name"),
            "model_type": self.metadata.get("model_type"),
            "feature_type": self.metadata.get("feature_type"),
            "calibration_method": self.metadata.get("calibration_method"),
            "threshold": self.threshold,
            "gray_zone": band,
            "val_metrics": self.metadata.get("val_metrics", {}),
            "test_metrics": self.metadata.get("test_metrics", {}),
            "test_ece": self.metadata.get("test_ece"),
        }
        return metadata

    def get_token_importance(self, limit: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        tables_dir = EXPERIMENTS_DIR / "tables"
        positive_path = tables_dir / "top_terms_positive.csv"
        negative_path = tables_dir / "top_terms_negative.csv"

        def _load(path: Path) -> List[Dict[str, Any]]:
            if not path.exists():
                return []
            df = pd.read_csv(path).head(limit)
            return [
                {"term": str(row["term"]), "weight": float(row["weight"])}
                for _, row in df.iterrows()
            ]

        return {"positive": _load(positive_path), "negative": _load(negative_path)}

    def get_token_frequency(self, limit: int = 20) -> List[Dict[str, Any]]:
        path = EXPERIMENTS_DIR / "tables" / "token_frequency_analysis.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path).head(limit)
        return [
            {
                "token": str(row["token"]),
                "positive_count": int(row["positive_count"]),
                "negative_count": int(row["negative_count"]),
                "difference": int(row["difference"]),
            }
            for _, row in df.iterrows()
        ]

    def _load_metadata(self) -> Dict[str, Any]:
        path = self.artifacts_dir / "metadata.json"
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_config(self) -> Dict[str, Any]:
        path = self.artifacts_dir / "config_used.yaml"
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _load_classical_artifacts(self) -> None:
        self.model = joblib.load(self.artifacts_dir / "model.joblib")
        self.vectorizer = joblib.load(self.artifacts_dir / "features" / "tfidf_vectorizer.joblib")
        self.scaler = joblib.load(self.artifacts_dir / "features" / "tabular_scaler.joblib")
        self.feature_names: List[str] = list(joblib.load(self.artifacts_dir / "features" / "tabular_feature_names.joblib"))

    def _load_transformer_artifacts(self) -> None:
        extra = self.metadata.get("extra", {})
        model_dir = Path(extra.get("model_dir", self.artifacts_dir / "transformer" / "best"))
        tokenizer_dir = Path(extra.get("tokenizer_dir", self.artifacts_dir / "transformer" / "tokenizer"))
        if not model_dir.is_absolute():
            model_dir = self.artifacts_dir / model_dir
        if not tokenizer_dir.is_absolute():
            tokenizer_dir = self.artifacts_dir / tokenizer_dir

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        if self.use_quantized:
            quant_meta = self.metadata.get("quantized", {})
            quant_path = None
            if isinstance(quant_meta, dict):
                rel_path = quant_meta.get("path")
                if rel_path:
                    candidate = self.artifacts_dir / rel_path
                    if candidate.exists():
                        quant_path = candidate
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            if quant_path:
                state_dict = torch.load(quant_path, map_location="cpu")
                self.model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded transformer in dynamic INT8 mode.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.transformer_max_length = self.config["models"]["transformer"]["max_length"]

    def predict(self, payload: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        records = list(payload)
        if not records:
            return []

        df_raw = pd.DataFrame(records)
        processed_df, _ = preprocess_dataframe(df_raw, self.config)

        if self.model_type == "classical":
            probabilities = self._predict_classical(processed_df)
        elif self.model_type == "transformer":
            probabilities = self._predict_transformer(processed_df)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        decisions = apply_gray_zone(probabilities, self.policy)
        band = self.get_gray_zone_band()
        labels = (probabilities >= self.threshold).astype(int)

        outputs = []
        for record, prob, label, decision in zip(records, probabilities, labels, decisions):
            outputs.append(
                {
                    "probability_fraud": float(prob),
                    "binary_label": int(label),
                    "decision": decision,
                    "threshold": self.threshold,
                    "gray_zone": band,
                    "meta": {"model_type": self.model_type, "model_name": self.metadata.get("model_name")},
                }
            )
        return outputs

    def _predict_classical(self, processed_df: pd.DataFrame) -> np.ndarray:
        tfidf_features = self.vectorizer.transform(processed_df["text_all"])
        tabular_df = create_tabular_features(processed_df, self.config)
        tabular_df = tabular_df.reindex(columns=self.feature_names, fill_value=0.0)
        tabular_scaled = self.scaler.transform(tabular_df.astype(np.float32))
        tabular_sparse = sparse.csr_matrix(tabular_scaled)

        if self.feature_type == "tfidf+tabular":
            X = sparse.hstack([tfidf_features, tabular_sparse]).tocsr()
        elif self.feature_type == "tabular":
            X = tabular_sparse
        elif self.feature_type == "tfidf":
            X = tfidf_features
        else:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)[:, 1]
        else:
            scores = self.model.decision_function(X)
            probabilities = 1.0 / (1.0 + np.exp(-scores))
        return probabilities

    def _predict_transformer(self, processed_df: pd.DataFrame) -> np.ndarray:
        texts = processed_df["text_all"].tolist()
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.transformer_max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = self.model(**encodings)
            probabilities = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
        return probabilities


__all__ = ["FraudPredictor"]
