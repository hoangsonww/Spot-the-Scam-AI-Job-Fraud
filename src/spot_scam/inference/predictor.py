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
        self.use_quantized = os.getenv("SPOT_SCAM_USE_QUANTIZED", "").lower() in {
            "1",
            "true",
            "yes",
        }

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

    def get_threshold_metrics(self, limit: int = 50) -> List[Dict[str, Any]]:
        path = EXPERIMENTS_DIR / "tables" / "threshold_metrics.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path).head(limit)
        return [
            {
                "threshold": float(row["threshold"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
            }
            for _, row in df.iterrows()
        ]

    def get_latency_summary(self) -> List[Dict[str, Any]]:
        path = EXPERIMENTS_DIR / "tables" / "benchmark_latency.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        if df.empty:
            return []

        summary_rows: List[Dict[str, Any]] = []
        for batch_size, group in df.groupby("batch_size"):
            latency_ms = group["latency_ms"].astype(float)
            throughput_rps = group["throughput_rps"].astype(float)
            summary_rows.append(
                {
                    "batch_size": int(batch_size),
                    "latency_p50_ms": float(np.percentile(latency_ms, 50)),
                    "latency_p95_ms": float(np.percentile(latency_ms, 95)),
                    "throughput_rps": float(throughput_rps.mean()),
                }
            )

        summary_rows.sort(key=lambda item: item["batch_size"])
        return summary_rows

    def get_slice_metrics(self, limit: int = 6) -> List[Dict[str, Any]]:
        path = EXPERIMENTS_DIR / "tables" / "slice_metrics.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        if df.empty:
            return []
        if "f1" in df.columns:
            df = df.sort_values(by="f1", ascending=True)
        df = df.head(limit)
        metrics: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            metrics.append(
                {
                    "slice": str(row.get("slice", "")),
                    "category": str(row.get("category", "")),
                    "count": int(row.get("count", 0) or 0),
                    "f1": float(row["f1"]) if pd.notna(row.get("f1")) else None,
                    "precision": (
                        float(row["precision"]) if pd.notna(row.get("precision")) else None
                    ),
                    "recall": float(row["recall"]) if pd.notna(row.get("recall")) else None,
                }
            )
        return metrics

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
        self.feature_names: List[str] = list(
            joblib.load(self.artifacts_dir / "features" / "tabular_feature_names.joblib")
        )
        base_path = self.artifacts_dir / "base_model.joblib"
        self.base_model = joblib.load(base_path) if base_path.exists() else None
        self.tfidf_feature_names = self.vectorizer.get_feature_names_out()

    def _load_transformer_artifacts(self) -> None:
        extra = self.metadata.get("extra", {})
        model_dir = Path(extra.get("model_dir", self.artifacts_dir / "transformer" / "best"))
        tokenizer_dir = Path(
            extra.get("tokenizer_dir", self.artifacts_dir / "transformer" / "tokenizer")
        )
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

    def predict(
        self,
        payload: Iterable[Dict[str, Any]],
        *,
        return_context: bool = False,
    ) -> Any:
        records = list(payload)
        if not records:
            return ([], []) if return_context else []

        df_raw = pd.DataFrame(records)
        processed_df, _ = preprocess_dataframe(df_raw, self.config)

        if self.model_type == "classical":
            classical_output = self._predict_classical(processed_df)
            probabilities = classical_output["probabilities"]
            explanations = classical_output["explanations"]
            tabular_snapshot = classical_output.get("tabular_df")
        elif self.model_type == "transformer":
            transformer_output = self._predict_transformer(processed_df)
            probabilities = transformer_output["probabilities"]
            explanations = transformer_output["explanations"]
            tabular_snapshot = None
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        decisions = apply_gray_zone(probabilities, self.policy)
        band = self.get_gray_zone_band()
        labels = (probabilities >= self.threshold).astype(int)

        outputs = []
        contexts = []
        for idx, (record, prob, label, decision) in enumerate(
            zip(records, probabilities, labels, decisions)
        ):
            tabular_dict = (
                {k: float(v) for k, v in tabular_snapshot.iloc[idx].to_dict().items()}
                if tabular_snapshot is not None
                else {}
            )
            contexts.append(
                {
                    "text_all": processed_df.iloc[idx]["text_all"],
                    "tabular_features": tabular_dict,
                }
            )
            outputs.append(
                {
                    "probability_fraud": float(prob),
                    "binary_label": int(label),
                    "decision": decision,
                    "threshold": self.threshold,
                    "gray_zone": band,
                    "meta": {
                        "model_type": self.model_type,
                        "model_name": self.metadata.get("model_name"),
                    },
                    "explanation": explanations[idx] if explanations else {},
                }
            )
        if return_context:
            return outputs, contexts
        return outputs

    def _predict_classical(self, processed_df: pd.DataFrame) -> Dict[str, Any]:
        tfidf_features = self.vectorizer.transform(processed_df["text_all"])
        tabular_df = create_tabular_features(processed_df, self.config)
        tabular_df = tabular_df.reindex(columns=self.feature_names, fill_value=0.0)
        tabular_scaled = self.scaler.transform(tabular_df.astype(np.float32))
        tabular_sparse = sparse.csr_matrix(tabular_scaled)

        if self.feature_type == "tfidf+tabular":
            feature_matrix = sparse.hstack([tfidf_features, tabular_sparse]).tocsr()
        elif self.feature_type == "tabular":
            feature_matrix = tabular_sparse
        elif self.feature_type == "tfidf":
            feature_matrix = tfidf_features
        else:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(feature_matrix)[:, 1]
        else:
            scores = self.model.decision_function(feature_matrix)
            probabilities = 1.0 / (1.0 + np.exp(-scores))

        if hasattr(self.model, "decision_function"):
            raw_scores = self.model.decision_function(feature_matrix)
        else:
            eps = 1e-6
            probs = np.clip(probabilities, eps, 1 - eps)
            raw_scores = np.log(probs / (1 - probs))

        explanations = self._build_classical_explanations(tfidf_features, tabular_sparse)

        return {
            "probabilities": probabilities,
            "raw_scores": raw_scores,
            "explanations": explanations,
            "tabular_df": tabular_df.reset_index(drop=True),
        }

    def _predict_transformer(self, processed_df: pd.DataFrame) -> Dict[str, Any]:
        texts = processed_df["text_all"].tolist()
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.transformer_max_length,
            return_tensors="pt",
        )
        cached_cpu = {k: v.clone() for k, v in tokenized.items()}
        encodings = {k: v.to(self.device) for k, v in tokenized.items()}

        if not self.use_quantized:
            try:
                probabilities, raw_scores, token_scores = self._run_transformer_with_gradients(
                    encodings
                )
            except RuntimeError as exc:
                logger.warning("Falling back to attention-based transformer explanations: %s", exc)
                probabilities, raw_scores, token_scores = self._run_transformer_with_attentions(
                    encodings
                )
        else:
            probabilities, raw_scores, token_scores = self._run_transformer_with_attentions(
                encodings
            )

        explanations = self._build_transformer_explanations(
            cached_cpu,
            token_scores,
            probabilities,
        )

        return {
            "probabilities": probabilities,
            "raw_scores": raw_scores,
            "explanations": explanations,
            "tabular_df": None,
        }

    def _run_transformer_with_gradients(
        self, encodings: Dict[str, torch.Tensor]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        embedding_layer = self.model.get_input_embeddings()
        input_embeddings = embedding_layer(encodings["input_ids"]).detach()
        input_embeddings.requires_grad_(True)
        input_embeddings.retain_grad()
        forward_kwargs = {
            "inputs_embeds": input_embeddings,
            "attention_mask": encodings["attention_mask"],
            "return_dict": True,
        }
        if "token_type_ids" in encodings:
            forward_kwargs["token_type_ids"] = encodings["token_type_ids"]

        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            outputs = self.model(**forward_kwargs)
            logits = outputs.logits
            class_logit = logits[:, 1]
            grad_outputs = torch.ones_like(class_logit)
            class_logit.backward(gradient=grad_outputs)
            token_scores = torch.sum(input_embeddings.grad * input_embeddings, dim=-1)
            self.model.zero_grad(set_to_none=True)

        probabilities = torch.softmax(logits, dim=1)[:, 1]
        return (
            probabilities.detach().cpu().numpy(),
            class_logit.detach().cpu().numpy(),
            token_scores.detach().cpu().numpy(),
        )

    def _run_transformer_with_attentions(
        self, encodings: Dict[str, torch.Tensor]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.inference_mode():
            outputs = self.model(**encodings, output_attentions=True, return_dict=True)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        attentions = outputs.attentions[-1].mean(dim=1)
        cls_attention = attentions[:, 0, :]
        centered_scores = cls_attention - cls_attention.mean(dim=1, keepdim=True)
        return (
            probabilities.cpu().numpy(),
            logits[:, 1].cpu().numpy(),
            centered_scores.cpu().numpy(),
        )

    def _build_transformer_explanations(
        self,
        cpu_encodings: Dict[str, torch.Tensor],
        token_scores: np.ndarray,
        probabilities: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        input_ids = cpu_encodings["input_ids"].numpy()
        attention_mask = cpu_encodings["attention_mask"].numpy()
        special_tokens = set(self.tokenizer.all_special_tokens)

        def clean_token(token: str) -> str:
            if token.startswith("##"):
                token = token[2:]
            token = token.replace("▁", " ").strip()
            return token

        def merge_tokens(tokens: List[str], scores: np.ndarray) -> List[Dict[str, Any]]:
            merged: List[Dict[str, Any]] = []
            for token, score in zip(tokens, scores):
                if token in special_tokens or not token:
                    continue
                cleaned = clean_token(token)
                if not cleaned:
                    continue
                if token.startswith("##") and merged:
                    merged[-1]["feature"] = f"{merged[-1]['feature']}{cleaned}"
                    merged[-1]["contribution"] += float(score)
                else:
                    merged.append({"feature": cleaned, "contribution": float(score)})
            return merged

        def list_to_text(items: List[str]) -> str:
            if not items:
                return ""
            if len(items) == 1:
                return items[0]
            if len(items) == 2:
                return f"{items[0]} and {items[1]}"
            return f"{', '.join(items[:-1])}, and {items[-1]}"

        explanations: List[Dict[str, Any]] = []
        for idx in range(len(probabilities)):
            mask_len = int(attention_mask[idx].sum())
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[idx][:mask_len])
            scores = token_scores[idx][:mask_len]
            merged = merge_tokens(tokens, scores)
            contributions = [
                {**item, "source": "token"}
                for item in merged
                if abs(item["contribution"]) > 1e-6 and item["feature"]
            ]
            positives = [item for item in contributions if item["contribution"] > 0]
            positives.sort(key=lambda item: item["contribution"], reverse=True)
            negatives = [item for item in contributions if item["contribution"] < 0]
            negatives.sort(key=lambda item: item["contribution"])

            pos_names = [item["feature"] for item in positives[:3]]
            neg_names = [item["feature"] for item in negatives[:3]]

            summary_parts = [f"Transformer probability {probabilities[idx]:.1%}."]
            if pos_names:
                summary_parts.append(
                    f"{list_to_text(pos_names).capitalize()} increased the fraud score"
                )
            if neg_names:
                summary_parts.append(
                    f"{list_to_text(neg_names).capitalize()} pushed the score toward legit"
                )
            if len(summary_parts) == 1:
                summary_parts.append(
                    "No dominant tokens pulled the prediction in either direction."
                )
            summary_text = " ".join(summary_parts)

            if summary_text and not summary_text.endswith("."):
                summary_text += "."

            explanations.append(
                {
                    "top_positive": positives[:top_k],
                    "top_negative": negatives[:top_k],
                    "intercept": None,
                    "summary": summary_text,
                }
            )

        return explanations

    def _build_classical_explanations(
        self,
        tfidf_matrix: sparse.csr_matrix,
        tabular_matrix: sparse.csr_matrix,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if getattr(self, "base_model", None) is None or not hasattr(self.base_model, "coef_"):
            summary = "Explanation available for linear models with accessible coefficients."
            return [
                {"top_positive": [], "top_negative": [], "intercept": None, "summary": summary}
                for _ in range(tfidf_matrix.shape[0])
            ]

        coef = np.asarray(self.base_model.coef_).ravel()
        intercept = float(np.asarray(getattr(self.base_model, "intercept_", [0.0])).ravel()[0])
        tfidf_names = self.tfidf_feature_names
        tfidf_dim = len(tfidf_names)
        tabular_dim = tabular_matrix.shape[1]
        explanations: List[Dict[str, Any]] = []

        def humanize(name: str) -> str:
            return name.replace("scam_term_", "").replace("_", " ").replace("  ", " ").strip()

        def list_to_text(items: List[str]) -> str:
            if not items:
                return ""
            if len(items) == 1:
                return items[0]
            if len(items) == 2:
                return f"{items[0]} and {items[1]}"
            return f"{', '.join(items[:-1])}, and {items[-1]}"

        if coef.shape[0] < tfidf_dim + tabular_dim:
            aligned = coef.shape[0]
            tfidf_dim = min(tfidf_dim, aligned)
            tabular_dim = max(0, aligned - tfidf_dim)

        tfidf_coef = coef[:tfidf_dim]
        tabular_coef = coef[tfidf_dim : tfidf_dim + tabular_dim]

        for i in range(tfidf_matrix.shape[0]):
            contributions: List[Dict[str, Any]] = []

            row = tfidf_matrix.getrow(i).tocoo()
            for col, value in zip(row.col, row.data):
                if col >= tfidf_dim:
                    continue
                contrib = float(value * tfidf_coef[col])
                if contrib == 0.0:
                    continue
                contributions.append(
                    {
                        "feature": tfidf_names[col],
                        "source": "token",
                        "contribution": contrib,
                    }
                )

            if tabular_dim > 0:
                tab_row = tabular_matrix.getrow(i).toarray().ravel()
                for idx, value in enumerate(tab_row):
                    if idx >= tabular_dim or value == 0.0:
                        continue
                    contrib = float(value * tabular_coef[idx])
                    if contrib == 0.0:
                        continue
                    feature_name = (
                        self.feature_names[idx]
                        if idx < len(self.feature_names)
                        else f"tabular_feature_{idx}"
                    )
                    contributions.append(
                        {
                            "feature": feature_name,
                            "source": "tabular",
                            "contribution": contrib,
                        }
                    )

            positives = [item for item in contributions if item["contribution"] > 0]
            positives.sort(key=lambda item: item["contribution"], reverse=True)
            negatives = [item for item in contributions if item["contribution"] < 0]
            negatives.sort(key=lambda item: item["contribution"])

            pos_names = [humanize(item["feature"]) for item in positives[:3]]
            neg_names = [humanize(item["feature"]) for item in negatives[:3]]

            summary_parts: List[str] = []
            if pos_names:
                summary_parts.append(
                    f"{list_to_text(pos_names).capitalize()} pushed the score toward fraud"
                )
            if neg_names:
                summary_parts.append(
                    f"{list_to_text(neg_names).capitalize()} reinforced the legit decision"
                )
            if len(summary_parts) > 1:
                summary_text = "; ".join(summary_parts)
            else:
                summary_text = (
                    summary_parts[0]
                    if summary_parts
                    else "No strong drivers were found for this decision."
                )

            if summary_text and not summary_text.endswith("."):
                summary_text = summary_text + "."

            explanations.append(
                {
                    "top_positive": positives[:top_k],
                    "top_negative": negatives[:top_k],
                    "intercept": intercept,
                    "summary": summary_text,
                }
            )

        return explanations


__all__ = ["FraudPredictor"]
