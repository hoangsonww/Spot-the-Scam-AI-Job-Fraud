from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from uuid import uuid4

from spot_scam.tracking.storage import (
    FEEDBACK_DIR,
    PREDICTIONS_DIR,
    sanitize_payload,
    serialize_json,
    write_partitioned_parquet,
)
from spot_scam.utils.paths import ensure_directories, TABLES_DIR


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def _hash_features(features: Dict[str, float]) -> str:
    ordered = json.dumps(features, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(ordered.encode("utf-8")).hexdigest()


def log_predictions(
    *,
    payloads: Sequence[Dict[str, object]],
    processed_text: Sequence[str],
    tabular_features: Sequence[Optional[Dict[str, float]]],
    predictions: Sequence[Dict[str, object]],
    model_name: str,
) -> List[Dict[str, object]]:
    """
    Persist per-request prediction metadata for active learning and review workflows.
    Returns the immutable records that were written (useful for downstream feedback).
    """
    ensure_directories()
    timestamp = datetime.utcnow().isoformat()
    records: List[Dict[str, object]] = []

    for idx, raw_prediction in enumerate(predictions):
        payload = payloads[idx]
        text_blob = processed_text[idx]
        features = tabular_features[idx] or {}

        text_hash = _hash_text(text_blob or "")
        feature_values = {k: float(v) for k, v in features.items()}
        features_hash = _hash_features(feature_values) if feature_values else _hash_text("")

        request_id = str(uuid4())
        sanitized_payload = sanitize_payload(payload)

        explanation = raw_prediction.get("explanation", {})
        record = {
            "request_id": request_id,
            "created_at": timestamp,
            "model_version": model_name,
            "probability": float(raw_prediction["probability_fraud"]),
            "predicted_label": raw_prediction.get("decision", "unknown"),
            "threshold": float(raw_prediction.get("threshold", np.nan)),
            "text_hash": text_hash,
            "features_hash": features_hash,
            "payload": serialize_json(sanitized_payload),
            "explanation": serialize_json(explanation) if explanation else None,
            "meta": serialize_json(raw_prediction.get("meta", {})),
        }
        records.append(record)

    write_partitioned_parquet(PREDICTIONS_DIR, records)
    return records


def _load_dataset(root: Path) -> Optional[pd.DataFrame]:
    if not root.exists():
        return None
    dataset = ds.dataset(str(root), format="parquet")
    if dataset.count_rows() == 0:
        return None
    table = dataset.to_table()
    return table.to_pandas()


def _load_active_sample() -> Optional[pd.DataFrame]:
    sample_path = TABLES_DIR / "active_sample.csv"
    if not sample_path.exists():
        return None
    try:
        df = pd.read_csv(sample_path)
    except Exception:
        return None
    if df.empty:
        return None
    if "request_id" not in df.columns or "probability" not in df.columns:
        return None
    return df


def load_predictions_dataframe() -> pd.DataFrame:
    df = _load_dataset(PREDICTIONS_DIR)
    if df is None:
        return pd.DataFrame()
    return df


def load_feedback_dataframe() -> pd.DataFrame:
    df = _load_dataset(FEEDBACK_DIR)
    if df is None:
        return pd.DataFrame()
    return df


def get_review_queue(
    *,
    policy: str,
    limit: int,
    threshold: float,
    gray_zone_width: float,
) -> Dict[str, object]:
    """
    Return a filtered queue of predictions awaiting human review.
    """
    predictions_df = load_predictions_dataframe()
    if predictions_df.empty:
        return {"total_pending": 0, "items": []}

    feedback_df = load_feedback_dataframe()
    reviewed_ids = set(feedback_df["request_id"]) if not feedback_df.empty else set()

    preds = predictions_df.copy()
    preds = preds[~preds["request_id"].isin(reviewed_ids)]

    if policy == "gray-zone":
        lower = max(0.0, threshold - gray_zone_width / 2)
        upper = min(1.0, threshold + gray_zone_width / 2)
        preds = preds[(preds["probability"] >= lower) & (preds["probability"] <= upper)]
    elif policy == "entropy":
        probs = preds["probability"].astype(float)
        entropy = -(probs * np.log2(probs + 1e-9) + (1 - probs) * np.log2(1 - probs + 1e-9))
        preds = preds.assign(_entropy=entropy).sort_values("_entropy", ascending=False)
    else:
        preds = preds.sort_values("created_at", ascending=False)

    active_sample = _load_active_sample()
    if active_sample is not None:
        sample = active_sample.copy()
        if "created_at" not in sample.columns:
            sample["created_at"] = datetime.utcnow().isoformat()
        if "payload" not in sample.columns:
            sample["payload"] = "{}"
        if "explanation" not in sample.columns:
            sample["explanation"] = "{}"
        sample = sample[~sample["request_id"].isin(reviewed_ids)]
        preds = pd.concat([preds, sample], ignore_index=True)

    if policy == "gray-zone" and preds.empty and not predictions_df.empty:
        probs = predictions_df["probability"].astype(float)
        entropy = -(probs * np.log2(probs + 1e-9) + (1 - probs) * np.log2(1 - probs + 1e-9))
        preds = predictions_df.assign(_entropy=entropy)[~predictions_df["request_id"].isin(reviewed_ids)]
        preds = preds.sort_values("_entropy", ascending=False)

    preds = preds.drop_duplicates(subset=["request_id"], keep="first")
    if "created_at" in preds.columns:
        preds["created_at"] = pd.to_datetime(preds["created_at"], errors="coerce")
        preds["created_at"] = preds["created_at"].fillna(pd.Timestamp.utcnow())
    else:
        preds["created_at"] = pd.Timestamp.utcnow()
    preds["payload"] = preds.get("payload", "{}").fillna("{}")
    preds["explanation"] = preds.get("explanation", "{}").fillna("{}")

    total_pending = len(preds)
    preds = preds.sort_values("created_at", ascending=False).head(limit)

    items: List[Dict[str, object]] = []
    for _, row in preds.iterrows():
        payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else {}
        explanation = json.loads(row["explanation"]) if isinstance(row["explanation"], str) else {}
        items.append(
            {
                "request_id": row["request_id"],
                "created_at": row["created_at"],
                "probability": float(row["probability"]),
                "predicted_label": row["predicted_label"],
                "model_version": row["model_version"],
                "text_hash": row["text_hash"],
                "features_hash": row["features_hash"],
                "threshold": float(row.get("threshold", np.nan)),
                "payload": payload,
                "explanation": explanation,
            }
        )

    return {"total_pending": int(total_pending), "items": items}


__all__ = [
    "log_predictions",
    "load_predictions_dataframe",
    "load_feedback_dataframe",
    "get_review_queue",
]
