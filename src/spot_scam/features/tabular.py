from __future__ import annotations

import re
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from spot_scam.utils.logging import configure_logging

logger = configure_logging(__name__)


SCAM_TERMS_PATTERN_CACHE: Dict[str, re.Pattern] = {}


def _get_pattern(term: str) -> re.Pattern:
    if term not in SCAM_TERMS_PATTERN_CACHE:
        SCAM_TERMS_PATTERN_CACHE[term] = re.compile(re.escape(term), flags=re.IGNORECASE)
    return SCAM_TERMS_PATTERN_CACHE[term]


def create_tabular_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    feature_conf = config["features"]["tabular"]

    if "text_all" not in df.columns:
        raise KeyError("Expected 'text_all' column to construct tabular features.")

    features = {}

    text_series = df["text_all"].fillna("")

    if feature_conf.get("include_text_length"):
        features["text_length"] = text_series.apply(len).astype(np.float32)

    if feature_conf.get("include_uppercase_ratio"):
        features["uppercase_ratio"] = text_series.apply(_uppercase_ratio).astype(np.float32)

    if feature_conf.get("include_digit_count"):
        features["digit_count"] = text_series.str.count(r"\d").astype(np.float32)

    if feature_conf.get("include_currency_token"):
        features["currency_token_count"] = text_series.str.count(r"[$€£]").astype(np.float32)

    if feature_conf.get("include_exclamation_count"):
        features["exclamation_count"] = text_series.str.count(r"!").astype(np.float32)

    if feature_conf.get("include_question_count"):
        features["question_count"] = text_series.str.count(r"\?").astype(np.float32)

    if feature_conf.get("include_url_count"):
        features["url_count"] = text_series.str.count(r"http[s]?://").astype(np.float32)

    if feature_conf.get("include_scamming_terms"):
        scam_terms: Iterable[str] = config["features"].get("scamming_terms", [])
        for term in scam_terms:
            pattern = _get_pattern(term)
            features[f"scam_term_{term.replace(' ', '_')}"] = text_series.apply(
                lambda txt, pat=pattern: len(pat.findall(txt))
            ).astype(np.float32)

    binary_columns = ["telecommuting", "has_company_logo", "has_questions"]
    for column in binary_columns:
        if column in df.columns:
            col = df[column]
            col = col.mask(col == "<missing>")
            numeric = pd.to_numeric(col, errors="coerce").fillna(0.0).astype(np.float32)
            features[column] = numeric

    optional_columns = [
        "employment_type",
        "required_experience",
        "required_education",
        "industry",
        "function",
    ]
    for column in optional_columns:
        if column in df.columns:
            raw = df[column]
            mask = raw.isna()
            raw_str = raw.astype(str).str.lower()
            mask |= raw_str == "<missing>"
            features[f"{column}_is_missing"] = mask.astype(np.float32)

    feature_df = pd.DataFrame(features, index=df.index)
    return feature_df


def _uppercase_ratio(text: str) -> float:
    if not text:
        return 0.0
    upper_chars = sum(1 for char in text if char.isupper())
    return upper_chars / max(len(text), 1)


__all__ = ["create_tabular_features"]
