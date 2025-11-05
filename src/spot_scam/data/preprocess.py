from __future__ import annotations

from typing import Dict, Iterable, Tuple

import pandas as pd

from spot_scam.data.ingest import clean_text, concatenate_text_fields, fill_missing_values
from spot_scam.utils.logging import configure_logging

logger = configure_logging(__name__)


def preprocess_dataframe(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Iterable[str]]:
    """
    Apply standard preprocessing: sanitize text fields, fill missing values, and create `text_all`.

    Returns the processed dataframe and the list of textual fields used to create `text_all`.
    """
    preprocessing_conf = config["preprocessing"]
    text_fields = config["data"]["text_fields"]
    fill_value = preprocessing_conf["fill_missing"]

    df = fill_missing_values(df, fill_value)
    df = df.copy()

    for field in text_fields:
        df[field] = df[field].apply(
            lambda x: clean_text(
                x,
                lowercase=preprocessing_conf["lowercase_text"],
                strip_html_flag=preprocessing_conf["strip_html"],
                strip_urls_flag=preprocessing_conf["strip_urls"],
                normalize_whitespace=preprocessing_conf["normalize_whitespace"],
                max_length=preprocessing_conf.get("max_text_length"),
            )
        )

    df["text_all"] = df.apply(concatenate_text_fields, axis=1, fields=text_fields)

    drop_cols = set(config["data"].get("drop_columns", []))
    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    if existing_drop_cols:
        logger.info("Dropping %d columns marked for removal: %s", len(existing_drop_cols), existing_drop_cols)
        df = df.drop(columns=existing_drop_cols)

    categorical_fields = config["data"].get("categorical_fields", [])
    for cat in categorical_fields:
        if cat in df.columns:
            df[cat] = df[cat].astype("category")

    return df, text_fields


__all__ = ["preprocess_dataframe"]
