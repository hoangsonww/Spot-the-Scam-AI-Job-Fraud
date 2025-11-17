from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from bs4 import BeautifulSoup

from spot_scam.utils.logging import configure_logging

logger = configure_logging(__name__)


def load_raw_dataset(config: Dict, raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and merge one or more CSV snapshots listed in the configuration."""
    data_conf = config["data"]
    directory = raw_dir or Path(data_conf["raw_dir"])
    filenames = data_conf.get("raw_filenames") or []

    if not filenames:
        primary = data_conf.get("raw_filename")
        alternate = data_conf.get("alternate_raw_filename")
        filenames = [name for name in [primary, alternate] if name]

    loaded_frames = []
    for name in filenames:
        path = directory / name
        if not path.exists():
            logger.warning("Dataset file missing: %s", path)
            continue
        logger.info("Loading raw dataset from %s", path)
        frame = pd.read_csv(path)
        frame = _normalize_columns(frame)
        frame["_source_file"] = name
        loaded_frames.append(frame)

    if not loaded_frames:
        raise FileNotFoundError(
            "No dataset CSVs found. Checked: "
            + ", ".join(str(directory / name) for name in filenames)
        )

    df = pd.concat(loaded_frames, ignore_index=True, sort=False)

    if "fraudulent" not in df.columns:
        raise KeyError("Combined dataset is missing the 'fraudulent' label column.")

    fraudulent_series = df["fraudulent"]
    if fraudulent_series.dtype == object:
        mapped = fraudulent_series.astype(str).str.strip().str.lower().map({"real": 0, "fake": 1})
        df.loc[mapped.notna(), "fraudulent"] = mapped[mapped.notna()]

    df["fraudulent"] = pd.to_numeric(df["fraudulent"], errors="coerce")
    if df["fraudulent"].isna().any():  # pragma: no cover - defensive fallback
        missing = df[df["fraudulent"].isna()].shape[0]
        logger.warning("Detected %d rows with unknown fraudulent label; dropping them.", missing)
        df = df.dropna(subset=["fraudulent"])
    df["fraudulent"] = df["fraudulent"].astype(int)

    key_columns = data_conf.get(
        "dedup_key_columns",
        [
            "title",
            "location",
            "requirements",
            "employment_type",
            "industry",
            "function",
            "fraudulent",
        ],
    )
    for column in key_columns:
        if column not in df.columns:
            df[column] = pd.NA
    key_values = df[key_columns].fillna("").astype(str).agg("||".join, axis=1)
    before = len(df)
    df = df.loc[~key_values.duplicated()].reset_index(drop=True)
    if len(df) != before:
        logger.info("Dropped %d potential duplicate rows after merging sources.", before - len(df))

    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to snake_case and strip whitespace.
    """
    df = df.copy()
    df.columns = [
        re.sub(r"[^0-9a-zA-Z]+", "_", col.strip().lower()).strip("_") for col in df.columns
    ]
    return df


def strip_html(text: str) -> str:
    """Remove HTML content using BeautifulSoup."""
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(" ", strip=True)


def strip_urls(text: str) -> str:
    """Replace URLs with a placeholder token."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"http[s]?://\S+", " ", text)


def clean_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_html_flag: bool = True,
    strip_urls_flag: bool = True,
    normalize_whitespace: bool = True,
    max_length: Optional[int] = None,
) -> str:
    """
    Apply basic cleaning (HTML stripping, URL removal, lowercasing).
    """
    if not isinstance(text, str):
        text = ""
    original = text
    if strip_html_flag:
        text = strip_html(text)
    if strip_urls_flag:
        text = strip_urls(text)
    if lowercase:
        text = text.lower()
    if normalize_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    if max_length:
        text = text[:max_length]
    if not text and original:
        logger.debug("Text emptied after cleaning; original length was %d chars", len(original))
    return text


def concatenate_text_fields(row: pd.Series, fields: Iterable[str]) -> str:
    """Concatenate specified text fields into a single blob."""
    values = []
    for field in fields:
        value = row.get(field, "")
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return "\n\n".join(values)


def compute_row_checksum(row: pd.Series, text_fields: Iterable[str]) -> str:
    """Create a checksum based on text fields to detect duplicates across splits."""
    text = concatenate_text_fields(row, text_fields)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fill_missing_values(df: pd.DataFrame, fill_value: str) -> pd.DataFrame:
    """Fill NA values in object columns with a sentinel."""
    df = df.copy()
    object_cols = df.select_dtypes(include=["object"]).columns
    df[object_cols] = df[object_cols].fillna(fill_value)
    return df


__all__ = [
    "load_raw_dataset",
    "clean_text",
    "concatenate_text_fields",
    "fill_missing_values",
    "compute_row_checksum",
]
