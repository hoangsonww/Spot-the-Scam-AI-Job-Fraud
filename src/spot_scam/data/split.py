from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from spot_scam.data.ingest import compute_row_checksum
from spot_scam.utils.logging import configure_logging

logger = configure_logging(__name__)


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def create_splits(df: pd.DataFrame, config: Dict, *, persist: bool = True) -> SplitResult:
    """
    Produce stratified train/validation/test splits and persist the indices for reproducibility.
    """
    splits_conf = config["splits"]
    data_conf = config["data"]
    target_col = data_conf["target_column"]

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataframe columns: {df.columns}"
        )

    logger.info(
        "Creating stratified train/val/test splits (%.0f/%.0f/%.0f)",
        splits_conf["train"] * 100,
        splits_conf["val"] * 100,
        splits_conf["test"] * 100,
    )

    # Compute deterministic checksums to avoid duplicates across splits
    checksum_col = "_checksum"
    df[checksum_col] = df.apply(
        compute_row_checksum,
        axis=1,
        text_fields=config["data"]["text_fields"],
    )

    if df[checksum_col].duplicated().any():
        dup_count = df[checksum_col].duplicated().sum()
        logger.warning(
            "Detected %d duplicate records based on text checksum; dropping duplicates.", dup_count
        )
        df = df.drop_duplicates(subset=checksum_col)

    stratify = df[target_col] if splits_conf.get("stratify", True) else None

    train_df, temp_df = train_test_split(
        df,
        test_size=splits_conf["val"] + splits_conf["test"],
        stratify=stratify,
        random_state=splits_conf["seed"],
    )

    val_size_fraction = splits_conf["val"] / (splits_conf["val"] + splits_conf["test"])
    stratify_temp = temp_df[target_col] if stratify is not None else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_size_fraction,
        stratify=stratify_temp,
        random_state=splits_conf["seed"],
    )

    if persist:
        _persist_splits_indices(train_df, val_df, test_df, config)

    for split_name, split_df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        ratio = split_df[target_col].mean()
        logger.info(
            "%s split size: %d | fraud ratio: %.3f", split_name.capitalize(), len(split_df), ratio
        )

    train_df = train_df.drop(columns=[checksum_col])
    val_df = val_df.drop(columns=[checksum_col])
    test_df = test_df.drop(columns=[checksum_col])

    return SplitResult(train=train_df, val=val_df, test=test_df)


def _persist_splits_indices(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, config: Dict
) -> None:
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_path = processed_dir / "split_indices.npz"
    np.savez_compressed(
        splits_path,
        train=train_df.index.values,
        val=val_df.index.values,
        test=test_df.index.values,
    )
    logger.info("Persisted split indices to %s", splits_path)


def load_split_indices(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    splits_path = Path(config["data"]["processed_dir"]) / "split_indices.npz"
    if not splits_path.exists():
        raise FileNotFoundError(f"Split indices file missing: {splits_path}")
    data = np.load(splits_path)
    return data["train"], data["val"], data["test"]


__all__ = ["create_splits", "SplitResult", "load_split_indices"]
