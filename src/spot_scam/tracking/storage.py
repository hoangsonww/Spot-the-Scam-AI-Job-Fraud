from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

import pandas as pd

from spot_scam.utils.paths import (
    TRACKING_DIR,
    TRACKING_FEEDBACK_DIR,
    TRACKING_PREDICTIONS_DIR,
)

TRACKING_ROOT = TRACKING_DIR
PREDICTIONS_DIR = TRACKING_PREDICTIONS_DIR
FEEDBACK_DIR = TRACKING_FEEDBACK_DIR

EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(
    r"(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4})"
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _partition_dir(root: Path, *, prefix: str = "date") -> Path:
    date_str = datetime.utcnow().strftime("%Y%m%d")
    partition = root / f"{prefix}={date_str}"
    _ensure_dir(partition)
    return partition


def write_partitioned_parquet(root: Path, records: Iterable[Mapping[str, object]]) -> Path:
    """
    Append records to a partitioned parquet dataset under the provided root.
    """
    records = list(records)
    if not records:
        raise ValueError("No records supplied for parquet write.")

    df = pd.DataFrame(records)
    partition_dir = _partition_dir(root)
    file_path = partition_dir / f"part-{uuid.uuid4().hex}.parquet"
    df.to_parquet(file_path, index=False)
    return file_path


def mask_pii(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    masked = EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)
    masked = PHONE_REGEX.sub("[REDACTED_PHONE]", masked)
    return masked


def sanitize_payload(payload: Mapping[str, object], *, truncate: int = 800) -> Mapping[str, object]:
    """
    Apply PII masking and truncation to a request payload dictionary.
    """
    clean: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            masked = mask_pii(value)
            if masked and len(masked) > truncate:
                masked = masked[: truncate - 3] + "..."
            clean[key] = masked
        else:
            clean[key] = value
    return clean


def serialize_json(data: Mapping[str, object]) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


__all__ = [
    "write_partitioned_parquet",
    "mask_pii",
    "sanitize_payload",
    "serialize_json",
    "TRACKING_ROOT",
    "PREDICTIONS_DIR",
    "FEEDBACK_DIR",
]
