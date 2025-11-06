from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Mapping, Optional

import pandas as pd
import pyarrow.dataset as ds

from spot_scam.tracking.storage import (
    FEEDBACK_DIR,
    mask_pii,
    write_partitioned_parquet,
)
from spot_scam.utils.paths import ensure_directories


def append_feedback(records: Iterable[Mapping[str, object]]) -> List[Mapping[str, object]]:
    """
    Persist reviewer feedback records in an append-only Parquet dataset.
    """
    ensure_directories()
    now = datetime.utcnow().isoformat()
    rows: List[Mapping[str, object]] = []
    for record in records:
        cleaned = dict(record)
        cleaned.setdefault("created_at", now)
        if "rationale" in cleaned:
            cleaned["rationale"] = mask_pii(cleaned.get("rationale"))
        if "notes" in cleaned:
            cleaned["notes"] = mask_pii(cleaned.get("notes"))
        rows.append(cleaned)

    write_partitioned_parquet(FEEDBACK_DIR, rows)
    return rows


def load_feedback_dataframe() -> pd.DataFrame:
    if not FEEDBACK_DIR.exists():
        return pd.DataFrame()
    dataset = ds.dataset(str(FEEDBACK_DIR), format="parquet")
    if dataset.count_rows() == 0:
        return pd.DataFrame()
    table = dataset.to_table()
    return table.to_pandas()


__all__ = ["append_feedback", "load_feedback_dataframe"]
