from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from spot_scam.config.loader import config_hash
from spot_scam.utils.paths import TRACKING_DIR


def append_run_record(artifacts: Any, config: Dict) -> None:
    """
    Append a run summary to the tracking CSV file for lightweight experiment management.
    """
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACKING_DIR / "runs.csv"
    is_new = not path.exists()
    fieldnames = [
        "timestamp",
        "model_name",
        "model_type",
        "calibration_method",
        "threshold",
        "val_f1",
        "val_precision",
        "val_recall",
        "test_f1",
        "test_precision",
        "test_recall",
        "config_hash",
    ]

    row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "model_name": artifacts.name,
        "model_type": artifacts.model_type,
        "calibration_method": artifacts.calibration_method or "",
        "threshold": artifacts.threshold,
        "val_f1": artifacts.val_metrics.values.get("f1"),
        "val_precision": artifacts.val_metrics.values.get("precision"),
        "val_recall": artifacts.val_metrics.values.get("recall"),
        "test_f1": artifacts.test_metrics.values.get("f1"),
        "test_precision": artifacts.test_metrics.values.get("precision"),
        "test_recall": artifacts.test_metrics.values.get("recall"),
        "config_hash": config_hash(config),
    }

    with path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


__all__ = ["append_run_record"]
