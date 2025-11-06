from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from spot_scam.utils.logging import configure_logging

logger = configure_logging(__name__)


def render_markdown_report(
    metadata: Dict,
    *,
    metrics_summary: pd.DataFrame,
    slice_metrics: Optional[pd.DataFrame],
    token_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Produce a concise markdown report enumerating core metrics, calibration, and slice analysis.
    """
    lines = []
    lines.append(f"# Spot the Scam Report - {metadata.get('model_name', 'Unknown Model')}")
    lines.append("")
    lines.append("## Configuration Snapshot")
    lines.append(f"- Model: **{metadata.get('model_name')}** ({metadata.get('model_type')})")
    lines.append(f"- Calibration: {metadata.get('calibration_method') or 'none'}")
    lines.append(f"- Decision threshold: {metadata.get('threshold'):.4f}")
    lines.append(f"- Gray-zone width: {metadata.get('gray_zone', {}).get('width', 'N/A')}")
    lines.append("")
    lines.append("## Metrics Overview")
    lines.append(metrics_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Token Signals (Top 20)")
    lines.append(token_table.head(20).to_markdown(index=False))
    lines.append("")
    if slice_metrics is not None and not slice_metrics.empty:
        lines.append("## Slice Analysis")
        lines.append(slice_metrics.to_markdown(index=False))
        lines.append("")
    lines.append("## Additional Visuals")
    lines.append(
        "- `experiments/figs/score_distribution_test.png`: probability density by class."
    )
    lines.append(
        "- `experiments/figs/threshold_sweep_val.png`: precision/recall/F1 trade-offs across thresholds."
    )
    lines.append(
        "- `experiments/figs/probability_vs_length.png`: regression view of text length vs fraud probability."
    )
    lines.append(
        "- `experiments/figs/latency_throughput.png`: latency vs throughput benchmark; see `experiments/tables/benchmark_summary.csv`."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append(
        "All metrics computed on the frozen test split. Gray-zone policy maps probabilities within the band to human review."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    logger.info("Wrote evaluation report to %s", output_path)


__all__ = ["render_markdown_report"]
