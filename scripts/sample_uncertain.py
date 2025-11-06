#!/usr/bin/env python3
"""Sample most uncertain predictions for human review."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer

from spot_scam.tracking.predictions import load_predictions_dataframe
from spot_scam.tracking.feedback import load_feedback_dataframe
from spot_scam.utils.paths import TABLES_DIR, ensure_directories

app = typer.Typer(add_completion=False)


def _entropy(probabilities: pd.Series) -> pd.Series:
    p = probabilities.astype(float).clip(1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


@app.command()
def main(
    limit: int = typer.Option(100, "--limit", "-n", help="Number of uncertain cases to sample."),
    policy: str = typer.Option(
        "entropy",
        "--policy",
        "-p",
        help="Sampling policy: entropy (default) or margin.",
    ),
    output: Path = typer.Option(
        TABLES_DIR / "active_sample.csv",
        "--output",
        "-o",
        help="Destination CSV for the sampled queue.",
    ),
) -> None:
    ensure_directories()
    predictions = load_predictions_dataframe()
    if predictions.empty:
        typer.secho("No prediction logs available. Run inference before sampling.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    feedback = load_feedback_dataframe()
    reviewed_ids = set(feedback["request_id"]) if not feedback.empty else set()

    frame = predictions[~predictions["request_id"].isin(reviewed_ids)].copy()
    if frame.empty:
        typer.secho("All logged predictions already have feedback.", fg=typer.colors.GREEN)
        raise typer.Exit(code=0)

    if policy == "entropy":
        frame["_score"] = _entropy(frame["probability"])
        frame = frame.sort_values("_score", ascending=False)
    elif policy == "margin":
        frame["_score"] = np.abs(frame["probability"] - 0.5)
        frame = frame.sort_values("_score", ascending=True)
    else:
        typer.secho(f"Unknown policy '{policy}'. Use 'entropy' or 'margin'.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    sample = frame.head(limit).copy()
    sample.drop(columns=[col for col in sample.columns if col.startswith("_")], inplace=True, errors="ignore")

    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False)
    typer.secho(f"Wrote {len(sample)} uncertain cases to {output}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
