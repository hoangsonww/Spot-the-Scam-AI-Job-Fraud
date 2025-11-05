#!/usr/bin/env python3
"""Utility script to download the Kaggle Fake Job Postings dataset."""

import subprocess
import sys
from pathlib import Path

import typer

DATA_DIR = Path("data")
DEFAULT_DATASET = "shivamb/real-or-fake-fake-jobposting"

app = typer.Typer(add_completion=False)


def _check_kaggle_cli() -> None:
    try:
        subprocess.run(["kaggle", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as exc:  # pragma: no cover
        typer.secho("Kaggle CLI not found. Install with `pip install kaggle` and configure your API token.", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


@app.command()
def download(dataset: str = typer.Argument(DEFAULT_DATASET, help="Kaggle dataset slug.")) -> None:
    """Download and extract the dataset into the ./data directory."""
    _check_kaggle_cli()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Downloading {dataset} to {DATA_DIR}...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(DATA_DIR), "--force"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        typer.secho(f"Failed to download dataset: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    archives = list(DATA_DIR.glob("*.zip"))
    if not archives:
        typer.secho("No archive downloaded. Verify dataset slug.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    for archive in archives:
        typer.echo(f"Extracting {archive.name}...")
        subprocess.run(["unzip", "-o", str(archive), "-d", str(DATA_DIR)], check=True)
        archive.unlink()

    expected = DATA_DIR / "fake_job_postings.csv"
    if expected.exists():
        typer.secho("Dataset download complete.", fg=typer.colors.GREEN)
    else:
        typer.secho("Warning: expected CSV not found after extraction.", fg=typer.colors.YELLOW)


if __name__ == "__main__":
    app()
