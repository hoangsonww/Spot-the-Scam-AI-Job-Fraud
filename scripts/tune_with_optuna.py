#!/usr/bin/env python
"""Hyperparameter tuning with Optuna for fraud detection models."""

import sys
from pathlib import Path

import typer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spot_scam.config.loader import load_config as _load_config
from spot_scam.data.ingest import load_raw_dataset
from spot_scam.data.preprocess import preprocess_dataframe
from spot_scam.data.split import create_splits
from spot_scam.features.builders import build_feature_bundle
from spot_scam.tuning.optuna_tuner import optimize_linear_svm, optimize_logistic_regression
from spot_scam.utils.logging import configure_logging

app = typer.Typer(add_completion=False)
logger = configure_logging(__name__)
DEFAULT_STORAGE_URL = "sqlite:///optuna_study.db"


@app.command()
def tune(
    model_type: str = typer.Option("logistic", help="Model type: logistic or svm"),
    n_trials: int = typer.Option(20, help="Number of Optuna trials"),
    config_name: str = typer.Option(
        "defaults.yaml", "-c", help="Config filename in configs/ directory"
    ),
):
    """Run Optuna hyperparameter optimization."""
    logger.info("Loading configuration from configs/%s", config_name)
    config = _load_config(Path(config_name))

    logger.info("Loading and preprocessing data...")
    df_raw = load_raw_dataset(config)
    df_clean, _ = preprocess_dataframe(df_raw, config)

    logger.info("Creating stratified splits...")
    splits = create_splits(df_clean, config)

    logger.info("Building feature bundle...")
    bundle = build_feature_bundle(splits.train, splits.val, splits.test, config)
    storage_url = DEFAULT_STORAGE_URL

    y_train = splits.train[config["data"]["target_column"]].values
    y_val = splits.val[config["data"]["target_column"]].values

    if model_type == "logistic":
        logger.info("Optimizing Logistic Regression with Optuna...")
        result = optimize_logistic_regression(
            bundle,
            y_train,
            y_val,
            config,
            n_trials=n_trials,
            storage_url=storage_url,
            study_name="logistic_regression_tuning",
        )
    elif model_type == "svm":
        logger.info("Optimizing Linear SVM with Optuna...")
        result = optimize_linear_svm(
            bundle,
            y_train,
            y_val,
            config,
            n_trials=n_trials,
            storage_url=storage_url,
            study_name="linear_svm_tuning",
        )
    else:
        logger.error("Unknown model type: %s. Use 'logistic' or 'svm'.", model_type)
        raise typer.Exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("OPTUNA OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    logger.info("Model: %s", model_type)
    logger.info("Best F1: %.4f", result["best_f1"])
    logger.info("Best params: %s", result["best_params"])
    logger.info("Trials: %d", result["n_trials"])
    logger.info("Time: %.1fs", result["optimization_time"])
    logger.info("=" * 60)

    # Optionally save study for visualization
    study = result["study"]
    logger.info("\nTo visualize results, use:")
    logger.info(
        "  OMP_NUM_THREADS=1 optuna-dashboard %s --server wsgiref --host 127.0.0.1 --port 8080"
        "  # select '%s' in the UI",
        storage_url,
        study.study_name,
    )


if __name__ == "__main__":
    app()
