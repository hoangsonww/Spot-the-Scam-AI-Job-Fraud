from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import typer
from scipy import sparse, stats

from spot_scam.config.loader import config_hash, dump_config, load_config
from spot_scam.data.ingest import load_raw_dataset
from spot_scam.data.preprocess import preprocess_dataframe
from spot_scam.data.split import SplitResult, create_splits
from spot_scam.api.schemas import JobPostingInput
from spot_scam.evaluation.calibration import calibrate_prefit_model
from spot_scam.evaluation.bias import compute_slice_metrics, slices_to_dataframe
from spot_scam.evaluation.curves import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_score_distribution,
    plot_threshold_sweep,
    plot_probability_vs_feature,
    plot_latency_curve,
)
from spot_scam.evaluation.metrics import MetricResults, compute_metrics, expected_calibration_error
from spot_scam.features.builders import FeatureBundle, build_feature_bundle
from spot_scam.models.classical import ModelRun, train_classical_models
from spot_scam.models.transformer import TransformerRun, train_transformer_model
from spot_scam.tracking.logger import append_run_record
from spot_scam.policy.gray_zone import classify_probability
from spot_scam.utils.logging import configure_logging
from spot_scam.utils.paths import ARTIFACTS_DIR, FIGS_DIR, TABLES_DIR, EXPERIMENTS_DIR, ensure_directories
from spot_scam.evaluation.reporting import render_markdown_report
from spot_scam.explainability.textual import top_tfidf_terms, token_frequency_analysis
from spot_scam.explainability.shapley import compute_tabular_shap
from spot_scam.export import log_model_to_mlflow, MLFlowExportError

app = typer.Typer(add_completion=False)
logger = configure_logging(__name__)


@dataclass
class BestModelArtifacts:
    name: str
    model_type: str
    estimator: object
    base_estimator: object
    threshold: float
    calibration_method: Optional[str]
    val_metrics: MetricResults
    val_probabilities: np.ndarray
    test_metrics: MetricResults
    test_probabilities: np.ndarray
    test_labels: np.ndarray
    feature_type: str
    extra: Dict = field(default_factory=dict)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


@app.command()
def run(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    skip_transformer: bool = typer.Option(False, help="Skip transformer fine-tuning to save time."),
) -> None:
    """Execute the end-to-end training pipeline."""
    config = load_config(config_path=config_path)
    set_global_seed(config["project"]["random_seed"])
    ensure_directories()

    logger.info("Configuration hash: %s", config_hash(config))

    raw_df = load_raw_dataset(config)
    processed_df, _ = preprocess_dataframe(raw_df, config)

    splits = create_splits(processed_df, config, persist=True)
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits.train.to_parquet(processed_dir / "train.parquet", index=False)
    splits.val.to_parquet(processed_dir / "val.parquet", index=False)
    splits.test.to_parquet(processed_dir / "test.parquet", index=False)
    bundle = build_feature_bundle(splits.train, splits.val, splits.test, config)

    y_train = splits.train[config["data"]["target_column"]].values
    y_val = splits.val[config["data"]["target_column"]].values
    y_test = splits.test[config["data"]["target_column"]].values

    classical_runs = train_classical_models(bundle, y_train, y_val, config)
    candidates = []
    best_run = max(classical_runs, key=lambda run: run.val_metrics.values.get("f1", -np.inf))
    candidates.append(("classical", best_run))

    transformer_run: Optional[TransformerRun] = None
    if not skip_transformer:
        transformer_run = train_transformer_model(
            splits.train,
            splits.val,
            splits.test,
            config,
            output_dir=ARTIFACTS_DIR,
        )
        candidates.append(("transformer", transformer_run))

    best_model_artifacts = _evaluate_and_select_best(
        candidates,
        bundle=bundle,
        splits=splits,
        config=config,
        y_val=y_val,
        y_test=y_test,
    )

    _persist_artifacts(best_model_artifacts, bundle, config, config_path)
    _generate_report_assets(
        best_model_artifacts,
        config=config,
        y_val=y_val,
        splits=splits,
        bundle=bundle,
    )
    append_run_record(best_model_artifacts, config)
    try:
        log_model_to_mlflow(best_model_artifacts, bundle, config, splits)
    except MLFlowExportError as exc:
        logger.warning("Skipping MLflow export: %s", exc)
    typer.echo("Training complete. Best model: " + best_model_artifacts.name)


def _evaluate_and_select_best(
    candidates,
    *,
    bundle: FeatureBundle,
    splits: SplitResult,
    config: Dict,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> BestModelArtifacts:
    best_candidate = None
    best_f1 = -np.inf

    for model_type, run in candidates:
        if isinstance(run, ModelRun):
            f1 = run.val_metrics.values.get("f1", -np.inf)
        elif isinstance(run, TransformerRun):
            f1 = run.val_metrics.values.get("f1", -np.inf)
        else:
            continue
        if f1 > best_f1:
            best_f1 = f1
            best_candidate = (model_type, run)

    if best_candidate is None:
        raise RuntimeError("No candidate models were trained.")

    model_type, run = best_candidate
    logger.info("Selected best model (%s): %s (val F1=%.3f)", model_type, run.name, best_f1)

    if isinstance(run, ModelRun):
        test_artifacts = _evaluate_classical_on_test(run, bundle, splits, config, y_val, y_test)
        return test_artifacts
    if isinstance(run, TransformerRun):
        test_artifacts = _evaluate_transformer_on_test(run, config, y_test)
        return test_artifacts
    raise AssertionError("Unexpected model run type.")


def _evaluate_classical_on_test(
    run: ModelRun,
    bundle: FeatureBundle,
    splits: SplitResult,
    config: Dict,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> BestModelArtifacts:
    X_val, X_test = _prepare_features_for_run(run, bundle, splits)

    calibration_methods = config["calibration"]["methods"]
    calibration_results = calibrate_prefit_model(run.estimator, X_val, y_val, calibration_methods)
    best_calibration = calibration_results[0] if calibration_results else None

    if best_calibration:
        estimator = best_calibration.estimator
        val_probs = best_calibration.val_probabilities
        chosen_method = best_calibration.method
    else:
        estimator = run.estimator
        chosen_method = None
        if hasattr(estimator, "predict_proba"):
            val_probs = estimator.predict_proba(X_val)[:, 1]
        else:
            val_scores = estimator.decision_function(X_val)
            val_probs = 1.0 / (1.0 + np.exp(-val_scores))

    val_metrics = compute_metrics(
        y_val,
        val_probs,
        metrics_list=config["evaluation"]["metrics"],
        threshold=run.threshold,
        positive_label=1,
    )

    if hasattr(estimator, "predict_proba"):
        test_probs = estimator.predict_proba(X_test)[:, 1]
    else:
        test_scores = estimator.decision_function(X_test)
        test_probs = 1.0 / (1.0 + np.exp(-test_scores))

    test_metrics = compute_metrics(
        y_test,
        test_probs,
        metrics_list=config["evaluation"]["metrics"],
        threshold=run.threshold,
        positive_label=1,
    )

    return BestModelArtifacts(
        name=run.name,
        model_type="classical",
        estimator=estimator,
        base_estimator=run.estimator,
        threshold=run.threshold,
        calibration_method=chosen_method,
        val_metrics=val_metrics,
        val_probabilities=val_probs,
        test_metrics=test_metrics,
        test_probabilities=test_probs,
        test_labels=y_test,
        feature_type=run.feature_type,
        extra={},
    )


def _prepare_features_for_run(run: ModelRun, bundle: FeatureBundle, splits: SplitResult):
    if run.feature_type == "tfidf+tabular":
        X_val = sparse.hstack([bundle.tfidf_val, bundle.tabular_val]).tocsr()
        X_test = sparse.hstack([bundle.tfidf_test, bundle.tabular_test]).tocsr()
    else:
        X_val = bundle.tabular_val
        X_test = bundle.tabular_test
    return X_val, X_test


def _evaluate_transformer_on_test(run: TransformerRun, config: Dict, y_test: np.ndarray) -> BestModelArtifacts:
    test_metrics = compute_metrics(
        run.test_labels,
        run.test_scores,
        metrics_list=config["evaluation"]["metrics"],
        threshold=run.threshold,
        positive_label=1,
    )
    return BestModelArtifacts(
        name=run.name,
        model_type="transformer",
        estimator=None,
        base_estimator=None,
        threshold=run.threshold,
        calibration_method=None,
        val_metrics=run.val_metrics,
        val_probabilities=run.val_scores,
        test_metrics=test_metrics,
        test_probabilities=run.test_scores,
        test_labels=y_test,
        feature_type="text",
        extra={
            "model_dir": str(run.model_dir),
            "tokenizer_dir": str(run.tokenizer_dir),
        },
    )


def _persist_artifacts(
    artifacts: BestModelArtifacts,
    bundle: FeatureBundle,
    config: Dict,
    config_path: Optional[Path],
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle_dir = ARTIFACTS_DIR / "features"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.tfidf_vectorizer, bundle_dir / "tfidf_vectorizer.joblib")
    joblib.dump(bundle.tabular_scaler, bundle_dir / "tabular_scaler.joblib")
    joblib.dump(bundle.feature_names, bundle_dir / "tabular_feature_names.joblib")

    if artifacts.estimator is not None:
        joblib.dump(artifacts.estimator, ARTIFACTS_DIR / "model.joblib")
    if artifacts.base_estimator is not None:
        joblib.dump(artifacts.base_estimator, ARTIFACTS_DIR / "base_model.joblib")

    metadata = {
        "model_name": artifacts.name,
        "model_type": artifacts.model_type,
        "feature_type": artifacts.feature_type,
        "calibration_method": artifacts.calibration_method,
        "threshold": artifacts.threshold,
        "gray_zone": {
            "width": config["gray_zone"]["width"],
            "positive_label": config["gray_zone"]["positive_label"],
            "negative_label": config["gray_zone"]["negative_label"],
            "review_label": config["gray_zone"]["review_label"],
        },
        "val_metrics": artifacts.val_metrics.values,
        "test_metrics": artifacts.test_metrics.values,
        "test_ece": expected_calibration_error(artifacts.test_labels, artifacts.test_probabilities),
        "extra": artifacts.extra,
    }
    with (ARTIFACTS_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    predictions_path = ARTIFACTS_DIR / "test_predictions.csv"
    import pandas as pd

    df = pd.DataFrame(
        {
            "prob_fraud": artifacts.test_probabilities,
            "label": artifacts.test_labels,
        }
    )

    width = config["gray_zone"]["width"]
    df["decision"] = df["prob_fraud"].apply(
        lambda p: classify_probability(
            p,
            threshold=artifacts.threshold,
            width=width,
            positive_label=config["gray_zone"]["positive_label"],
            negative_label=config["gray_zone"]["negative_label"],
            review_label=config["gray_zone"]["review_label"],
        )
    )
    df.to_csv(predictions_path, index=False)

    config_dump_path = ARTIFACTS_DIR / "config_used.yaml"
    dump_config(config, config_dump_path)

    if config_path:
        logger.info("Saved configuration snapshot to %s", config_dump_path)


def _prepare_benchmark_payload(df: pd.DataFrame) -> List[Dict[str, object]]:
    payload: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        record: Dict[str, object] = {}
        for key, value in row.items():
            if key in {"fraudulent", "text_all"}:
                continue
            if pd.isna(value):
                record[key] = None
            elif key in {"telecommuting", "has_company_logo", "has_questions"}:
                record[key] = int(value)
            else:
                record[key] = value
        payload.append(record)
    return payload


def _run_benchmarks(artifacts: BestModelArtifacts, config: Dict, splits: SplitResult) -> None:
    from time import perf_counter

    test_df = splits.test.copy()
    payload = _prepare_benchmark_payload(test_df)
    if not payload:
        logger.warning("Benchmark skipped: no payload rows available.")
        return

    batch_sizes = config.get("evaluation", {}).get("benchmark_batch_sizes", [1, 8, 32, 128])
    repeats = int(config.get("evaluation", {}).get("benchmark_repeats", 3))
    batch_sizes = [int(b) for b in batch_sizes]

    try:
        from spot_scam.inference.predictor import FraudPredictor
    except ImportError as exc:  # pragma: no cover
        logger.warning("Benchmark skipped: could not import predictor (%s)", exc)
        return

    predictor = FraudPredictor()
    predictor.predict(payload[: min(32, len(payload))])  # warm-up

    records: List[Dict[str, float]] = []
    for batch_size in batch_sizes:
        for repeat_idx in range(repeats):
            if batch_size <= len(payload):
                batch = random.sample(payload, k=batch_size)
            else:
                batch = random.choices(payload, k=batch_size)
            start = perf_counter()
            predictor.predict(batch)
            elapsed = perf_counter() - start
            latency_ms = elapsed * 1000
            throughput = batch_size / elapsed if elapsed > 0 else float("inf")
            records.append(
                {
                    "model_name": artifacts.name,
                    "model_type": artifacts.model_type,
                    "batch_size": batch_size,
                    "repeat": repeat_idx,
                    "latency_ms": latency_ms,
                    "throughput_rps": throughput,
                }
            )

    if not records:
        return

    df = pd.DataFrame(records)
    df.to_csv(TABLES_DIR / "benchmark_latency.csv", index=False)

    summary = (
        df.groupby("batch_size")
        .agg(
            mean_latency_ms=("latency_ms", "mean"),
            std_latency_ms=("latency_ms", "std"),
            mean_throughput_rps=("throughput_rps", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(TABLES_DIR / "benchmark_summary.csv", index=False)

    plot_latency_curve(
        summary["batch_size"].to_numpy(),
        summary["mean_latency_ms"].to_numpy(),
        summary["mean_throughput_rps"].to_numpy(),
        path=FIGS_DIR / "latency_throughput.png",
    )


def _generate_report_assets(
    artifacts: BestModelArtifacts,
    config: Dict,
    y_val: np.ndarray,
    splits: SplitResult,
    bundle: FeatureBundle,
) -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    plot_pr_curve(
        artifacts.test_labels,
        artifacts.test_probabilities,
        label=artifacts.name,
        path=FIGS_DIR / "pr_curve_test.png",
    )
    plot_calibration_curve(
        artifacts.test_labels,
        artifacts.test_probabilities,
        path=FIGS_DIR / "calibration_curve_test.png",
    )
    plot_confusion_matrix(
        artifacts.test_metrics.confusion_matrix,
        labels=["legit", "fraud"],
        path=FIGS_DIR / "confusion_matrix_test.png",
        normalize=False,
    )

    plot_score_distribution(
        artifacts.test_labels,
        artifacts.test_probabilities,
        path=FIGS_DIR / "score_distribution_test.png",
    )

    thresholds = np.linspace(0.05, 0.95, 19)
    sweep_records = []
    metric_curves = {"precision": [], "recall": [], "f1": []}
    for threshold in thresholds:
        result = compute_metrics(
            y_val,
            artifacts.val_probabilities,
            metrics_list=["precision", "recall", "f1"],
            threshold=float(threshold),
            positive_label=1,
        )
        sweep_records.append(
            {
                "threshold": float(threshold),
                "precision": result.values.get("precision", np.nan),
                "recall": result.values.get("recall", np.nan),
                "f1": result.values.get("f1", np.nan),
            }
        )
        for metric_name in metric_curves:
            metric_curves[metric_name].append(sweep_records[-1][metric_name])

    if sweep_records:
        sweep_df = pd.DataFrame(sweep_records)
        sweep_df.to_csv(TABLES_DIR / "threshold_metrics.csv", index=False)
        plot_threshold_sweep(
            thresholds,
            {k: np.array(v) for k, v in metric_curves.items()},
            path=FIGS_DIR / "threshold_sweep_val.png",
        )

    text_lengths = splits.test["text_all"].str.len().to_numpy(dtype=float)
    plot_probability_vs_feature(
        feature_values=text_lengths,
        probabilities=artifacts.test_probabilities,
        y_true=artifacts.test_labels,
        feature_name="Text length (chars)",
        path=FIGS_DIR / "probability_vs_length.png",
    )
    try:
        regression = stats.linregress(text_lengths, artifacts.test_probabilities)
        pd.DataFrame(
            [
                {
                    "feature": "text_length_chars",
                    "slope": regression.slope,
                    "intercept": regression.intercept,
                    "rvalue": regression.rvalue,
                    "pvalue": regression.pvalue,
                    "stderr": regression.stderr,
                }
            ]
        ).to_csv(TABLES_DIR / "probability_regression.csv", index=False)
    except ValueError as exc:  # pragma: no cover
        logger.warning("Regression analysis skipped: %s", exc)

    if artifacts.feature_type.startswith("tfidf") and artifacts.base_estimator is not None:
        try:
            top_terms = top_tfidf_terms(bundle.tfidf_vectorizer, artifacts.base_estimator, top_n=30)
            for sentiment, df in top_terms.items():
                df.to_csv(TABLES_DIR / f"top_terms_{sentiment}.csv", index=False)
        except Exception as exc:  # pragma: no cover - diagnostic
            logger.warning("Failed to compute TF-IDF term importance: %s", exc)

    if artifacts.feature_type == "tabular" and artifacts.base_estimator is not None:
        try:
            compute_tabular_shap(
                artifacts.base_estimator,
                bundle.tabular_test.toarray(),
                bundle.feature_names,
                sample_size=512,
                output_path=TABLES_DIR / "shap_summary.csv",
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Unable to compute SHAP values: %s", exc)

    freq_df = token_frequency_analysis(
        splits.test,
        artifacts.test_probabilities,
        top_k=30,
        threshold=artifacts.threshold,
    )
    freq_df.to_csv(TABLES_DIR / "token_frequency_analysis.csv", index=False)

    slice_metrics = compute_slice_metrics(
        splits.test,
        artifacts.test_probabilities,
        artifacts.test_labels,
        threshold=artifacts.threshold,
        slice_columns=config["evaluation"]["slice_columns"],
        metrics_list=config["evaluation"]["metrics"],
        min_count=config["evaluation"].get("slice_min_count", 30),
    )
    slices_df = slices_to_dataframe(slice_metrics)
    if not slices_df.empty:
        slices_df.to_csv(TABLES_DIR / "slice_metrics.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {
                "split": "validation",
                **artifacts.val_metrics.values,
            },
            {
                "split": "test",
                **artifacts.test_metrics.values,
            },
        ]
    )
    summary_df.to_csv(TABLES_DIR / "metrics_summary.csv", index=False)

    _run_benchmarks(artifacts, config, splits)

    report_metadata = {
        "model_name": artifacts.name,
        "model_type": artifacts.model_type,
        "calibration_method": artifacts.calibration_method,
        "threshold": artifacts.threshold,
        "gray_zone": config["gray_zone"],
    }
    render_markdown_report(
        report_metadata,
        metrics_summary=summary_df,
        slice_metrics=slices_df if not slices_df.empty else None,
        token_table=freq_df,
        output_path=EXPERIMENTS_DIR / "report.md",
    )


def entrypoint():
    app()


if __name__ == "__main__":
    entrypoint()
