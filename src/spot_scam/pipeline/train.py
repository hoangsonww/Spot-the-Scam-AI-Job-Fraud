from __future__ import annotations

import json
import random
import time
import os
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

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
from spot_scam.evaluation.metrics import (
    MetricResults,
    compute_metrics,
    expected_calibration_error,
    optimal_threshold,
)
from spot_scam.features.builders import FeatureBundle, build_feature_bundle
from spot_scam.models.classical import ModelRun, train_classical_models
from spot_scam.models.xgboost_model import XGBoostModel
from sklearn.calibration import CalibratedClassifierCV

if TYPE_CHECKING:  # avoid heavy imports unless actually training transformer
    from spot_scam.models.transformer import TransformerRun  # type: ignore
from spot_scam.tracking.logger import append_run_record
from spot_scam.tracking.feedback import load_feedback_dataframe
from spot_scam.policy.gray_zone import classify_probability
from spot_scam.utils.logging import configure_logging
from spot_scam.utils.paths import (
    ARTIFACTS_DIR,
    FIGS_DIR,
    TABLES_DIR,
    EXPERIMENTS_DIR,
    ensure_directories,
)
from spot_scam.evaluation.reporting import render_markdown_report
from spot_scam.explainability.textual import top_tfidf_terms, token_frequency_analysis
from spot_scam.export import log_model_to_mlflow, MLFlowExportError

app = typer.Typer(add_completion=False)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _apply_feedback_labels(df: pd.DataFrame, target_column: str) -> int:
    feedback_df = load_feedback_dataframe()
    if feedback_df.empty:
        return 0

    valid = feedback_df[feedback_df["reviewer_label"].isin(["fraud", "legit"])]
    if valid.empty:
        return 0

    valid = valid.sort_values("created_at").drop_duplicates(subset=["text_hash"], keep="last")
    label_map = {"fraud": 1, "legit": 0}
    overrides = 0
    for _, row in valid.iterrows():
        text_hash = str(row["text_hash"])
        label = label_map[str(row["reviewer_label"])]
        mask = df["text_hash"] == text_hash
        if mask.any():
            df.loc[mask, target_column] = label
            df.loc[mask, "_feedback_applied"] = True
            overrides += int(mask.sum())
    return overrides


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
    use_feedback: bool = typer.Option(
        False,
        "--use-feedback",
        help="Incorporate reviewer feedback when available (overridden by USE_FEEDBACK env).",
    ),
) -> None:
    """Execute the end-to-end training pipeline."""
    print("RUN FUNCTION ENTERED")
    config = load_config(config_path=config_path)
    use_feedback_env = os.getenv("USE_FEEDBACK", "").lower() in {"1", "true", "yes"}
    use_feedback_enabled = use_feedback or use_feedback_env

    set_global_seed(config["project"]["random_seed"])
    ensure_directories()

    logger.info("Configuration hash: %s", config_hash(config))
    if use_feedback_enabled:
        logger.info("Feedback integration enabled (USE_FEEDBACK).")

    raw_df = load_raw_dataset(config)
    processed_df, _ = preprocess_dataframe(raw_df, config)
    target_col = config["data"]["target_column"]
    processed_df["text_hash"] = processed_df["text_all"].apply(_hash_text)
    processed_df["_original_label"] = processed_df[target_col]
    processed_df["_feedback_applied"] = False

    feedback_override_count = 0
    if use_feedback_enabled:
        feedback_override_count = _apply_feedback_labels(processed_df, target_col)
        if feedback_override_count:
            logger.info("Applied reviewer feedback to %d samples.", feedback_override_count)
        else:
            logger.info(
                "No matching reviewer feedback hashes found; proceeding with baseline labels."
            )

    splits = create_splits(processed_df, config, persist=True)
    baseline_val_labels = splits.val["_original_label"].to_numpy()
    baseline_test_labels = splits.test["_original_label"].to_numpy()
    feedback_counts = {
        "train": int(splits.train["_feedback_applied"].sum()),
        "validation": int(splits.val["_feedback_applied"].sum()),
        "test": int(splits.test["_feedback_applied"].sum()),
    }
    for split_df in (splits.train, splits.val, splits.test):
        split_df.drop(
            columns=["_original_label", "_feedback_applied", "text_hash"],
            inplace=True,
            errors="ignore",
        )

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

    # ------------------------------------------------------------------
    # Train multiple XGBoost variants to seek higher validation/test F1.
    # Grid can be constrained via config for quicker experimentation.
    xgb_grid: List[Dict] = []
    xgb_conf = config["models"].get("xgboost", {})
    if xgb_conf.get("enabled", True):
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        spw_default = max(int(neg / max(pos, 1)), 1)  # scale_pos_weight heuristic

        def _values_from_conf(key: str, fallback: List):
            values = xgb_conf.get(key)
            return list(values) if values else fallback

        max_depths = _values_from_conf("max_depths", [4, 6, 8])
        learning_rates = _values_from_conf("learning_rates", [0.05, 0.03, 0.02])
        n_estimators_list = _values_from_conf("n_estimators_list", [600, 800, 1000])
        subsamples = _values_from_conf("subsamples", [0.9])
        colsample_bytrees = _values_from_conf("colsample_bytrees", [0.9])
        min_child_weights = _values_from_conf("min_child_weights", [1, 5, 10])
        reg_alphas = _values_from_conf("reg_alphas", [0.0, 0.1, 0.3])
        reg_lambdas = _values_from_conf("reg_lambdas", [1.0, 2.0, 5.0])
        spw_factors = _values_from_conf("scale_pos_weight_factors", [0.7, 1.0, 1.3])

        for max_depth in max_depths:
            for learning_rate in learning_rates:
                for n_estimators in n_estimators_list:
                    for subsample in subsamples:
                        for colsample in colsample_bytrees:
                            for min_child_weight in min_child_weights:
                                for reg_alpha in reg_alphas:
                                    for reg_lambda in reg_lambdas:
                                        for spw_factor in spw_factors:
                                            spw = max(int(spw_default * float(spw_factor)), 1)
                                            xgb_grid.append(
                                                {
                                                    "max_depth": int(max_depth),
                                                    "learning_rate": float(learning_rate),
                                                    "n_estimators": int(n_estimators),
                                                    "subsample": float(subsample),
                                                    "colsample_bytree": float(colsample),
                                                    "min_child_weight": int(min_child_weight),
                                                    "reg_alpha": float(reg_alpha),
                                                    "reg_lambda": float(reg_lambda),
                                                    "scale_pos_weight": spw,
                                                }
                                            )
    else:
        logger.info("Skipping XGBoost variants (disabled via config).")

    # Limit total variants to avoid excessive runtime if grid explodes
    MAX_XGB_VARIANTS = int(xgb_conf.get("max_variants", 12))
    if len(xgb_grid) > MAX_XGB_VARIANTS and MAX_XGB_VARIANTS > 0:
        # Simple down-select: keep first MAX_XGB_VARIANTS evenly spread
        step = len(xgb_grid) / MAX_XGB_VARIANTS
        xgb_grid = [xgb_grid[int(i * step)] for i in range(MAX_XGB_VARIANTS)]

    xgb_result = None
    for params in xgb_grid:
        try:
            start_xgb = time.time()
            # Construct model instance on-the-fly overriding base estimator hyperparams
            xgb_model = XGBoostModel(config)
            # Override internal base_estimator parameters
            xgb_model.base_estimator.set_params(
                **params, eval_metric="logloss", tree_method="hist", use_label_encoder=False
            )
            xgb_result = xgb_model.fit(bundle, y_train, y_val)
            xgb_train_time = time.time() - start_xgb
            variant_name = (
                f"XGBOOST_d{params['max_depth']}_lr{params['learning_rate']}_ne{params['n_estimators']}"
                f"_mcw{params['min_child_weight']}_a{params['reg_alpha']}_l{params['reg_lambda']}_spw{params['scale_pos_weight']}"
            )
            logger.info(
                "%s trained (%.1fs) F1=%.3f P=%.3f R=%.3f Thr=%.3f",
                variant_name,
                xgb_train_time,
                xgb_result.val_metrics.values.get("f1", float("nan")),
                xgb_result.val_metrics.values.get("precision", float("nan")),
                xgb_result.val_metrics.values.get("recall", float("nan")),
                xgb_result.threshold if xgb_result.threshold is not None else float("nan"),
            )
            classical_runs.append(
                ModelRun(
                    name=variant_name,
                    estimator=xgb_result.calibrated_estimator,
                    val_scores=xgb_result.val_probabilities,
                    val_metrics=xgb_result.val_metrics,
                    train_time=xgb_train_time,
                    config=params,
                    threshold=xgb_result.threshold,
                    feature_type="tfidf+tabular",
                )
            )
            # Persist each variant artifacts in its own subdirectory for potential analysis
            try:
                variant_dir = ARTIFACTS_DIR / "xgboost_variants" / variant_name
                variant_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(xgb_result.calibrated_estimator, variant_dir / "model.joblib")
                joblib.dump(xgb_result.base_estimator, variant_dir / "base_model.joblib")
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to persist artifacts for %s: %s", variant_name, exc)
        except Exception as exc:  # pragma: no cover - continue other variants
            logger.warning("XGBoost variant %s failed: %s", params, exc)

    # Persist XGBoost artifacts regardless of selection outcome
    try:
        if xgb_result is not None:
            xgb_dir = ARTIFACTS_DIR / "xgboost"
            xgb_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(xgb_result.calibrated_estimator, xgb_dir / "model.joblib")
            joblib.dump(xgb_result.base_estimator, xgb_dir / "base_model.joblib")
    except Exception as exc:  # pragma: no cover - non critical
        logger.warning("Failed to persist XGBoost artifacts: %s", exc)
    classical_artifacts = [
        _evaluate_classical_on_test(run, bundle, splits, config, y_val, y_test)
        for run in classical_runs
    ]

    transformer_artifact: Optional[BestModelArtifacts] = None
    if not skip_transformer:
        # Lazy import to avoid pulling heavy deps during classical-only runs
        from spot_scam.models.transformer import train_transformer_model

        transformer_run = train_transformer_model(
            splits.train,
            splits.val,
            splits.test,
            config,
            output_dir=ARTIFACTS_DIR,
        )
        transformer_artifact = _evaluate_transformer_on_test(transformer_run, config, y_test)

    candidate_artifacts: List[BestModelArtifacts] = list(classical_artifacts)

    # ------------------------------------------------------------------
    # Probabilistic ensemble of top K tfidf+tabular classical models.
    try:
        TOP_K = 3
        tfidf_artifacts = [a for a in classical_artifacts if a.feature_type == "tfidf+tabular"]
        tfidf_sorted = sorted(
            tfidf_artifacts,
            key=lambda a: a.val_metrics.values.get("f1", -np.inf),
            reverse=True,
        )
        ensemble_components = tfidf_sorted[:TOP_K]
        if len(ensemble_components) >= 2:
            # Average calibrated probabilities
            val_stack = np.vstack([c.val_probabilities for c in ensemble_components])
            test_stack = np.vstack([c.test_probabilities for c in ensemble_components])
            ensemble_val = val_stack.mean(axis=0)
            ensemble_test = test_stack.mean(axis=0)
            ensemble_threshold = optimal_threshold(
                y_val,
                ensemble_val,
                metric=config["evaluation"]["thresholds"]["optimize_metric"],
            )
            ensemble_val_metrics = compute_metrics(
                y_val,
                ensemble_val,
                metrics_list=config["evaluation"]["metrics"],
                threshold=ensemble_threshold,
                positive_label=1,
            )
            ensemble_test_metrics = compute_metrics(
                y_test,
                ensemble_test,
                metrics_list=config["evaluation"]["metrics"],
                threshold=ensemble_threshold,
                positive_label=1,
            )
            ensemble_artifact = BestModelArtifacts(
                name="ensemble_top3",
                model_type="classical",  # treat as classical for inference compatibility
                estimator=None,
                base_estimator=None,
                threshold=ensemble_threshold,
                calibration_method=None,
                val_metrics=ensemble_val_metrics,
                val_probabilities=ensemble_val,
                test_metrics=ensemble_test_metrics,
                test_probabilities=ensemble_test,
                test_labels=y_test,
                feature_type="tfidf+tabular",
                extra={
                    "components": [c.name for c in ensemble_components],
                    "is_ensemble": True,
                },
            )
            candidate_artifacts.append(ensemble_artifact)
            logger.info(
                "Ensemble (top %d) F1=%.3f Precision=%.3f Recall=%.3f",
                TOP_K,
                ensemble_val_metrics.values.get("f1", np.nan),
                ensemble_val_metrics.values.get("precision", np.nan),
                ensemble_val_metrics.values.get("recall", np.nan),
            )

            # Weighted ensemble via simple grid search on validation
            weights = None
            best_w_val = ensemble_val
            best_w_test = ensemble_test
            best_w_threshold = ensemble_threshold
            best_w_metrics = ensemble_val_metrics
            K = len(ensemble_components)
            if K == 2:
                import numpy as _np

                for w in [i / 10.0 for i in range(1, 10)]:
                    w_vec = _np.array([w, 1 - w])[:, None]
                    val_mix = (val_stack[:2] * w_vec).sum(axis=0)
                    thr = optimal_threshold(
                        y_val, val_mix, metric=config["evaluation"]["thresholds"]["optimize_metric"]
                    )
                    m = compute_metrics(
                        y_val,
                        val_mix,
                        metrics_list=config["evaluation"]["metrics"],
                        threshold=thr,
                        positive_label=1,
                    )
                    if m.values.get("f1", -_np.inf) > best_w_metrics.values.get("f1", -_np.inf):
                        weights = [w, 1 - w]
                        best_w_metrics = m
                        best_w_threshold = thr
                        best_w_val = val_mix
                        best_w_test = (test_stack[:2] * w_vec).sum(axis=0)
            elif K >= 3:
                import numpy as _np

                grid = [i / 10.0 for i in range(0, 11)]
                for w1 in grid:
                    for w2 in grid:
                        if w1 + w2 > 1.0:
                            continue
                        w3 = 1.0 - w1 - w2
                        w_vec = _np.array([w1, w2, w3])[:, None]
                        val_mix = (val_stack[:3] * w_vec).sum(axis=0)
                        thr = optimal_threshold(
                            y_val,
                            val_mix,
                            metric=config["evaluation"]["thresholds"]["optimize_metric"],
                        )
                        m = compute_metrics(
                            y_val,
                            val_mix,
                            metrics_list=config["evaluation"]["metrics"],
                            threshold=thr,
                            positive_label=1,
                        )
                        if m.values.get("f1", -_np.inf) > best_w_metrics.values.get("f1", -_np.inf):
                            weights = [w1, w2, w3]
                            best_w_metrics = m
                            best_w_threshold = thr
                            best_w_val = val_mix
                            best_w_test = (test_stack[:3] * w_vec).sum(axis=0)
            if weights is not None:
                w_test_metrics = compute_metrics(
                    y_test,
                    best_w_test,
                    metrics_list=config["evaluation"]["metrics"],
                    threshold=best_w_threshold,
                    positive_label=1,
                )
                w_artifact = BestModelArtifacts(
                    name="ensemble_weighted_top3",
                    model_type="classical",
                    estimator=None,
                    base_estimator=None,
                    threshold=best_w_threshold,
                    calibration_method=None,
                    val_metrics=best_w_metrics,
                    val_probabilities=best_w_val,
                    test_metrics=w_test_metrics,
                    test_probabilities=best_w_test,
                    test_labels=y_test,
                    feature_type="tfidf+tabular",
                    extra={
                        "components": [c.name for c in ensemble_components[:3]],
                        "weights": weights,
                        "is_ensemble": True,
                    },
                )
                candidate_artifacts.append(w_artifact)
                logger.info(
                    "Weighted ensemble F1=%.3f (weights=%s)",
                    best_w_metrics.values.get("f1", np.nan),
                    weights,
                )
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to build ensemble: %s", exc)
    if transformer_artifact is not None:
        candidate_artifacts.append(transformer_artifact)

    best_model_artifacts = _select_best_artifact(candidate_artifacts)

    if use_feedback_enabled:
        best_model_artifacts.extra.setdefault("feedback", {})
        best_model_artifacts.extra["feedback"].update(
            {
                "used": True,
                "overrides": int(feedback_override_count),
                "counts": feedback_counts,
            }
        )

    _persist_artifacts(best_model_artifacts, bundle, config, config_path)
    _generate_report_assets(
        best_model_artifacts,
        config=config,
        y_val=y_val,
        splits=splits,
        bundle=bundle,
        baseline_val_labels=baseline_val_labels,
        baseline_test_labels=baseline_test_labels,
        feedback_counts=feedback_counts if use_feedback_enabled else None,
        feedback_overrides=feedback_override_count if use_feedback_enabled else 0,
        feedback_enabled=use_feedback_enabled,
    )
    for artifact in candidate_artifacts:
        append_run_record(artifact, config)
    try:
        log_model_to_mlflow(best_model_artifacts, bundle, config, splits)
    except MLFlowExportError as exc:
        logger.warning("Skipping MLflow export: %s", exc)
    typer.echo("Training complete. Best model: " + best_model_artifacts.name)


def _select_best_artifact(artifacts: List[BestModelArtifacts]) -> BestModelArtifacts:
    if not artifacts:
        raise RuntimeError("No candidate models were trained.")

    best_artifact = max(
        artifacts,
        key=lambda item: item.val_metrics.values.get("f1", -np.inf),
    )
    logger.info(
        "Selected best model (%s): %s (val F1=%.3f)",
        best_artifact.model_type,
        best_artifact.name,
        best_artifact.val_metrics.values.get("f1", np.nan),
    )
    return best_artifact


def _infer_calibration_method(estimator: CalibratedClassifierCV) -> str:
    """
    Sklearn 1.4+ renamed the attribute on _CalibratedClassifier from `calibrator`
    to `calibrators`. Inspect both for compatibility.
    """
    if not estimator.calibrated_classifiers_:
        return "isotonic"
    calibrated = estimator.calibrated_classifiers_[0]
    cal = getattr(calibrated, "calibrator", None)
    if cal is None:
        calibrators = getattr(calibrated, "calibrators", None)
        if calibrators:
            cal = calibrators[0]
    if cal is None:
        return "isotonic"
    return "platt" if cal.__class__.__name__ == "LogisticRegression" else "isotonic"


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
    # Skip second calibration for any pre-calibrated estimator
    if isinstance(run.estimator, CalibratedClassifierCV):
        calibration_results = []
    else:
        calibration_results = calibrate_prefit_model(
            run.estimator, X_val, y_val, calibration_methods
        )
    best_calibration = calibration_results[0] if calibration_results else None

    if best_calibration:
        estimator = best_calibration.estimator
        val_probs = best_calibration.val_probabilities
        chosen_method = best_calibration.method
    else:
        estimator = run.estimator
        # Attempt to record calibration method if estimator is already calibrated
        if isinstance(estimator, CalibratedClassifierCV):
            chosen_method = _infer_calibration_method(estimator)
            val_probs = estimator.predict_proba(X_val)[:, 1]
        else:
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

    extra = {}
    if run.name == "XGBOOST":
        extra["artifact_subdir"] = "xgboost"
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
        extra=extra,
    )


def _prepare_features_for_run(run: ModelRun, bundle: FeatureBundle, splits: SplitResult):
    if run.feature_type == "tfidf+tabular":
        X_val = sparse.hstack([bundle.tfidf_val, bundle.tabular_val]).tocsr()
        X_test = sparse.hstack([bundle.tfidf_test, bundle.tabular_test]).tocsr()
    else:
        X_val = bundle.tabular_val
        X_test = bundle.tabular_test
    return X_val, X_test


def _evaluate_transformer_on_test(
    run: "TransformerRun", config: Dict, y_test: np.ndarray
) -> BestModelArtifacts:
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
    *,
    baseline_val_labels: Optional[np.ndarray] = None,
    baseline_test_labels: Optional[np.ndarray] = None,
    feedback_counts: Optional[Dict[str, int]] = None,
    feedback_overrides: int = 0,
    feedback_enabled: bool = False,
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
            from spot_scam.explainability.shapley import compute_tabular_shap

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

    if feedback_enabled and baseline_test_labels is not None:
        metrics_list = config["evaluation"]["metrics"]
        delta_rows = []
        split_specs = [
            ("validation", baseline_val_labels, artifacts.val_probabilities, artifacts.val_metrics),
            ("test", baseline_test_labels, artifacts.test_probabilities, artifacts.test_metrics),
        ]
        for split_name, baseline_labels, probs, metrics_obj in split_specs:
            if baseline_labels is None:
                continue
            baseline_metrics = compute_metrics(baseline_labels, probs, metrics_list)
            for metric_name, baseline_value in baseline_metrics.values.items():
                feedback_value = metrics_obj.values.get(metric_name)
                if baseline_value is None or feedback_value is None:
                    continue
                delta_rows.append(
                    {
                        "split": split_name,
                        "metric": metric_name,
                        "baseline": baseline_value,
                        "with_feedback": feedback_value,
                        "delta": feedback_value - baseline_value,
                    }
                )
        if delta_rows:
            pd.DataFrame(delta_rows).to_csv(TABLES_DIR / "metrics_with_feedback.csv", index=False)

        baseline_test_df = splits.test.copy()
        baseline_test_df[config["data"]["target_column"]] = baseline_test_labels
        baseline_slice_metrics = compute_slice_metrics(
            baseline_test_df,
            artifacts.test_probabilities,
            baseline_test_labels,
            threshold=artifacts.threshold,
            slice_columns=config["evaluation"]["slice_columns"],
            metrics_list=metrics_list,
            min_count=config["evaluation"].get("slice_min_count", 30),
        )
        baseline_slice_df = slices_to_dataframe(baseline_slice_metrics)
        if not baseline_slice_df.empty:
            baseline_slice_df.to_csv(TABLES_DIR / "slice_metrics_baseline.csv", index=False)
        if not slices_df.empty and not baseline_slice_df.empty:
            merged = baseline_slice_df.merge(
                slices_df,
                on=["slice", "category"],
                suffixes=("_baseline", "_feedback"),
            )
            for metric_name in ["f1", "precision", "recall"]:
                baseline_col = f"{metric_name}_baseline"
                feedback_col = f"{metric_name}_feedback"
                if baseline_col in merged.columns and feedback_col in merged.columns:
                    merged[f"{metric_name}_delta"] = merged[feedback_col] - merged[baseline_col]
            merged.to_csv(TABLES_DIR / "slice_metrics_feedback_delta.csv", index=False)

        if feedback_counts:
            pd.DataFrame(
                [
                    {"split": split_name, "overrides": count}
                    for split_name, count in feedback_counts.items()
                ]
            ).to_csv(TABLES_DIR / "feedback_counts.csv", index=False)

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
