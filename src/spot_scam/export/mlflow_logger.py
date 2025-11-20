from __future__ import annotations

import json
import os
import warnings
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

try:  # pragma: no cover
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None  # type: ignore[assignment]
import yaml

from spot_scam.data.preprocess import preprocess_dataframe
from spot_scam.features.builders import FeatureBundle
from spot_scam.features.tabular import create_tabular_features
from spot_scam.policy.gray_zone import classify_probability
from spot_scam.evaluation.calibration import IsotonicCalibratedModel
from spot_scam.config.loader import config_hash
from spot_scam.utils.logging import configure_logging
from spot_scam.utils.paths import ARTIFACTS_DIR

logger = configure_logging(__name__)

warnings.filterwarnings(
    "ignore",
    message=".*Add type hints to the `predict` method.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Unsupported type hint.*",
    category=UserWarning,
)

API_INPUT_COLUMNS: List[Tuple[str, str]] = [
    ("title", "string"),
    ("location", "string"),
    ("department", "string"),
    ("salary_range", "string"),
    ("company_profile", "string"),
    ("description", "string"),
    ("requirements", "string"),
    ("benefits", "string"),
    ("telecommuting", "long"),
    ("has_company_logo", "long"),
    ("has_questions", "long"),
    ("employment_type", "string"),
    ("required_experience", "string"),
    ("required_education", "string"),
    ("industry", "string"),
    ("function", "string"),
]

DEFAULT_NUMERIC_VALUES = {
    "telecommuting": 0,
    "has_company_logo": 0,
    "has_questions": 0,
}

try:  # pragma: no cover
    import mlflow
    from mlflow import pyfunc
    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import ColSpec, Schema
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore[assignment]
    pyfunc = None  # type: ignore[assignment]
    ModelSignature = None  # type: ignore[assignment]
    Schema = None  # type: ignore[assignment]
    ColSpec = None  # type: ignore[assignment]


if mlflow:
    from mlflow.exceptions import MlflowException  # pragma: no cover
else:  # pragma: no cover
    MlflowException = RuntimeError

try:  # pragma: no cover
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except Exception:  # pragma: no cover
    convert_sklearn = None  # type: ignore[assignment]
    FloatTensorType = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from onnxmltools import convert_lightgbm
except Exception:  # pragma: no cover
    convert_lightgbm = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from onnxruntime import InferenceSession, SessionOptions
    from onnxruntime.quantization import QuantType, quantize_dynamic
except Exception:  # pragma: no cover
    InferenceSession = None  # type: ignore[assignment]
    SessionOptions = None  # type: ignore[assignment]
    quantize_dynamic = None  # type: ignore[assignment]
    QuantType = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from optimum.onnxruntime import ORTModelForSequenceClassification
except Exception:  # pragma: no cover
    ORTModelForSequenceClassification = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore[assignment]

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from spot_scam.pipeline.train import BestModelArtifacts
    from spot_scam.data.split import SplitResult


class MLFlowExportError(RuntimeError):
    """Raised when MLflow export fails."""


@dataclass
class ExportedModelArtifacts:
    python_model: Any
    artifacts: Dict[str, str]
    signature: Optional[ModelSignature]
    input_example: Optional[Any]
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    tags: Dict[str, str]


if pyfunc is not None:  # pragma: no branch

    class SpotScamPyfuncModel(pyfunc.PythonModel):  # type: ignore[misc]
        def __init__(self, variant: str, prefer_quantized: bool = True):
            self.variant = variant
            self.prefer_quantized = prefer_quantized
            self._use_quantized = prefer_quantized
            self._session: Optional[Any] = None
            self._quantized_session: Optional[Any] = None
            self._tokenizer = None
            self._vectorizer = None
            self._scaler = None
            self._feature_names: Optional[Tuple[str, ...]] = None
            self._feature_type = "tfidf+tabular"
            self._input_name: Optional[str] = None
            self._output_names: List[str] = []
            self._metadata: Dict[str, Any] = {}
            self._config: Dict[str, Any] = {}
            self._gray_zone: Dict[str, Any] = {}
            self._threshold: float = 0.5
            self._calibration: Dict[str, Any] = {"method": "none"}
            self._calibration_model = None
            self._max_length: Optional[int] = None

        def load_context(self, context: pyfunc.PythonModelContext) -> None:  # type: ignore[override]
            if "metadata" in context.artifacts:
                with open(context.artifacts["metadata"], "r", encoding="utf-8") as handle:
                    self._metadata = json.load(handle)
            if "config" in context.artifacts:
                with open(context.artifacts["config"], "r", encoding="utf-8") as handle:
                    self._config = yaml.safe_load(handle)

            if self._metadata:
                self._threshold = float(self._metadata.get("threshold", 0.5))
                self._gray_zone = self._metadata.get("gray_zone", {})
                self._feature_type = self._metadata.get("feature_type", "tfidf+tabular")

            if "calibration" in context.artifacts:
                with open(context.artifacts["calibration"], "r", encoding="utf-8") as handle:
                    self._calibration = json.load(handle)
                artifact_key = self._calibration.get("artifact_key")
                if artifact_key and artifact_key in context.artifacts:
                    self._calibration_model = joblib.load(context.artifacts[artifact_key])

            prefer_quantized = self.prefer_quantized or os.getenv(
                "SPOT_SCAM_USE_QUANTIZED", ""
            ).lower() in {
                "1",
                "true",
                "yes",
            }
            self._use_quantized = prefer_quantized

            if self.variant == "classical":
                self._vectorizer = joblib.load(context.artifacts["vectorizer"])
                self._scaler = joblib.load(context.artifacts["scaler"])
                self._feature_names = tuple(joblib.load(context.artifacts["feature_names"]))
                use_quant_path = prefer_quantized and "onnx_quantized" in context.artifacts
                base_model_path = context.artifacts["onnx_model"]
                if use_quant_path:
                    self._session = _create_session(context.artifacts["onnx_quantized"])
                    self._quantized_session = self._session
                else:
                    self._session = _create_session(base_model_path)
                    if "onnx_quantized" in context.artifacts:
                        self._quantized_session = _create_session(
                            context.artifacts["onnx_quantized"]
                        )
                    else:
                        self._quantized_session = None
                        self._use_quantized = False
                self._input_name = self._session.get_inputs()[0].name
                self._output_names = [out.name for out in self._session.get_outputs()]
            elif self.variant == "transformer":
                if AutoTokenizer is None:
                    raise MLFlowExportError(
                        "transformers dependency is required for transformer pyfunc model."
                    )
                tokenizer_dir = context.artifacts["tokenizer"]
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
                self._max_length = (
                    self._config.get("models", {}).get("transformer", {}).get("max_length", 128)
                )
                use_quant_path = prefer_quantized and "onnx_quantized" in context.artifacts
                base_model_path = context.artifacts["onnx_model"]
                if use_quant_path:
                    self._session = _create_session(context.artifacts["onnx_quantized"])
                    self._quantized_session = self._session
                else:
                    self._session = _create_session(base_model_path)
                    if "onnx_quantized" in context.artifacts:
                        self._quantized_session = _create_session(
                            context.artifacts["onnx_quantized"]
                        )
                    else:
                        self._quantized_session = None
                        self._use_quantized = False
                self._output_names = [out.name for out in self._session.get_outputs()]
            else:  # pragma: no cover - guarded earlier
                raise MLFlowExportError(f"Unsupported variant: {self.variant}")

        def __getstate__(self) -> Dict[str, Any]:  # pragma: no cover - pickling helper
            state = self.__dict__.copy()
            state["_session"] = None
            state["_quantized_session"] = None
            return state

        def __setstate__(self, state: Dict[str, Any]) -> None:  # pragma: no cover - pickling helper
            self.__dict__.update(state)

        def predict(  # type: ignore[override]
            self,
            context: pyfunc.PythonModelContext,
            model_input: List[Any],
        ) -> List[Any]:
            if isinstance(model_input, pd.DataFrame):
                records = model_input.to_dict(orient="records")
            elif isinstance(model_input, dict):
                records = [model_input]
            elif (
                isinstance(model_input, list)
                and model_input
                and isinstance(model_input[0], pd.DataFrame)
            ):
                records = pd.concat(model_input, ignore_index=True).to_dict(orient="records")
            else:
                records = list(model_input)

            if not records:
                return []

            df = pd.DataFrame(records)
            df = self._ensure_columns(df)

            if self.variant == "classical":
                prediction_output = self._predict_classical(df)
            else:
                prediction_output = self._predict_transformer(df)

            probabilities = np.clip(prediction_output["probabilities"], 1e-6, 1 - 1e-6)
            threshold = float(self._threshold)
            width = float(self._gray_zone.get("width", 0.0))
            positive_label = self._gray_zone.get("positive_label", "fraud")
            negative_label = self._gray_zone.get("negative_label", "legit")
            review_label = self._gray_zone.get("review_label", "review")

            binary = (probabilities >= threshold).astype(int)
            decisions = [
                classify_probability(
                    float(p),
                    threshold=threshold,
                    width=width,
                    positive_label=positive_label,
                    negative_label=negative_label,
                    review_label=review_label,
                )
                for p in probabilities
            ]

            explanations = prediction_output.get("explanations", [{} for _ in probabilities])

            return [
                {
                    "probability_fraud": float(probabilities[idx]),
                    "binary_label": int(binary[idx]),
                    "decision": decisions[idx],
                    "threshold": float(threshold),
                    "explanation": json.dumps(explanations[idx]),
                }
                for idx in range(len(probabilities))
            ]

        def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
            for column, dtype in API_INPUT_COLUMNS:
                if column not in df.columns:
                    df[column] = DEFAULT_NUMERIC_VALUES.get(column, None)
            for col, default in DEFAULT_NUMERIC_VALUES.items():
                df[col] = df[col].fillna(default).astype(int, errors="ignore")
            return df

        def _predict_classical(self, df: pd.DataFrame) -> np.ndarray:
            if self._vectorizer is None or self._scaler is None or self._session is None:
                raise MLFlowExportError("Classical predictor state not fully initialised.")

            processed_df, _ = preprocess_dataframe(df, self._config)
            tfidf = self._vectorizer.transform(processed_df["text_all"])
            tabular = create_tabular_features(processed_df, self._config)
            if self._feature_names:
                tabular = tabular.reindex(columns=self._feature_names, fill_value=0.0)
            tabular_scaled = self._scaler.transform(tabular.astype(np.float32))
            tabular_sparse = sparse.csr_matrix(tabular_scaled)

            if self._feature_type == "tfidf+tabular":
                feature_matrix = sparse.hstack([tfidf, tabular_sparse]).tocsr()
            elif self._feature_type == "tfidf":
                feature_matrix = tfidf.tocsr()
            else:
                feature_matrix = tabular_sparse.tocsr()

            dense_input = feature_matrix.astype(np.float32).toarray()
            session = self._choose_session()
            outputs = session.run(None, {self._input_name: dense_input})  # type: ignore[arg-type]

            probs = _extract_probabilities(outputs, self._output_names)
            raw_scores = _extract_raw_scores(outputs, self._output_names, probs)
            calibrated = self._apply_calibration(probs, raw_scores)
            return calibrated

        def _predict_transformer(self, df: pd.DataFrame) -> np.ndarray:
            if self._tokenizer is None or self._session is None:
                raise MLFlowExportError("Transformer predictor state not fully initialised.")

            processed_df, _ = preprocess_dataframe(df, self._config)
            texts = processed_df["text_all"].tolist()
            tokenizer_kwargs = {
                "padding": True,
                "truncation": True,
                "max_length": self._max_length or 128,
                "return_tensors": "np",
            }
            encoded = self._tokenizer(texts, **tokenizer_kwargs)
            session = self._choose_session()
            ort_inputs = {}
            for node in session.get_inputs():
                name = node.name
                if name in encoded:
                    tensor = encoded[name]
                else:
                    alt_name = name.replace(":", "_")
                    if alt_name in encoded:
                        tensor = encoded[alt_name]
                    else:
                        raise MLFlowExportError(f"Tokenizer output missing required input: {name}")
                ort_inputs[name] = tensor.astype(np.int64)
            outputs = session.run(None, ort_inputs)
            logits = _extract_logits(outputs, self._output_names)
            probabilities = _softmax(logits)[:, 1]
            return probabilities

        def _apply_calibration(
            self, probabilities: np.ndarray, raw_scores: np.ndarray
        ) -> np.ndarray:
            method = (self._calibration or {}).get("method", "none")
            if method == "none":
                return probabilities
            if method == "platt":
                coef = float(self._calibration.get("coef", 1.0))
                intercept = float(self._calibration.get("intercept", 0.0))
                scores = raw_scores
                calibrated = 1.0 / (1.0 + np.exp(-(coef * scores + intercept)))
                return calibrated
            if method == "isotonic":
                if self._calibration_model is None:
                    raise MLFlowExportError("Isotonic calibration artefact missing.")
                return np.asarray(self._calibration_model.predict(raw_scores))  # type: ignore[call-arg, no-any-return]
            logger.warning(
                "Unknown calibration method '%s'; defaulting to uncalibrated probabilities.", method
            )
            return probabilities

        def _choose_session(self):
            if self._use_quantized and self._quantized_session is not None:
                return self._quantized_session
            if self._session is None:
                raise MLFlowExportError("Inference session not initialised.")
            return self._session

else:  # pragma: no cover - mlflow optional dependency

    class SpotScamPyfuncModel:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise MLFlowExportError("MLflow pyfunc dependencies are not installed.")


def log_model_to_mlflow(
    best_model: "BestModelArtifacts",
    bundle: FeatureBundle,
    config: Dict[str, Any],
    splits: "SplitResult",
) -> None:
    mlflow_conf = config.get("mlflow", {})
    if not mlflow_conf.get("enabled", False):
        logger.info("MLflow export disabled in configuration; skipping model registry step.")
        return

    if mlflow is None or pyfunc is None:
        logger.warning(
            "MLflow dependencies are not available. Install optional dependencies to enable export."
        )
        return

    tracking_uri = mlflow_conf.get("tracking_uri")
    registry_uri = mlflow_conf.get("registry_uri")
    experiment_name = mlflow_conf.get("experiment") or config["project"]["name"]

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)

    mlflow.set_experiment(experiment_name)
    export_root = Path(tempfile.mkdtemp(prefix="spot_scam_onnx_"))
    logger.info("Exporting model artifacts to temporary directory: %s", export_root)

    if best_model.extra.get("is_ensemble"):
        logger.warning(
            "MLflow export skipped: %s is an ensemble model (ONNX export not supported).",
            best_model.name,
        )
        return

    if (
        best_model.model_type == "classical"
        and best_model.base_estimator is None
        and best_model.estimator is None
    ):
        logger.warning(
            "MLflow export skipped: %s has no underlying estimator (e.g., ensemble probabilities only).",
            best_model.name,
        )
        return

    try:
        exported = _prepare_export_bundle(
            best_model=best_model,
            bundle=bundle,
            config=config,
            export_root=export_root,
            mlflow_conf=mlflow_conf,
            splits=splits,
        )

        run_name = best_model.name
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(exported.params)
            mlflow.log_metrics(exported.metrics)
            if exported.tags:
                mlflow.set_tags(exported.tags)

            requirement_overrides = [
                "onnxruntime>=1.17.0",
                "onnx>=1.15.0",
                "numpy>=1.23.0",
                "pandas>=2.0.0",
                "scikit-learn>=1.2.0",
                "transformers>=4.38.0",
                "optimum[onnxruntime]>=1.17.0",
            ]

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Add type hints to the `predict` method to enable data validation",
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Unsupported type hint",
                )
                log_kwargs = {
                    "artifact_path": "model",
                    "python_model": exported.python_model,
                    "artifacts": {key: str(path) for key, path in exported.artifacts.items()},
                    "pip_requirements": requirement_overrides,
                }
                if exported.signature is not None:
                    log_kwargs["signature"] = exported.signature
                if exported.input_example is not None:
                    log_kwargs["input_example"] = exported.input_example

                pyfunc.log_model(**log_kwargs)  # type: ignore[union-attr]

            logger.info("Logged MLflow model for %s", run_name)
    except Exception as exc:  # pragma: no cover - guard main training loop
        logger.exception("Failed to export model to MLflow: %s", exc)
        raise MLFlowExportError(str(exc)) from exc
    finally:
        shutil.rmtree(export_root, ignore_errors=True)


def _prepare_export_bundle(
    best_model: "BestModelArtifacts",
    bundle: FeatureBundle,
    config: Dict[str, Any],
    export_root: Path,
    mlflow_conf: Dict[str, Any],
    splits: "SplitResult",
) -> ExportedModelArtifacts:
    if best_model.model_type == "classical":
        return _prepare_classical_export(
            best_model, bundle, config, export_root, mlflow_conf, splits
        )
    if best_model.model_type == "transformer":
        return _prepare_transformer_export(best_model, config, export_root, mlflow_conf, splits)
    raise MLFlowExportError(f"Unsupported model type for MLflow export: {best_model.model_type}")


def _create_session(model_path: str):
    if InferenceSession is None or SessionOptions is None:
        raise MLFlowExportError("onnxruntime dependency is required to load ONNX models.")
    opts = SessionOptions()
    graph_level = getattr(SessionOptions, "GRAPH_OPT_LEVEL_ALL", None)
    if graph_level is not None:
        opts.graph_optimization_level = graph_level
    return InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])


def _extract_probabilities(outputs: List[np.ndarray], output_names: List[str]) -> np.ndarray:
    if "probabilities" in output_names:
        probs = outputs[output_names.index("probabilities")]
    else:
        probs = outputs[-1]
    probs = np.asarray(probs)
    if probs.ndim == 1:
        return probs
    if probs.shape[1] == 1:
        return probs.ravel()
    return probs[:, -1]


def _extract_raw_scores(
    outputs: List[np.ndarray],
    output_names: List[str],
    probabilities: np.ndarray,
) -> np.ndarray:
    if "scores" in output_names:
        scores = np.asarray(outputs[output_names.index("scores")]).ravel()
        return scores
    eps = 1e-6
    probs = np.clip(probabilities, eps, 1 - eps)
    return np.log(probs / (1 - probs))


def _extract_logits(outputs: List[np.ndarray], output_names: List[str]) -> np.ndarray:
    if "logits" in output_names:
        logits = outputs[output_names.index("logits")]
    else:
        logits = outputs[0]
    return np.asarray(logits)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def _serialize_calibration(
    best_model: "BestModelArtifacts", export_root: Path
) -> Tuple[Dict[str, Any], Optional[Path]]:
    calibration = {"method": best_model.calibration_method or "none"}
    calibrator_path: Optional[Path] = None
    estimator = best_model.estimator

    if calibration["method"] == "none" or estimator is None:
        return calibration, calibrator_path

    if isinstance(estimator, IsotonicCalibratedModel):
        calibrator_path = export_root / "calibrator_model.joblib"
        joblib.dump(estimator.iso_model, calibrator_path)
        calibration.update(
            {"method": "isotonic", "artifact_key": "calibrator_model", "score_source": "logit"}
        )
    elif isinstance(estimator, CalibratedClassifierCV):
        calibrated = estimator.calibrated_classifiers_[0]
        cal = getattr(calibrated, "calibrator", None)
        if cal is None:
            calibrators = getattr(calibrated, "calibrators", None)
            if calibrators:
                cal = calibrators[0]
        if cal is None:
            calibration["method"] = "none"
        elif cal.__class__.__name__ == "LogisticRegression":
            coef = float(cal.coef_.ravel()[0])
            intercept = float(cal.intercept_.ravel()[0])
            calibration.update(
                {
                    "method": "platt",
                    "coef": coef,
                    "intercept": intercept,
                    "score_source": "raw",
                }
            )
        else:
            calibrator_path = export_root / "calibrator_model.joblib"
            joblib.dump(cal, calibrator_path)
            calibration.update(
                {"method": "isotonic", "artifact_key": "calibrator_model", "score_source": "raw"}
            )
    else:
        calibration["method"] = "none"

    return calibration, calibrator_path


def _build_signature() -> ModelSignature:
    if Schema is None or ColSpec is None:
        raise MLFlowExportError("MLflow schema utilities unavailable.")
    inputs = Schema([ColSpec(dtype, name) for name, dtype in API_INPUT_COLUMNS])
    outputs = Schema(
        [
            ColSpec("double", "probability_fraud"),
            ColSpec("integer", "binary_label"),
            ColSpec("string", "decision"),
            ColSpec("double", "threshold"),
        ]
    )
    return ModelSignature(inputs=inputs, outputs=outputs)


def _empty_input_example(config: Dict[str, Any]) -> pd.DataFrame:
    fill_value = config.get("preprocessing", {}).get("fill_missing", "")
    record: Dict[str, Any] = {
        name: fill_value if dtype == "string" else DEFAULT_NUMERIC_VALUES.get(name, 0)
        for name, dtype in API_INPUT_COLUMNS
    }
    for col, default in DEFAULT_NUMERIC_VALUES.items():
        record[col] = default
    return pd.DataFrame([record])


def _collect_metrics(best_model: "BestModelArtifacts") -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for split, metric in ("val", best_model.val_metrics.values), (
        "test",
        best_model.test_metrics.values,
    ):
        for key, value in metric.items():
            if value is None:
                continue
            metrics[f"{split}_{key}"] = float(value)
    return metrics


def _collect_params(
    best_model: "BestModelArtifacts",
    config: Dict[str, Any],
    opset: int,
    quantized: bool,
    prefer_quantized: bool,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "model_name": best_model.name,
        "model_type": best_model.model_type,
        "feature_type": best_model.feature_type,
        "calibration_method": best_model.calibration_method or "none",
        "threshold": float(best_model.threshold),
        "onnx_opset": int(opset),
        "quantized": str(quantized),
        "prefer_quantized": str(prefer_quantized),
        "config_hash": config_hash(config),
    }
    gray = config.get("gray_zone", {})
    params.update({f"gray_zone_{k}": v for k, v in gray.items()})
    return params


def _collect_tags(best_model: "BestModelArtifacts") -> Dict[str, str]:
    return {
        "pipeline": best_model.model_type,
        "calibration": best_model.calibration_method or "none",
        "feature_type": best_model.feature_type,
    }


def _prepare_classical_export(
    best_model: "BestModelArtifacts",
    bundle: FeatureBundle,
    config: Dict[str, Any],
    export_root: Path,
    mlflow_conf: Dict[str, Any],
    splits: "SplitResult",
) -> ExportedModelArtifacts:
    _ = splits
    if convert_sklearn is None or FloatTensorType is None:
        raise MLFlowExportError("skl2onnx dependency is required for classical export.")

    export_conf = mlflow_conf.get("export", {})
    opset = int(export_conf.get("opset", 17))
    quantize_flag = bool(export_conf.get("dynamic_quantization", True))
    prefer_quantized = bool(export_conf.get("prefer_quantized", True))

    base_estimator = best_model.base_estimator or best_model.estimator
    if base_estimator is None:
        raise MLFlowExportError("Classical export requires a fitted base estimator.")

    if best_model.feature_type == "tfidf+tabular":
        feature_dim = bundle.tfidf_train.shape[1] + bundle.tabular_train.shape[1]
    elif best_model.feature_type == "tfidf":
        feature_dim = bundle.tfidf_train.shape[1]
    else:
        feature_dim = bundle.tabular_train.shape[1]

    initial_types = [("input", FloatTensorType([None, feature_dim]))]
    options = {id(base_estimator): {"raw_scores": True, "output_class_labels": False}}

    if isinstance(base_estimator, (LogisticRegression, LinearSVC)):
        onnx_model = convert_sklearn(
            base_estimator,
            initial_types=initial_types,
            target_opset=opset,
            options=options,
        )
    else:  # pragma: no cover - uncommon branch
        raise MLFlowExportError(
            f"Unsupported classical estimator for ONNX export: {type(base_estimator).__name__}"
        )

    onnx_path = export_root / "classical.onnx"
    with open(onnx_path, "wb") as handle:
        handle.write(onnx_model.SerializeToString())

    quantized_path: Optional[Path] = None
    if quantize_flag and quantize_dynamic is not None and QuantType is not None:
        quantized_path = export_root / "classical.int8.onnx"
        try:  # pragma: no branch
            quantize_dynamic(str(onnx_path), str(quantized_path), weight_type=QuantType.QInt8)
        except Exception as exc:  # pragma: no cover - quantization optional
            logger.warning("Dynamic quantization failed: %s", exc)
            quantized_path.unlink(missing_ok=True)
            quantized_path = None

    vectorizer_path = export_root / "tfidf_vectorizer.joblib"
    joblib.dump(bundle.tfidf_vectorizer, vectorizer_path)
    scaler_path = export_root / "tabular_scaler.joblib"
    joblib.dump(bundle.tabular_scaler, scaler_path)
    feature_names_path = export_root / "tabular_feature_names.joblib"
    joblib.dump(bundle.feature_names, feature_names_path)

    metadata_src = ARTIFACTS_DIR / "metadata.json"
    metadata_path = export_root / "metadata.json"
    if metadata_src.exists():
        shutil.copy2(metadata_src, metadata_path)
    else:  # pragma: no cover - defensive fallback
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump({}, handle)

    config_src = ARTIFACTS_DIR / "config_used.yaml"
    config_path = export_root / "config.yaml"
    if config_src.exists():
        shutil.copy2(config_src, config_path)
    else:
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle)

    calibration_info, calibrator_path = _serialize_calibration(best_model, export_root)
    calibration_json = export_root / "calibration.json"
    with calibration_json.open("w", encoding="utf-8") as handle:
        json.dump(calibration_info, handle)

    artifacts = {
        "onnx_model": onnx_path,
        "vectorizer": vectorizer_path,
        "scaler": scaler_path,
        "feature_names": feature_names_path,
        "metadata": metadata_path,
        "config": config_path,
        "calibration": calibration_json,
    }
    if quantized_path:
        artifacts["onnx_quantized"] = quantized_path
    if calibrator_path:
        artifacts["calibrator_model"] = calibrator_path

    signature = None
    metrics = _collect_metrics(best_model)
    params = _collect_params(
        best_model, config, opset, quantized_path is not None, prefer_quantized
    )
    tags = _collect_tags(best_model)

    python_model = SpotScamPyfuncModel("classical", prefer_quantized=prefer_quantized)

    return ExportedModelArtifacts(
        python_model=python_model,
        artifacts={key: str(path) for key, path in artifacts.items()},
        signature=signature,
        input_example=None,
        params=params,
        metrics=metrics,
        tags=tags,
    )


def _prepare_transformer_export(
    best_model: "BestModelArtifacts",
    config: Dict[str, Any],
    export_root: Path,
    mlflow_conf: Dict[str, Any],
    splits: "SplitResult",
) -> ExportedModelArtifacts:
    _ = splits
    if ORTModelForSequenceClassification is None or AutoTokenizer is None:
        raise MLFlowExportError(
            "optimum[onnxruntime] and transformers are required for transformer export."
        )

    export_conf = mlflow_conf.get("export", {})
    opset = int(export_conf.get("opset", 17))
    quantize_flag = bool(export_conf.get("dynamic_quantization", True))
    prefer_quantized = bool(export_conf.get("prefer_quantized", True))

    model_dir = Path(best_model.extra.get("model_dir", ARTIFACTS_DIR / "transformer" / "best"))
    tokenizer_dir = Path(
        best_model.extra.get("tokenizer_dir", ARTIFACTS_DIR / "transformer" / "tokenizer")
    )
    if not model_dir.exists():
        raise MLFlowExportError(f"Expected transformer directory does not exist: {model_dir}")

    onnx_export_dir = export_root / "onnx_transformer"
    onnx_export_dir.mkdir(parents=True, exist_ok=True)
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_dir,
        from_transformers=True,
        export=True,
        opset=opset,
    )
    ort_model.save_pretrained(onnx_export_dir)
    raw_onnx = onnx_export_dir / "model.onnx"
    if not raw_onnx.exists():  # pragma: no cover - defensive
        raise MLFlowExportError("Exported ONNX file missing from optimum output.")

    onnx_path = export_root / "transformer.onnx"
    shutil.copy2(raw_onnx, onnx_path)

    quantized_path: Optional[Path] = None
    if quantize_flag and quantize_dynamic is not None and QuantType is not None:
        quantized_path = export_root / "transformer.int8.onnx"
        try:  # pragma: no branch
            quantize_dynamic(str(onnx_path), str(quantized_path), weight_type=QuantType.QInt8)
        except Exception as exc:  # pragma: no cover - optional quantization
            logger.warning("Dynamic quantization for transformer failed: %s", exc)
            quantized_path.unlink(missing_ok=True)
            quantized_path = None

    tokenizer_copy = export_root / "tokenizer"
    if tokenizer_copy.exists():
        shutil.rmtree(tokenizer_copy)
    shutil.copytree(tokenizer_dir, tokenizer_copy)

    metadata_src = ARTIFACTS_DIR / "metadata.json"
    metadata_path = export_root / "metadata.json"
    if metadata_src.exists():
        shutil.copy2(metadata_src, metadata_path)
    else:
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump({}, handle)

    config_src = ARTIFACTS_DIR / "config_used.yaml"
    config_path = export_root / "config.yaml"
    if config_src.exists():
        shutil.copy2(config_src, config_path)
    else:
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle)

    calibration_info = {"method": "none"}
    calibration_json = export_root / "calibration.json"
    with calibration_json.open("w", encoding="utf-8") as handle:
        json.dump(calibration_info, handle)

    artifacts = {
        "onnx_model": onnx_path,
        "tokenizer": tokenizer_copy,
        "metadata": metadata_path,
        "config": config_path,
        "calibration": calibration_json,
    }
    if quantized_path:
        artifacts["onnx_quantized"] = quantized_path

    signature = None
    metrics = _collect_metrics(best_model)
    params = _collect_params(
        best_model, config, opset, quantized_path is not None, prefer_quantized
    )
    tags = _collect_tags(best_model)

    python_model = SpotScamPyfuncModel("transformer", prefer_quantized=prefer_quantized)

    return ExportedModelArtifacts(
        python_model=python_model,
        artifacts={key: str(path) for key, path in artifacts.items()},
        signature=signature,
        input_example=None,
        params=params,
        metrics=metrics,
        tags=tags,
    )
