from __future__ import annotations

import os
import csv
from functools import lru_cache
from typing import Dict, List, Tuple
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware

from spot_scam.api.schemas import (
    GrayZonePolicy,
    HealthResponse,
    JobPostingInput,
    MetadataResponse,
    ModelSummary,
    ModelsResponse,
    MetricSet,
    PredictionBatchRequest,
    PredictionBatchResponse,
    PredictionResponse,
    TokenFrequency,
    TokenFrequencyResponse,
    TokenImportanceResponse,
    TokenWeight,
    ThresholdMetricsResponse,
    LatencySummaryResponse,
    SliceMetricsResponse,
    CasesResponse,
    ReviewCase,
    ReviewCasePayload,
    PredictionExplanation,
    FeedbackIn,
)
from spot_scam.inference.predictor import FraudPredictor
from spot_scam.utils.logging import configure_logging
from spot_scam.tracking.predictions import (
    log_predictions,
    get_review_queue,
    load_predictions_dataframe,
)
from spot_scam.tracking.feedback import append_feedback
from spot_scam.utils.paths import TRACKING_DIR, ensure_directories

logger = configure_logging(__name__)
app = FastAPI(title="Spot the Scam API", version="1.0.0")

default_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
allowed_origins_env = os.getenv("SPOT_SCAM_ALLOWED_ORIGINS")
if allowed_origins_env:
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    allowed_origins = default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_predictor() -> FraudPredictor:
    logger.info("Loading fraud predictor artifacts.")
    return FraudPredictor()


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    predictor = get_predictor()
    return HealthResponse(status="ok", model_type=predictor.model_type, threshold=predictor.threshold)


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    predictor = get_predictor()
    meta = predictor.get_model_metadata()
    gray = GrayZonePolicy(**meta["gray_zone"])
    return MetadataResponse(
        model_name=meta["model_name"],
        model_type=meta["model_type"],
        feature_type=meta["feature_type"],
        calibration_method=meta.get("calibration_method"),
        threshold=meta["threshold"],
        gray_zone=gray,
        val_metrics=MetricSet(**meta.get("val_metrics", {})),
        test_metrics=MetricSet(**meta.get("test_metrics", {})),
        test_ece=meta.get("test_ece"),
    )


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@app.get("/models", response_model=ModelsResponse)
def list_models(limit: int = Query(20, ge=1, le=200)) -> ModelsResponse:
    """
    Return the most recent tracked runs for each trained model configuration.
    """
    path = TRACKING_DIR / "runs.csv"
    if not path.exists():
        return ModelsResponse(items=[])

    records: Dict[Tuple[str, str], ModelSummary] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model_name = (row.get("model_name") or "").strip()
            if not model_name:
                continue
            config_hash = (row.get("config_hash") or "").strip()
            key = (model_name, config_hash)

            timestamp_raw = row.get("timestamp")
            try:
                timestamp = datetime.fromisoformat(timestamp_raw) if timestamp_raw else None
            except ValueError:
                timestamp = None

            summary = ModelSummary(
                model_name=model_name,
                model_type=(row.get("model_type") or "").strip(),
                calibration_method=(row.get("calibration_method") or "").strip() or None,
                threshold=_parse_float(row.get("threshold")),
                timestamp=timestamp,
                validation=MetricSet(
                    f1=_parse_float(row.get("val_f1")),
                    precision=_parse_float(row.get("val_precision")),
                    recall=_parse_float(row.get("val_recall")),
                ),
                test=MetricSet(
                    f1=_parse_float(row.get("test_f1")),
                    precision=_parse_float(row.get("test_precision")),
                    recall=_parse_float(row.get("test_recall")),
                ),
            )

            existing = records.get(key)
            if existing and existing.timestamp and summary.timestamp:
                if summary.timestamp <= existing.timestamp:
                    continue
            elif existing and not summary.timestamp:
                continue

            records[key] = summary

    items = list(records.values())
    items.sort(key=lambda item: item.test.f1 or 0.0, reverse=True)
    if limit and len(items) > limit:
        items = items[:limit]
    return ModelsResponse(items=items)


@app.get("/insights/token-importance", response_model=TokenImportanceResponse)
def token_importance(limit: int = Query(20, ge=1, le=100)) -> TokenImportanceResponse:
    predictor = get_predictor()
    tokens = predictor.get_token_importance(limit=limit)
    return TokenImportanceResponse(
        positive=[TokenWeight(**item) for item in tokens.get("positive", [])],
        negative=[TokenWeight(**item) for item in tokens.get("negative", [])],
    )


@app.get("/insights/token-frequency", response_model=TokenFrequencyResponse)
def token_frequency(limit: int = Query(20, ge=1, le=200)) -> TokenFrequencyResponse:
    predictor = get_predictor()
    freq = predictor.get_token_frequency(limit=limit)
    return TokenFrequencyResponse(items=[TokenFrequency(**item) for item in freq])


@app.get("/insights/threshold-metrics", response_model=ThresholdMetricsResponse)
def threshold_metrics(limit: int = Query(50, ge=5, le=200)) -> ThresholdMetricsResponse:
    predictor = get_predictor()
    metrics = predictor.get_threshold_metrics(limit=limit)
    return ThresholdMetricsResponse(points=metrics)


@app.get("/insights/latency", response_model=LatencySummaryResponse)
def latency_summary() -> LatencySummaryResponse:
    predictor = get_predictor()
    summary = predictor.get_latency_summary()
    return LatencySummaryResponse(items=summary)


@app.post("/feedback", status_code=201)
def post_feedback(
    items: List[FeedbackIn] = Body(...),
) -> Dict[str, int]:
    ensure_directories()
    if not items:
        raise HTTPException(status_code=400, detail="Empty feedback payload.")
    if len(items) > 100:
        raise HTTPException(status_code=400, detail="Feedback batch exceeds limit of 100.")

    predictions_df = load_predictions_dataframe()
    known_ids = set(predictions_df["request_id"]) if not predictions_df.empty else set()

    rows = []
    for entry in items:
        data = entry.model_dump()
        if known_ids and data["request_id"] not in known_ids:
            raise HTTPException(status_code=404, detail=f"Unknown request_id {data['request_id']}.")
        data["created_at"] = datetime.utcnow().isoformat()
        rows.append(data)

    append_feedback(rows)
    return {"inserted": len(rows)}


@app.get("/cases", response_model=CasesResponse)
def review_cases(
    policy: str = Query("gray-zone", description="Sampling policy (gray-zone, entropy)."),
    limit: int = Query(25, ge=1, le=200),
    predictor: FraudPredictor = Depends(get_predictor),
) -> CasesResponse:
    band = predictor.get_gray_zone_band()
    queue = get_review_queue(
        policy=policy,
        limit=limit,
        threshold=float(predictor.threshold),
        gray_zone_width=float(band["width"]),
    )

    items = []
    for raw in queue["items"]:
        explanation = raw.get("explanation") or {}
        payload = raw.get("payload") or {}
        try:
            created_at = datetime.fromisoformat(raw["created_at"])
        except Exception:
            created_at = datetime.utcnow()
        items.append(
            ReviewCase(
                request_id=str(raw["request_id"]),
                created_at=created_at,
                probability=float(raw["probability"]),
                predicted_label=str(raw["predicted_label"]),
                model_version=str(raw["model_version"]),
                threshold=float(raw.get("threshold") or 0.0),
                text_hash=str(raw["text_hash"]),
                features_hash=str(raw["features_hash"]),
                payload=ReviewCasePayload.model_validate(payload),
                explanation=PredictionExplanation.model_validate(explanation),
            )
        )

    return CasesResponse(total_pending=int(queue["total_pending"]), items=items)

@app.get("/insights/slice-metrics", response_model=SliceMetricsResponse)
def slice_metrics(limit: int = Query(6, ge=1, le=50)) -> SliceMetricsResponse:
    predictor = get_predictor()
    metrics = predictor.get_slice_metrics(limit=limit)
    return SliceMetricsResponse(items=metrics)


@app.post("/predict", response_model=PredictionBatchResponse)
def predict(
    request: PredictionBatchRequest,
    predictor: FraudPredictor = Depends(get_predictor),
) -> PredictionBatchResponse:
    payload = [item.model_dump() for item in request.instances]
    predictions, contexts = predictor.predict(payload, return_context=True)
    model_name = predictor.metadata.get("model_name", predictor.metadata.get("model_type", "unknown"))

    logged_records = log_predictions(
        payloads=payload,
        processed_text=[ctx["text_all"] for ctx in contexts],
        tabular_features=[ctx["tabular_features"] for ctx in contexts],
        predictions=predictions,
        model_name=model_name,
    )

    enriched = []
    for pred, record in zip(predictions, logged_records):
        enriched.append(PredictionResponse(**{**pred, "request_id": record["request_id"]}))

    return PredictionBatchResponse(predictions=enriched)


@app.post("/predict/single", response_model=PredictionResponse)
def predict_single(
    item: JobPostingInput,
    predictor: FraudPredictor = Depends(get_predictor),
) -> PredictionResponse:
    predictions, contexts = predictor.predict([item.model_dump()], return_context=True)
    if not predictions:
        raise HTTPException(status_code=400, detail="No predictions generated.")
    model_name = predictor.metadata.get("model_name", predictor.metadata.get("model_type", "unknown"))
    logged_records = log_predictions(
        payloads=[item.model_dump()],
        processed_text=[contexts[0]["text_all"]],
        tabular_features=[contexts[0]["tabular_features"]],
        predictions=predictions,
        model_name=model_name,
    )
    enriched = {**predictions[0], "request_id": logged_records[0]["request_id"]}
    return PredictionResponse(**enriched)


__all__ = ["app"]
