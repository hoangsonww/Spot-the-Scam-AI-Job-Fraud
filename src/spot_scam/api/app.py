from __future__ import annotations

import os
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from spot_scam.api.schemas import (
    GrayZonePolicy,
    HealthResponse,
    JobPostingInput,
    MetadataResponse,
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
)
from spot_scam.inference.predictor import FraudPredictor
from spot_scam.utils.logging import configure_logging

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
    predictions = predictor.predict(payload)
    return PredictionBatchResponse(predictions=[PredictionResponse(**pred) for pred in predictions])


@app.post("/predict/single", response_model=PredictionResponse)
def predict_single(
    item: JobPostingInput,
    predictor: FraudPredictor = Depends(get_predictor),
) -> PredictionResponse:
    predictions = predictor.predict([item.model_dump()])
    if not predictions:
        raise HTTPException(status_code=400, detail="No predictions generated.")
    return PredictionResponse(**predictions[0])


__all__ = ["app"]
