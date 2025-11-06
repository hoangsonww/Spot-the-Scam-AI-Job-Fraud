from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


class JobPostingInput(BaseModel):
    title: str = Field(..., description="Job title.")
    location: Optional[str] = Field(default=None, description="Job location.")
    department: Optional[str] = Field(default=None)
    salary_range: Optional[str] = Field(default=None)
    company_profile: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    requirements: Optional[str] = Field(default=None)
    benefits: Optional[str] = Field(default=None)
    telecommuting: Optional[int] = Field(default=0)
    has_company_logo: Optional[int] = Field(default=0)
    has_questions: Optional[int] = Field(default=0)
    employment_type: Optional[str] = Field(default=None)
    required_experience: Optional[str] = Field(default=None)
    required_education: Optional[str] = Field(default=None)
    industry: Optional[str] = Field(default=None)
    function: Optional[str] = Field(default=None)


class HealthResponse(BaseModel):
    status: str
    model_type: str
    threshold: float


class GrayZonePolicy(BaseModel):
    width: float
    lower: float
    upper: float
    positive_label: str
    negative_label: str
    review_label: str


class MetricSet(BaseModel):
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    brier: Optional[float] = None


class MetadataResponse(BaseModel):
    model_name: str
    model_type: str
    feature_type: str
    calibration_method: Optional[str] = None
    threshold: float
    gray_zone: GrayZonePolicy
    val_metrics: MetricSet
    test_metrics: MetricSet
    test_ece: Optional[float] = None


class TokenWeight(BaseModel):
    term: str
    weight: float


class TokenImportanceResponse(BaseModel):
    positive: List[TokenWeight]
    negative: List[TokenWeight]


class TokenFrequency(BaseModel):
    token: str
    positive_count: int
    negative_count: int
    difference: int


class TokenFrequencyResponse(BaseModel):
    items: List[TokenFrequency]


class FeatureContribution(BaseModel):
    feature: str
    source: str
    contribution: float


class PredictionExplanation(BaseModel):
    top_positive: List[FeatureContribution] = Field(default_factory=list)
    top_negative: List[FeatureContribution] = Field(default_factory=list)
    intercept: Optional[float] = None
    summary: Optional[str] = None


class PredictionResponse(BaseModel):
    request_id: str = Field(..., description="Identifier for this prediction request.")
    probability_fraud: float = Field(..., description="Calibrated probability of the posting being fraudulent.")
    binary_label: int = Field(..., description="Thresholded label (1 = fraud).")
    decision: str = Field(..., description="Final decision after applying the gray-zone policy.")
    threshold: float = Field(..., description="Threshold used to assign binary_label.")
    gray_zone: Dict[str, Any] = Field(..., description="Gray-zone policy summary.")
    meta: Dict[str, Optional[str]] = Field(..., description="Metadata about the serving model.")
    explanation: PredictionExplanation = Field(..., description="Local explanation for the prediction.")


class PredictionBatchRequest(BaseModel):
    instances: List[JobPostingInput]


class PredictionBatchResponse(BaseModel):
    predictions: List[PredictionResponse]


class ThresholdMetricPoint(BaseModel):
    threshold: float
    precision: float
    recall: float
    f1: float


class ThresholdMetricsResponse(BaseModel):
    points: List[ThresholdMetricPoint]


class LatencySummaryPoint(BaseModel):
    batch_size: int
    latency_p50_ms: float
    latency_p95_ms: float
    throughput_rps: float


class LatencySummaryResponse(BaseModel):
    items: List[LatencySummaryPoint]

class SliceMetric(BaseModel):
    slice: str
    category: str
    count: int
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None

class SliceMetricsResponse(BaseModel):
    items: List[SliceMetric]


class ReviewCasePayload(BaseModel):
    title: Optional[str] = None
    company_profile: Optional[str] = None
    description: Optional[str] = None
    requirements: Optional[str] = None
    benefits: Optional[str] = None
    location: Optional[str] = None
    employment_type: Optional[str] = None
    required_experience: Optional[str] = None
    required_education: Optional[str] = None
    industry: Optional[str] = None
    function: Optional[str] = None


class ReviewCase(BaseModel):
    request_id: str
    created_at: datetime
    probability: float
    predicted_label: str
    model_version: str
    threshold: Optional[float] = None
    text_hash: str
    features_hash: str
    payload: ReviewCasePayload
    explanation: PredictionExplanation


class CasesResponse(BaseModel):
    total_pending: int
    items: List[ReviewCase]


class FeedbackIn(BaseModel):
    request_id: str
    model_version: str
    proba: float
    predicted_label: Literal["fraud", "legit", "review"]
    reviewer_label: Literal["fraud", "legit", "unsure"]
    text_hash: str
    features_hash: str
    rationale: Optional[str] = None
    notes: Optional[str] = None


__all__ = [
    "JobPostingInput",
    "PredictionBatchRequest",
    "PredictionResponse",
    "PredictionBatchResponse",
    "HealthResponse",
    "MetadataResponse",
    "TokenImportanceResponse",
    "TokenFrequencyResponse",
    "ThresholdMetricsResponse",
    "LatencySummaryResponse",
    "PredictionExplanation",
    "FeatureContribution",
    "SliceMetricsResponse",
    "SliceMetric",
    "CasesResponse",
    "ReviewCase",
    "FeedbackIn",
]
