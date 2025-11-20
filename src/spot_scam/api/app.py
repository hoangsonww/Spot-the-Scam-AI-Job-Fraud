from __future__ import annotations

import json
import os
import csv
from functools import lru_cache
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Load env vars from .env file
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

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
    ChatRequest,
    ChatStreamChunk,
)
from spot_scam.inference.predictor import FraudPredictor
from spot_scam.utils.logging import configure_logging
from spot_scam.tracking.predictions import (
    log_predictions,
    get_review_queue,
    load_predictions_dataframe,
    load_active_sample_dataframe,
)
from spot_scam.tracking.feedback import append_feedback
from spot_scam.utils.paths import TRACKING_DIR, ensure_directories

logger = configure_logging(__name__)
app = FastAPI(title="Spot the Scam API", version="1.0.0")

default_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
allowed_origins_env = os.getenv("SPOT_SCAM_ALLOWED_ORIGINS")
if allowed_origins_env:
    allowed_origins = [
        origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()
    ]
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
    return HealthResponse(
        status="ok", model_type=predictor.model_type, threshold=predictor.threshold
    )


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

    active_sample_df = load_active_sample_dataframe()
    if not active_sample_df.empty and "request_id" in active_sample_df.columns:
        known_ids.update(str(rid) for rid in active_sample_df["request_id"].dropna().astype(str))

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
    offset: int = Query(0, ge=0, le=5000),
    predictor: FraudPredictor = Depends(get_predictor),
) -> CasesResponse:
    band = predictor.get_gray_zone_band()
    queue = get_review_queue(
        policy=policy,
        limit=limit,
        threshold=float(predictor.threshold),
        gray_zone_width=float(band["width"]),
        offset=offset,
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
    model_name = predictor.metadata.get(
        "model_name", predictor.metadata.get("model_type", "unknown")
    )

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
    model_name = predictor.metadata.get(
        "model_name", predictor.metadata.get("model_type", "unknown")
    )
    logged_records = log_predictions(
        payloads=[item.model_dump()],
        processed_text=[contexts[0]["text_all"]],
        tabular_features=[contexts[0]["tabular_features"]],
        predictions=predictions,
        model_name=model_name,
    )
    enriched = {**predictions[0], "request_id": logged_records[0]["request_id"]}
    return PredictionResponse(**enriched)


@app.post("/chat")
async def chat_stream(request: ChatRequest):
    """
    Stream chat responses from Google Gemini API with optional fraud detection context.
    Automatically detects job postings and runs them through the fraud detection model.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Google Generative AI package not installed. Run: pip install google-generativeai",
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set.")

    try:
        genai.configure(api_key=api_key)

        system_parts = [
            "You are an AI assistant specialized in job fraud detection and analysis.",
            "When analyzing job postings, focus on specific fraud indicators and red flags.",
            "Provide actionable advice and clear explanations.",
            "Use markdown formatting (lists, tables, bold, etc.) to make your responses easy to read.",
            "Be concise but thorough - prioritize the most important information.",
        ]
        assistant_instruction = "\n".join(system_parts)

        assistant_model = genai.GenerativeModel(
            "gemini-2.0-flash-lite",
            system_instruction=assistant_instruction,
        )

        classifier_model = genai.GenerativeModel(
            "gemini-2.0-flash-lite",
            system_instruction=(
                "You determine whether a message contains information about a job posting. "
                "Respond ONLY with JSON using this schema: "
                '{"is_job_posting": bool, "confidence": number (0-1), "reason": "string"}'
            ),
        )

        predictor = get_predictor()

        def _extract_response_text(response_obj) -> str:
            if hasattr(response_obj, "text") and response_obj.text:
                return response_obj.text
            if hasattr(response_obj, "candidates"):
                for candidate in getattr(response_obj, "candidates", []):
                    content = getattr(candidate, "content", None)
                    if content and getattr(content, "parts", None):
                        parts_text = "".join(
                            getattr(part, "text", "")
                            for part in content.parts
                            if hasattr(part, "text")
                        )
                        if parts_text:
                            return parts_text
            return ""

        def _parse_bool(value) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"true", "1", "yes", "y"}
            return False

        def _parse_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _parse_classifier_json(raw: str) -> Dict[str, object] | None:
            if not raw:
                return None
            candidate = raw.strip()
            if "```" in candidate:
                parts = candidate.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("{") and part.endswith("}"):
                        candidate = part
                        break
            if not candidate.startswith("{"):
                start = candidate.find("{")
                end = candidate.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidate = candidate[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                logger.warning("Failed to parse classifier JSON: %s", candidate)
                return None

        def classify_message_with_llm(message: str) -> Dict[str, object] | None:
            prompt = (
                "Classify the following user message. " "Return JSON only.\n" f"Message:\n{message}"
            )
            try:
                response = classifier_model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.1, "max_output_tokens": 200},
                )
            except Exception as classification_error:
                logger.warning("LLM classification failed: %s", classification_error)
                return None
            raw_text = _extract_response_text(response)
            return _parse_classifier_json(raw_text)

        # Determine whether to run the fraud predictor
        llm_detection = classify_message_with_llm(request.message)
        llm_confidence = (
            _parse_float(llm_detection["confidence"])
            if llm_detection and "confidence" in llm_detection
            else None
        )
        llm_is_job = (
            _parse_bool(llm_detection["is_job_posting"])
            if llm_detection and "is_job_posting" in llm_detection
            else False
        )
        message_lower = request.message.lower()
        job_keywords = [
            "job description",
            "we're hiring",
            "we are hiring",
            "position",
            "role",
            "responsibilities",
            "requirements",
            "experience",
            "qualifications",
            "about the job",
            "about the role",
            "tech stack",
            "looking for",
            "years of experience",
            "full-time",
            "part-time",
            "remote",
            "salary",
        ]
        heuristic_job_posting = (
            any(keyword in message_lower for keyword in job_keywords) and len(request.message) > 200
        )

        should_run_predictor = False
        job_detection_source = None
        if not request.context and llm_detection:
            if llm_is_job:
                should_run_predictor = True
                job_detection_source = "llm"
        if not request.context and not llm_detection and heuristic_job_posting:
            should_run_predictor = True
            job_detection_source = "heuristic"

        auto_prediction = None
        if should_run_predictor:
            try:
                logger.info(
                    "LLM detected potential job posting (%s). Running fraud detection...",
                    job_detection_source,
                )
                job_input = JobPostingInput(
                    title="Auto-detected job posting",
                    description=request.message,
                    company_profile=None,
                    requirements=None,
                    benefits=None,
                    location=None,
                    employment_type=None,
                    required_experience=None,
                    required_education=None,
                    industry=None,
                    function=None,
                    telecommuting=0,
                    has_company_logo=0,
                    has_questions=0,
                )

                predictions, _contexts = predictor.predict(
                    [job_input.model_dump()], return_context=True
                )
                if predictions:
                    pred = predictions[0]
                    auto_prediction = {
                        "probability_fraud": pred["probability_fraud"],
                        "binary_label": pred["binary_label"],
                        "decision": pred["decision"],
                        "explanation": pred["explanation"],
                    }
                    logger.info(
                        "Auto-detection result: %s (%.2f%%)",
                        pred["decision"],
                        pred["probability_fraud"] * 100,
                    )
            except Exception as e:
                logger.warning(f"Failed to auto-detect job posting: {str(e)}")
                auto_prediction = None

        chat_history = []
        if request.history:
            # keep last 10 messages for bot context
            for msg in request.history[-10:]:
                role = "user" if msg.role == "user" else "model"
                chat_history.append({"role": role, "parts": [msg.content]})

        context_info: List[str] = []

        include_classification_context = llm_is_job or heuristic_job_posting

        if include_classification_context and llm_detection:
            classification_header = "=== MESSAGE CLASSIFICATION ==="
            if classification_header not in context_info:
                context_info.append(classification_header)
            assessment = "Job Posting" if llm_is_job else "General Inquiry"
            if llm_confidence is not None:
                context_info.append(
                    f"LLM Assessment: {assessment} ({llm_confidence * 100:.1f}% confidence)"
                )
            else:
                context_info.append(f"LLM Assessment: {assessment}")
            reason = llm_detection.get("reason")
            if isinstance(reason, str) and reason.strip():
                context_info.append(f"Rationale: {reason.strip()}")
        elif include_classification_context and heuristic_job_posting:
            context_info.append("=== MESSAGE CLASSIFICATION ===")
            context_info.append("Heuristic Assessment: Potential job posting based on keywords.")

        if auto_prediction:
            fraud_pct = auto_prediction["probability_fraud"] * 100
            context_info.append("=== FRAUD DETECTION ANALYSIS ===")
            context_info.append(f"Fraud Probability: {fraud_pct:.1f}%")
            context_info.append(f"Decision: {auto_prediction['decision']}")
            context_info.append(
                f"Classification: {'FRAUDULENT' if auto_prediction['binary_label'] == 1 else 'LEGITIMATE'}"
            )

            if auto_prediction.get("explanation"):
                exp = auto_prediction["explanation"]
                if exp.get("top_positive"):
                    context_info.append("\nTop Fraud Indicators:")
                    for contrib in exp["top_positive"][:5]:
                        context_info.append(
                            f"  - {contrib['feature']}: {contrib['contribution']:.3f}"
                        )

                if exp.get("top_negative"):
                    context_info.append("\nTop Legitimacy Signals:")
                    for contrib in exp["top_negative"][:5]:
                        context_info.append(
                            f"  - {contrib['feature']}: {contrib['contribution']:.3f}"
                        )

            context_info.append(
                "\nBased on this analysis, provide a detailed but concise explanation of:"
            )
            context_info.append("1. Whether this job posting appears fraudulent and why")
            context_info.append("2. The main red flags or positive signals")
            context_info.append("3. Actionable advice for the job seeker")

        elif request.context and request.context.prediction:
            pred = request.context.prediction
            fraud_pct = pred.probability_fraud * 100
            context_info.append(f"Fraud Detection Result:")
            context_info.append(f"- Fraud Probability: {fraud_pct:.1f}%")
            context_info.append(f"- Decision: {pred.decision}")
            context_info.append(
                f"- Binary Label: {'Fraudulent' if pred.binary_label == 1 else 'Legitimate'}"
            )

            if pred.explanation:
                if pred.explanation.top_positive:
                    context_info.append("\nTop Fraud Indicators:")
                    for contrib in pred.explanation.top_positive[:3]:
                        context_info.append(f"  - {contrib.feature}: {contrib.contribution:.3f}")

                if pred.explanation.top_negative:
                    context_info.append("\nTop Legitimacy Indicators:")
                    for contrib in pred.explanation.top_negative[:3]:
                        context_info.append(f"  - {contrib.feature}: {contrib.contribution:.3f}")

        if request.context and request.context.job_posting:
            job = request.context.job_posting
            context_info.append("\nJob Posting Details:")
            if job.title:
                context_info.append(f"- Title: {job.title}")
            if job.company_profile:
                context_info.append(f"- Company: {job.company_profile[:200]}...")
            if job.description:
                context_info.append(f"- Description: {job.description[:300]}...")
            if job.location:
                context_info.append(f"- Location: {job.location}")
            if job.employment_type:
                context_info.append(f"- Employment Type: {job.employment_type}")

        context_block = "\n".join(context_info).strip()
        if context_block:
            combined_user_message = (
                "Context from the fraud detection pipeline:\n"
                f"{context_block}\n\n"
                "Original user message:\n"
                f"{request.message}"
            )
        else:
            combined_user_message = request.message

        def generate():
            try:
                logger.info(
                    "Generating response for message: %s...",
                    request.message[:50].replace("\n", " "),
                )

                if chat_history:
                    chat = assistant_model.start_chat(history=chat_history)
                    response = chat.send_message(combined_user_message, stream=True)
                else:
                    response = assistant_model.generate_content(combined_user_message, stream=True)

                has_content = False
                for chunk in response:
                    if hasattr(chunk, "prompt_feedback"):
                        logger.warning(f"Prompt feedback: {chunk.prompt_feedback}")

                    text = None
                    try:
                        if hasattr(chunk, "text") and chunk.text:
                            text = chunk.text
                        elif hasattr(chunk, "parts") and chunk.parts:
                            text = "".join(
                                part.text for part in chunk.parts if hasattr(part, "text")
                            )
                    except Exception as chunk_error:
                        logger.warning(f"Error extracting text from chunk: {chunk_error}")
                        if hasattr(chunk, "candidates") and chunk.candidates:
                            for candidate in chunk.candidates:
                                if hasattr(candidate, "content") and hasattr(
                                    candidate.content, "parts"
                                ):
                                    text = "".join(
                                        p.text
                                        for p in candidate.content.parts
                                        if hasattr(p, "text")
                                    )
                                    break

                    if text:
                        has_content = True
                        chunk_json = ChatStreamChunk(chunk=text, done=False).model_dump_json()
                        yield f"data: {chunk_json}\n\n"

                if not has_content:
                    logger.warning("No content generated from Gemini")
                    error_msg = "I apologize, but I couldn't generate a response. This might be due to content safety filters or an API issue. Please try rephrasing your question."
                    chunk_json = ChatStreamChunk(chunk=error_msg, done=False).model_dump_json()
                    yield f"data: {chunk_json}\n\n"

                final_json = ChatStreamChunk(chunk="", done=True).model_dump_json()
                yield f"data: {final_json}\n\n"
                logger.info("Response generation completed")

            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}", exc_info=True)
                error_json = ChatStreamChunk(chunk=f"Error: {str(e)}", done=True).model_dump_json()
                yield f"data: {error_json}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


__all__ = ["app"]
