# Explainability in Spot the Scam

This document explains how explanations are produced, where they come from in the code, and which artifacts support explainability and insights in the API and frontend.

## Table of Contents

- [Design Goals](#design-goals)
- [Classical Model Explanations](#classical-model-explanations)
- [Transformer Model Explanations](#transformer-model-explanations)
- [Explainability Artifacts and Insights Tables](#explainability-artifacts-and-insights-tables)
- [How Explanations Flow Through the API](#how-explanations-flow-through-the-api)
- [Frontend Rendering](#frontend-rendering)
- [Validation and Practical Checks](#validation-and-practical-checks)
- [Related Documentation](#related-documentation)

## Design Goals

Explanations in this project are designed to be:

- Directional: signals toward fraud vs signals toward legit.
- Lightweight: fast enough to run on every prediction.
- Consistent: the API and frontend share a common explanation schema.

## Classical Model Explanations

Classical explanations are built in `FraudPredictor._build_classical_explanations` in `src/spot_scam/inference/predictor.py`.

Core idea:

- Multiply activated feature values by linear coefficients to obtain per-feature contributions.
- Rank positive contributions as fraud signals and negative contributions as legit signals.
- Include the model intercept when available.

Classical explanations require access to a linear base model (`coef_`). When coefficients are not available, the system returns a safe fallback summary.

## Transformer Model Explanations

Transformer explanations are built in `FraudPredictor._build_transformer_explanations`.

Current approach:

- Primary path: gradient × input token attribution on the positive-class logit.
- Fallback path: centered CLS-attention scores when gradients are unavailable (for example, in quantized mode).

The output mirrors the classical explanation schema so the frontend can render both uniformly.

## Explainability Artifacts and Insights Tables

The training pipeline generates explainability-related artifacts under `experiments/tables/`.

Key tables:

- Token coefficients:
  - `experiments/tables/top_terms_positive.csv`
  - `experiments/tables/top_terms_negative.csv`
- Token frequency deltas:
  - `experiments/tables/token_frequency_analysis.csv`
- Slice performance summaries:
  - `experiments/tables/slice_metrics.csv`
- Threshold sweep points:
  - `experiments/tables/threshold_metrics.csv`

These power the insights endpoints and dashboard panels.

## How Explanations Flow Through the API

The explanation lifecycle is:

1. Training creates artifacts and insight tables.
2. `FraudPredictor.predict(...)` builds explanations per request.
3. FastAPI returns the explanation under the `explanation` field.
4. The frontend renders the explanation directly.

Example shape:

```json
{
  "probability_fraud": 0.72,
  "decision": "fraud",
  "explanation": {
    "top_positive": [
      { "feature": "wire transfer", "source": "token", "contribution": 0.42 }
    ],
    "top_negative": [
      { "feature": "benefits", "source": "token", "contribution": -0.18 }
    ],
    "intercept": -0.12,
    "summary": "Wire transfer pushed the score toward fraud."
  }
}
```

## Frontend Rendering

The frontend consumes the explanation schema and renders:

- A summary sentence.
- Signals toward fraud.
- Signals toward legit.
- An intercept line when provided.

Frontend types and request helpers live in `frontend/src/lib/api.ts`.

## Validation and Practical Checks

There are limited unit tests specifically for explanations. Use these practical checks:

- Run training and confirm insight tables are created in `experiments/tables/`.
- Start the API and score known examples.
- Inspect `explanation.top_positive` and `explanation.top_negative` in responses.

Quick check:

```bash
PYTHONPATH=src uvicorn spot_scam.api.app:app --reload
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{"title":"Remote Data Entry","description":"...wire transfer..."}'
```

## Related Documentation

- System design: [ARCHITECTURE.md](../ARCHITECTURE.md)
- Training strategy: [TRAINING_ANALYSIS.md](../TRAINING_ANALYSIS.md)
- Metrics and diagnostics: [RESULTS.md](../RESULTS.md)
- Pipeline narrative: [docs/pipeline_walkthrough.md](pipeline_walkthrough.md)
- Setup and operations: [INSTRUCTIONS.md](../INSTRUCTIONS.md)
