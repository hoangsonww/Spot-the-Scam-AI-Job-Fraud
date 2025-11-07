# Explainability in Spot the Scam

This brief dives into how we generate and surface explanations for each prediction.

## 1. Classical models (Logistic Regression / Linear SVM - Support Vector Machine)
- **Core idea:** multiply the standardized feature vector by the learned coefficients to get per-feature contributions.
- **Implementation:** `spot_scam.inference.predictor.FraudPredictor._build_classical_explanations`.
  - Extract TF-IDF activations for the posting.
  - Extract tabular feature values (text length, digit count, scam term counts, etc.).
  - Multiply by coefficient vector (includes intercept).
  - Sort contributions descending → *top_positive*; ascending → *top_negative*.
  - Summaries convert feature names to readable text:
    ```
    “Immediate start, data entry, and immediate pushed the score toward fraud; Quickbooks, experience, and preferred reinforced the legit decision.”
    ```
- **Output schema:** returned with every `/predict` response under `explanation`.

## 2. Transformer models (distilbert-base-uncased)
- **Current behavior:** compute per-token contributions with a gradient × input attribution on the classification head. We build `inputs_embeds` manually, backprop the positive-class logit, and sum `(grad * embed)` across the hidden dimension to capture directional influence (positive ⇒ fraud leaning, negative ⇒ legit leaning). The UI now receives `top_positive` and `top_negative` tokens exactly like the classical stack.
- **Fallback:** when gradients are unavailable (e.g., quantized INT8 mode) we fall back to centered CLS-attention scores so we can still highlight salient tokens, albeit without the signed magnitude guarantees of gradients.

## 3. Frontend display
- `frontend/src/components/home-page.tsx` renders:
  - Summary sentence.
  - Two columns titled “Signals toward fraud” and “Signals toward legit” (list view of contributors).
  - Intercept line (“Model intercept: -4.284”).
- Styling ensures capitalized beginnings and sentence punctuation.

## 4. Additional explainability artifacts
- `experiments/tables/top_terms_positive.csv`, `top_terms_negative.csv`
  - Derived from TF-IDF coefficients over validation set.
- `experiments/tables/token_frequency_analysis.csv`
  - Token frequency delta between predicted fraud vs legit posts.
- `experiments/tables/slice_metrics.csv`
  - Metrics per demographic/industry slice (for fairness audits).
- `experiments/tables/shap_summary.csv`
  - Only populated if LightGBM or other tabular-only models win; SHAP computed via `spot_scam.explainability.shapley`.

## 5. Usage in API / UI
- FastAPI responses include explanation dictionaries:
  ```json
  {
    "probability_fraud": 0.72,
    "decision": "fraud",
    "explanation": {
      "top_positive": [...],
      "top_negative": [...],
      "intercept": -4.28,
      "summary": "Immediate start...; Quickbooks..."
    }
  }
  ```
- Frontend uses that structure directly, so any API integration (internal tooling, moderation bots) can reuse the fields.

## 6. Testing & Validation
- Unit coverage is limited; rely on integration tests by scoring known samples.
- To verify contributions manually:
  1. Load `FraudPredictor` in REPL.
  2. Run `.predict()` on a crafted posting.
  3. Inspect `result["explanation"]`.
- Future enhancement: add regression tests comparing explanation output to stored golden values for a curated set of posts.

---

Refer back to `docs/pipeline_walkthrough.md` for how the explanations fit into the broader training and serving pipeline.
