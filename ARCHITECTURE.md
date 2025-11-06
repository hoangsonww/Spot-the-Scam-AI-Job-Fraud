# Spot the Scam - Architecture Overview

This document describes the technical architecture of Spot the Scam across data, modeling, inference, and presentation layers. It also highlights the major modules, data flow, and deployment footprint.

---

## 1. High-Level System Diagram

```mermaid
flowchart TD
    subgraph Offline Training
        A1[Raw Kaggle CSV] -->|Download| A2[Data Ingestion]
        A2 --> A3[Preprocessing & Feature Engineering]
        A3 --> A4[Classical Models]
        A3 --> A5[DistilBERT Fine-tuning]
        A4 --> A6[Calibration & Selection]
        A5 --> A6
        A6 -->|Persist| A7[Artifacts & Reports]
    end

    subgraph Online Serving
        B1[FastAPI Service]
        B2[FraudPredictor]
        B3[Artifacts - Model, Feature Pipelines]
    end
    A7 -->|Load| B3
    B3 --> B2 --> B1

    subgraph Frontend & Registry
        C1[Next.js Dashboard]
        C2[MLflow Model Registry]
    end
    B1 <-->|REST| C1
    A6 -->|Register| C2
    C2 -->|pyfunc / ONNX| B1
```

---

## 2. Repository Layout

| Path | Description |
|------|-------------|
| `configs/` | YAML configs (defaults, overrides). |
| `scripts/` | Utility CLIs (`download_data.py`, `run_api.py`). |
| `src/spot_scam/` | Python package (ETL, features, models, evaluation, inference, API). |
| `artifacts/` | Generated model assets (vectorizers, weights, metadata). |
| `experiments/` | Reports, figures, tables after training. |
| `frontend/` | Next.js + Tailwind + shadcn dashboard. |
| `tests/` | Python unit tests. |

---

## 3. Python Package Architecture

```mermaid
graph LR
    subgraph Data Pipeline
        D1[data.ingest]
        D2[data.preprocess]
        D3[data.split]
    end
    subgraph Feature Eng.
        F1[features.text]
        F2[features.tabular]
        F3[features.builders]
    end
    subgraph Models
        M1[models.classical]
        M2[models.transformer]
    end
    subgraph Evaluation
        E1[evaluation.metrics]
        E2[evaluation.curves]
        E3[evaluation.calibration]
        E4[evaluation.bias]
        E5[evaluation.reporting]
    end
    subgraph Inference and API
        I1[inference.predictor]
        I2[policy.gray_zone]
        I3[api.schemas]
        I4[api.app]
    end

    D1 --> D2 --> D3 --> F3
    F3 --> M1
    F3 --> M2
    M1 --> E1
    M2 --> E1
    E1 --> E2
    E1 --> E3
    E1 --> E4
    M1 --> I1
    M2 --> I1
    I1 --> I2
    I1 --> I4
    I4 --> I3
    I1 --> I5[explanations.local]
```

### Key Modules

- **Data ingest / preprocess / split**  
  Standardize columns, drop leakage, compute checksums, stratify into train/val/test with persisted indices.

- **Feature builders**  
  - `features.text`: TF-IDF vectorizer.  
  - `features.tabular`: engineered features (lengths, counts, binary flags).  
  - `features.builders`: orchestrates vectorizer + tabular scaler combo.

- **Models**  
  - `models.classical`: logistic regression, linear SVM, LightGBM with grid search.  
  - `models.transformer`: DistilBERT fine-tune with HF Trainer (AMP + early stopping).

- **Evaluation**  
  - Metrics (F1, PR-AUC, calibration).  
  - Plots (PR, calibration, confusion).  
  - Calibration (Platt / isotonic).  
  - Bias slice analysis.  
  - Markdown report generation.

- **Inference**  
  - `FraudPredictor`: loads artifacts, preprocessing pipelines, and gray-zone policy.  
  - `policy.gray_zone`: threshold band logic.  
  - FastAPI routes (`api.app`) with typed schemas, returning calibrated scores and local explanations per request.
  - MLflow export utilities (`export/mlflow_logger.py`) package ONNX + preprocessing into a pyfunc model for serving parity.

---

## 4. Training Flow (Detailed)

```mermaid
sequenceDiagram
    participant CLI as CLI (Typer)
    participant Config as Config Loader
    participant Data as Data Pipeline
    participant Features as Feature Builder
    participant Models as Model Trainers
    participant Eval as Evaluation Suite
    participant Persist as Artifact Writer

    CLI->>Config: load_config()
    CLI->>Data: load_raw_dataset()
    Data->>Data: preprocess_dataframe()
    Data->>Data: create_splits()
    Data->>Features: build_feature_bundle()
    Features->>Models: train_classical_models()
    Features->>Models: train_transformer_model()
    Models->>Eval: compute_metrics()
    Eval->>Persist: save artifacts + plots + tables
    Persist->>CLI: append_run_record()
```

### 4.1 Python Entrypoints

```mermaid
flowchart TD
    A[Typer CLI
    spot_scam.pipeline.train] --> B[load_config]
    B --> C[load_raw_dataset & merge CSVs]
    C --> D[preprocess_dataframe]
    D --> E[create_splits & persist indices]
    E --> F[build_feature_bundle]
    F --> G[train_classical_models]
    F --> H[train_transformer_model]
    G --> I[select best model]
    H --> I
    I --> J[calibrate & evaluate]
    J --> K[persist artifacts & metadata]
    K --> L[generate reports & benchmarks]
    L --> M[append tracking CSV]
```

---

## 5. Artifacts & Reporting

| Location                         | Contents                                               |
|----------------------------------|--------------------------------------------------------|
| `artifacts/model.joblib`         | Calibrated estimator (used in inference).              |
| `artifacts/base_model.joblib`    | Uncalibrated base (pre-calibration).                   |
| `artifacts/features/`            | TF-IDF vectorizer (`*.joblib`), scaler, feature names. |
| `artifacts/transformer/`         | DistilBERT checkpoints (`best/`, tokenizers).          |
| `artifacts/metadata.json`        | Metrics summary, gray-zone policy, threshold.          |
| `artifacts/test_predictions.csv` | Final test set with decisions.                         |
| `experiments/figs/`              | PR curve, calibration curve, confusion matrix.         |
| `experiments/tables/`            | Token importances, frequency analysis, slice metrics.  |
| `experiments/report.md`          | Markdown report for quick consumption.                 |

---

## 6. Inference Architecture

1. **FastAPI** loads a singleton `FraudPredictor` (cached via `functools.lru_cache`).
2. `FraudPredictor` restores model weights, vectorizer, scaler, and metadata.
3. `/predict` route accepts Pydantic models, runs preprocessing → features → scoring → calibration → gray-zone assignment.
4. Additional routes expose metadata, token importance, and frequency analysis for the frontend.

---

## 7. Frontend Architecture (Next.js)

```mermaid
flowchart LR
    FApp[Next.js App Router] -->|useSWR| FAPI[lib/api.ts]
    FAPI -->|fetch| REST[(FastAPI Endpoints)]
    FApp --> FUI[shadcn UI Components]
    FApp --> State[React State Hooks]

    subgraph Pages and Components
        page[/app/page.tsx/]
        HomePage[components/home-page.tsx]
        UI[components/ui/*]
    end
    page --> HomePage
    HomePage --> UI
```

![UI Screenshot](experiments/figs/ui.png)

### Frontend Highlights
- App directory with `page.tsx` wrapper around `HomePage`.
- `lib/api.ts` centralizes REST calls (metadata, predictions, insights).
- `home-page.tsx` uses SWR for metadata + insights, handles prediction form, and renders natural-language rationales for each decision alongside contribution lists.
- shadcn components (Card, Tabs, Table, Badge, etc.) for cohesive styling.
- `.env.local` controls `NEXT_PUBLIC_API_BASE_URL`.

---

## 8. Environment & Configuration

- **Configuration loader (`config/loader.py`)** merges `configs/defaults.yaml` with optional overrides. Dot-notation overrides supported.
- **Paths utility (`utils/paths.py`)** centralizes directories relative to project root (artifacts, experiments, tracking, etc.).
- **Experiment tracking (`tracking/logger.py`)** appends CSV entries with run metadata.

---

## 9. Quality & Testing

- Python unit tests validate configuration, ingest, and policy utilities (`pytest`).
- Linting via `ruff` and `black`.
- Frontend lint via `eslint`.
- CLI orchestration via `Makefile` shortcuts.

---

With this architecture, Spot the Scam maintains a reproducible end-to-end pipeline, calibrated serving layer, and user-facing analytics dashboard ready for operational deployment or further research iterations. !*** End Patch
