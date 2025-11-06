# Spot the Scam - Detailed Setup & Operations Guide

This guide walks through end-to-end setup, training, and serving of the Spot the Scam project. Follow the sections in order for a clean experience.

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.12+** | Project tested on 3.12; virtual environment strongly recommended. |
| **Node.js 18+ / npm 9+** | Required for the Next.js dashboard in `frontend/`. |
| **CUDA-capable GPU** | Optional but recommended for transformer training. |
| **Kaggle CLI** | Needed to download the Kaggle dataset (set up API token). |

Ensure `python3`, `pip`, `node`, and `npm` are on your `PATH`.

---

## 2. Initial Repository Setup

```bash
git clone <repo-url> spot-the-scam-project
cd spot-the-scam-project
```

Create and activate a virtual environment (example using `venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

Install Python dependencies (includes dev tooling):

```bash
pip install -e '.[dev]'
```

> [!TIP]
> **Tip:** If you hit network restrictions, rerun with `pip install --no-cache-dir -e .[dev]`.

> [!TIP]
> Remove the quote marks around `.[dev]` if using Windows Command Prompt.

---

## 3. Data Acquisition

1. Configure Kaggle CLI (`~/.kaggle/kaggle.json` must exist with your API key).
2. Download the dataset into `data/`:

```bash
./scripts/download_data.py
```

The script downloads and extracts `fake_job_postings.csv` (and the alternate `Fake_Real_Job_Posting.csv`) into `data/`. Raw files remain Git-ignored.

> [!TIP]
> We have included the full datasets in `data/` for convenience. If you re-download, existing files will be overwritten.

---

## 4. Preparing the Environment

The training pipeline expects certain directories; they are created automatically, but you can ensure they exist with:

```bash
python -c "from spot_scam.utils.paths import ensure_directories; ensure_directories()"
```

---

## 5. Training the Models

### 5.1 Full Training (classical + transformer)

```bash
PYTHONPATH=src python -m spot_scam.pipeline.train
```

This run:
- Cleans data, creates stratified splits (`data/processed/`).
- Trains classical baselines (logistic regression, linear SVM, LightGBM).
- Fine-tunes DistilBERT (mixed precision if GPU available).
- Calibrates probabilities, writes artifacts to `artifacts/`.
- Generates figures/metrics/tables in `experiments/`.

### 5.2 Classical-Only Training (faster)

```bash
PYTHONPATH=src python -m spot_scam.pipeline.train --skip-transformer
```

> **Outputs:**  
> - `artifacts/metadata.json` - summary (metrics, threshold, gray-zone).  
> - `artifacts/model.joblib` - calibrated estimator for inference.  
> - `artifacts/transformer/` - DistilBERT weights (if transformer was trained).  
> - `experiments/report.md` - markdown report with figures/tables.  

After training, you should get:

![Training Complete](experiments/figs/output.png)

### 5.3 Optional Transformer Quantization

Create an INT8 dynamic-quantized checkpoint for faster CPU inference:

```bash
PYTHONPATH=src python -m spot_scam.pipeline.quantize
# or `make quantize-transformer`
```

This writes `artifacts/transformer/quantized/model.pt` and updates metadata. Quantization is opt-in; the standard (FP32) weights remain default.

### 5.4 Automatic ONNX + MLflow Export (OPTIONAL)

Every completed training run attempts to:

1. Convert the selected winner to ONNX (both classical and transformer variants are supported).
2. Package preprocessing assets (TF‑IDF vectorizer, scaler, tokenizer) alongside the executable ONNX graph.
3. Log a full MLflow pyfunc model at `runs:/<RUN_ID>/model`, bundling the gray-zone policy so served predictions behave like the API.

Artifacts land inside the configured tracking URI (defaults to `mlruns/`). You can inspect them via the MLflow UI or serve locally:

```bash
mlflow models serve --env-manager local -m runs:/<RUN_ID>/model -p 8080
```

Use `--env-manager local` so MLflow reuses the existing `.venv` (the default `virtualenv` manager expects `pyenv` to be installed system-wide).

Set `mlflow.enabled: false` in `configs/defaults.yaml` if you need to disable export.

---

## 6. Running the FastAPI Service

Activate the virtual environment if not already active, then:

```bash
source .venv/bin/activate
PYTHONPATH=src uvicorn spot_scam.api.app:app --host 0.0.0.0 --port 8000 --reload
```

To force the API to use the quantized transformer (if available):

```bash
export SPOT_SCAM_USE_QUANTIZED=1
PYTHONPATH=src uvicorn spot_scam.api.app:app --host 0.0.0.0 --port 8000 --reload
```

If the dashboard runs on a different origin, enable CORS before launching the API:

```bash
export SPOT_SCAM_ALLOWED_ORIGINS="http://localhost:3000,https://your-domain.com"
PYTHONPATH=src uvicorn spot_scam.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Useful Endpoints

| Endpoint                         | Description                                                    |
|----------------------------------|----------------------------------------------------------------|
| `GET /health`                    | Simple health + model summary.                                 |
| `GET /metadata`                  | Model metadata, thresholds, metrics.                           |
| `POST /predict`                  | Batch predictions (`{ "instances": [JobPostingInput, ...] }`). |
| `POST /predict/single`           | Single prediction (body is `JobPostingInput`).                 |
| `GET /insights/token-importance` | Top TF‑IDF coefficients for fraud/legit terms.                 |
| `GET /insights/token-frequency`  | Token frequency differences between classes.                   |

Example curl:

```bash
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{
        "title": "Remote Data Entry Specialist",
        "description": "We are urgently hiring... purchase laptop...",
        "requirements": "Detail oriented..."
      }'
```

---

## 7. Frontend Dashboard (Next.js + Tailwind + shadcn)

### 7.1 Install Dependencies

```bash
cd frontend
npm install
```

### 7.2 Configure API URL

Copy the example env file and tweak if the API runs elsewhere:

```bash
cp .env.local.example .env.local
# edit NEXT_PUBLIC_API_BASE_URL if needed (default http://localhost:8000)
```

### 7.3 Development Server

```bash
npm run dev
```

Visit `http://localhost:3000` to access the dashboard:
- Submit job postings for scoring.
- Review calibrated metrics, decision rationale, and gray-zone band.
- Inspect token-level insights (requires classical model artifacts).

### 7.4 Linting (optional)

```bash
npm run lint
```

---

## 8. Helpful Make Targets

From project root:

| Command                 | Purpose                                                        |
|-------------------------|----------------------------------------------------------------|
| `make install`          | Install Python dependencies (same as `pip install -e .[dev]`). |
| `make train`            | Run full training pipeline.                                    |
| `make train-fast`       | Train classical models only.                                   |
| `make serve`            | Launch FastAPI server on port 8000.                            |
| `make test`             | Execute unit tests with coverage.                              |
| `make lint`             | Run Ruff + Black checks.                                       |
| `make frontend-install` | Install frontend dependencies (`npm install`).                 |
| `make frontend`         | Start Next.js dev server (`npm run dev`).                      |

---

## 9. Testing & Quality Assurance

- **Python tests:** `source .venv/bin/activate && PYTHONPATH=src pytest`
- **Frontend lint:** `cd frontend && npm run lint`
- **Manual verification:** Use FastAPI’s `/metadata` and `/predict/single` endpoints or the dashboard.

---

## 10. Docker & Devcontainer

### 10.1 Docker Compose Runtime

```bash
docker compose build
docker compose up -d
```

- FastAPI: <http://localhost:8000> · Next.js dashboard: <http://localhost:3000>.
- Host directories `configs/`, `artifacts/`, `experiments/`, and `data/` are mounted for persistence. Ensure trained artifacts exist (or train inside the container) before relying on the API.
- Execute ad-hoc tasks inside the API container:

```bash
docker compose exec api bash
PYTHONPATH=src python -m spot_scam.pipeline.train --skip-transformer
```

### 10.2 Devcontainer (VS Code)

1. Install the Dev Containers extension.
2. Open the repository and select **“Reopen in Container”**.
3. The container provisions Python 3.12 and Node 20, runs `pip install -e .[dev] && npm install --prefix frontend`, and forwards ports `8000/3000`.

---

## 11. Cleanup & Regeneration

To regenerate artifacts/experiments:

```bash
rm -rf artifacts/* experiments/* data/processed/* tracking/*
PYTHONPATH=src python -m spot_scam.pipeline.train
```

> **Warning:** Removing these folders deletes trained models and reports. Ensure you have backups if needed.

---

## 12. Troubleshooting Tips

- **Missing CUDA / GPU fallback:** Transformer training will automatically use CPU if CUDA is unavailable (slower). Ensure `torch.cuda.is_available()` returns `True` for GPU acceleration.
- **Network hiccups when installing packages:** Retry with `--no-cache-dir`, or pre-download wheels if proxies are involved.
- **Kaggle authentication errors:** Ensure `KAGGLE_USERNAME` and `KAGGLE_KEY` env vars are set or `~/.kaggle/kaggle.json` has correct permissions (`chmod 600`).
- **Frontend 404s:** Confirm FastAPI and Next.js are running; check `NEXT_PUBLIC_API_BASE_URL`.

---

You’re all set! The project now provides a calibrated fraud detector with an interactive UI, ready for experimentation and extension. Happy modeling! 🎯
