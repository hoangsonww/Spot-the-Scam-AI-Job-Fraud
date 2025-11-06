FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./

RUN pip install --upgrade pip setuptools wheel && pip install .

COPY artifacts ./artifacts
COPY experiments ./experiments
COPY mlruns ./mlruns

ENV SPOT_SCAM_ARTIFACT_DIR=/app/artifacts \
    SPOT_SCAM_EXPERIMENT_DIR=/app/experiments \
    SPOT_SCAM_MLFLOW_HOME=/app/mlruns

CMD ["bash"]
