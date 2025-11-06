FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        git && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./

RUN pip install --upgrade pip setuptools wheel && pip install .

COPY configs ./configs
COPY scripts ./scripts
COPY src ./src

RUN mkdir -p artifacts experiments mlruns

EXPOSE 8000

CMD ["uvicorn", "spot_scam.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
