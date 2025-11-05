PYTHON ?= python3

.PHONY: install train train-fast serve test lint frontend-install frontend

install:
	$(PYTHON) -m pip install -e .[dev]

train:
	PYTHONPATH=src $(PYTHON) -m spot_scam.pipeline.train

train-fast:
	PYTHONPATH=src $(PYTHON) -m spot_scam.pipeline.train --skip-transformer

serve:
	PYTHONPATH=src uvicorn spot_scam.api.app:app --host 0.0.0.0 --port 8000 --reload

test:
	PYTHONPATH=src pytest

lint:
	ruff check src tests && black --check src tests

frontend-install:
	npm install --prefix frontend

frontend:
	npm run dev --prefix frontend

quantize-transformer:
	PYTHONPATH=src $(PYTHON) -m spot_scam.pipeline.quantize transformer
