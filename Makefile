PYTHON ?= python3

.PHONY: install train train-fast serve test lint format format-check lint-fix type-check check-all clean help
.PHONY: frontend-install frontend frontend-format frontend-format-check frontend-lint frontend-lint-fix frontend-type-check frontend-check

help:
	@echo "Python Backend Commands:"
	@echo "  make install        - Install Python dependencies"
	@echo "  make format         - Format Python code with Black"
	@echo "  make format-check   - Check Python code formatting"
	@echo "  make lint           - Lint Python code with Ruff"
	@echo "  make lint-fix       - Lint and auto-fix Python code"
	@echo "  make type-check     - Run mypy type checking"
	@echo "  make test           - Run pytest tests"
	@echo "  make check-all      - Run all Python checks (format, lint, type-check)"
	@echo "  make clean          - Clean cache and build files"
	@echo ""
	@echo "Frontend Commands:"
	@echo "  make frontend-install      - Install frontend dependencies"
	@echo "  make frontend              - Run frontend dev server"
	@echo "  make frontend-format       - Format TypeScript code"
	@echo "  make frontend-format-check - Check TypeScript formatting"
	@echo "  make frontend-lint         - Lint TypeScript code"
	@echo "  make frontend-lint-fix     - Lint and auto-fix TypeScript"
	@echo "  make frontend-type-check   - Run TypeScript type checking"
	@echo "  make frontend-check        - Run all frontend checks"
	@echo ""
	@echo "Training & Serving:"
	@echo "  make train          - Train models"
	@echo "  make train-fast     - Train without transformer"
	@echo "  make serve          - Start API server"
	@echo ""
	@echo "Please refer to the INSTRUCTIONS.md file for more details on each of these commands."

install:
	$(PYTHON) -m pip install -e '.[dev]'

# Python formatting
format:
	black src/ tests/ scripts/

format-check:
	black --check src/ tests/ scripts/

# Python linting
lint:
	ruff check src/ tests/ scripts/

lint-fix:
	ruff check --fix src/ tests/ scripts/

# Python type checking
type-check:
	mypy src/

# Run all Python checks
check-all: format-check lint type-check
	@echo "✓ All Python checks passed!"

# Clean Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ .coverage htmlcov/

train:
	PYTHONPATH=src $(PYTHON) -m spot_scam.pipeline.train

train-fast:
	PYTHONPATH=src $(PYTHON) -m spot_scam.pipeline.train --skip-transformer

serve:
	PYTHONPATH=src uvicorn spot_scam.api.app:app --host 0.0.0.0 --port 8000 --reload

test:
	PYTHONPATH=src pytest

frontend-install:
	npm install --prefix frontend

frontend:
	npm run dev --prefix frontend

frontend-format:
	cd frontend && npm run format

frontend-format-check:
	cd frontend && npm run format:check

frontend-lint:
	cd frontend && npm run lint

frontend-lint-fix:
	cd frontend && npm run lint:fix

frontend-type-check:
	cd frontend && npm run type-check

frontend-check: frontend-format-check frontend-lint frontend-type-check
	@echo "✓ All frontend checks passed!"

quantize-transformer:
	PYTHONPATH=src $(PYTHON) -m spot_scam.pipeline.quantize

serve-queue:
	PYTHONPATH=src $(PYTHON) -m uvicorn spot_scam.api.app:app --host 0.0.0.0 --port 8000 --reload

review-sample:
	PYTHONPATH=src $(PYTHON) scripts/sample_uncertain.py --limit 50

retrain-with-feedback:
	USE_FEEDBACK=1 PYTHONPATH=src $(PYTHON) -m spot_scam.pipeline.train
