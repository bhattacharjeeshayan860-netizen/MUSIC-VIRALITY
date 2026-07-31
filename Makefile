.PHONY: help install install-dev test test-fast run run-pred api clean lint format

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  make install      - Install runtime dependencies"
	@echo "  make install-dev  - Install runtime + dev dependencies"
	@echo "  make test         - Run tests with coverage"
	@echo "  make test-fast    - Run tests quickly"
	@echo "  make run          - Run detection dashboard"
	@echo "  make run-pred     - Run prediction dashboard"
	@echo "  make api          - Run FastAPI server"
	@echo "  make clean        - Clean up cache and artifacts"
	@echo "  make lint         - Run linting (requires pylint)"
	@echo "  make format       - Format code (requires black)"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -q

run:
	$(PYTHON) -m streamlit run dashboard.py

run-pred:
	$(PYTHON) -m streamlit run dashboard_prediction.py

api:
	$(PYTHON) -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true

lint:
	pylint src/ dashboard.py dashboard_prediction.py

format:
	black src/ dashboard.py dashboard_prediction.py tests/
