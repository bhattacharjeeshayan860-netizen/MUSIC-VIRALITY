.PHONY: help install test run clean lint format

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Run tests"
	@echo "  make run         - Run dashboard"
	@echo "  make clean       - Clean up cache and artifacts"
	@echo "  make lint        - Run linting"
	@echo "  make format      - Format code"

install:
	pip install -r requirements.txt

test:
	pytest tests/

run:
	streamlit run dashboard/app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache
	rm -rf build/ dist/ *.egg-info/

lint:
	pylint src/ dashboard/

format:
	black src/ dashboard/ tests/
