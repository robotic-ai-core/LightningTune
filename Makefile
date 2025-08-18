.PHONY: help install test test-unit test-integration test-e2e coverage lint format clean

help:
	@echo "Lightning BOHB Development Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install       Install package in development mode"
	@echo "  make install-test  Install testing dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run all tests"
	@echo "  make test-unit     Run unit tests"
	@echo "  make test-integration Run integration tests"
	@echo "  make test-e2e      Run end-to-end tests"
	@echo "  make coverage      Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code with black and isort"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove build artifacts and cache"

install:
	pip install -e .
	pip install pytorch-lightning torch --index-url https://download.pytorch.org/whl/cpu
	pip install ray[tune] optuna hyperopt

install-test:
	pip install -r requirements-test.txt

test:
	pytest tests -v

test-unit:
	pytest tests/unit -v

test-integration:
	pytest tests/integration -v

test-e2e:
	pytest tests/e2e -v --timeout=300

coverage:
	pytest tests --cov=lightning_bohb --cov-report=term-missing --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

lint:
	flake8 lightning_bohb tests --max-line-length=100 --ignore=E203,W503
	mypy lightning_bohb --ignore-missing-imports
	isort --check-only lightning_bohb tests
	black --check lightning_bohb tests

format:
	isort lightning_bohb tests
	black lightning_bohb tests

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf **/__pycache__ **/*.pyc
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete