# Lightning BOHB Test Suite

## Overview

Comprehensive test suite for the Lightning BOHB hyperparameter optimization package with pause/resume capabilities.

## Test Structure

```
tests/
├── unit/                   # Unit tests for individual components
│   ├── test_strategies.py  # Strategy pattern tests
│   └── test_pause_resume.py # Pause/resume component tests
├── integration/            # Integration tests
│   └── test_optimizer_integration.py
├── e2e/                   # End-to-end tests
│   ├── test_full_optimization.py
│   └── test_pause_resume_workflow.py
├── fixtures/              # Test fixtures and utilities
│   ├── dummy_model.py     # Dummy PyTorch Lightning model
│   └── test_config.yaml   # Test configuration
├── test_basic.py          # Basic tests (no dependencies)
└── conftest.py            # Pytest configuration
```

## Test Categories

### 1. Unit Tests (`tests/unit/`)

#### **test_strategies.py**
Tests for optimization strategy implementations:
- ✅ Strategy initialization
- ✅ Dependency injection pattern
- ✅ Strategy pickling/unpickling
- ✅ Strategy self-containment
- ✅ All strategies (BOHB, Optuna, Random, PBT, Grid)

#### **test_pause_resume.py**
Tests for pause/resume functionality:
- ✅ TuneSessionState serialization
- ✅ TuneReflowCLI initialization
- ✅ Optimizer save/restore
- ✅ Pause signal detection
- ✅ Checkpoint management
- ✅ Resume command generation

### 2. Integration Tests (`tests/integration/`)

#### **test_optimizer_integration.py**
Tests optimizer integration with strategies:
- ✅ Optimizer initialization with strategies
- ✅ Config merging
- ✅ Strategy parameters usage
- ✅ Callback injection
- ✅ Different strategy compatibility

### 3. End-to-End Tests (`tests/e2e/`)

#### **test_full_optimization.py**
Complete optimization workflows:
- ✅ Basic optimization workflow
- ✅ BOHB strategy E2E
- ✅ Pause/resume E2E
- ✅ Callbacks integration
- ✅ Multiple strategies E2E

#### **test_pause_resume_workflow.py**
Complete pause/resume cycles:
- ✅ Complete pause/resume cycle
- ✅ Pause signal detection
- ✅ Session state completeness
- ✅ CLI workflow simulation

### 4. Basic Tests (`tests/test_basic.py`)

No-dependency tests that verify core functionality:
- ✅ Strategy imports
- ✅ Strategy pickling
- ✅ Session state serialization
- ✅ Dependency injection pattern
- ✅ Pause callback basic functionality

## Running Tests

### Quick Test (No Dependencies)
```bash
# Run basic tests without Ray/PyTorch
python tests/test_basic.py
```

### Unit Tests
```bash
# Requires pytest
pytest tests/unit -v
```

### Integration Tests
```bash
# Requires Ray
pytest tests/integration -v
```

### End-to-End Tests
```bash
# Requires Ray and PyTorch Lightning
pytest tests/e2e -v --timeout=300
```

### Full Test Suite
```bash
# Run all tests
pytest tests -v

# With coverage
pytest tests --cov=lightning_bohb --cov-report=html
```

### Using Make
```bash
make test           # Run all tests
make test-unit      # Run unit tests only
make test-integration # Run integration tests
make test-e2e       # Run end-to-end tests
make coverage       # Run with coverage report
```

## Test Requirements

### Minimal (Basic Tests)
- Python 3.8+
- No external dependencies

### Unit Tests
- pytest >= 7.0.0
- pytest-mock >= 3.10.0

### Integration Tests
- All unit test requirements
- ray[tune] >= 2.0.0
- optuna >= 3.0.0 (optional)

### End-to-End Tests
- All integration test requirements
- pytorch-lightning >= 1.5.0
- torch >= 1.8.0

## CI/CD Configuration

GitHub Actions workflow (`.github/workflows/test.yml`):
- Matrix testing: Python 3.8, 3.9, 3.10, 3.11
- Parallel test execution by type
- Coverage reporting with Codecov
- Linting with flake8, black, isort, mypy

## Test Coverage

Target coverage: 80%+

Key areas covered:
- ✅ Strategy creation and configuration
- ✅ Optimizer initialization
- ✅ Config merging
- ✅ Pause/resume state management
- ✅ Signal-based communication
- ✅ Checkpoint handling
- ✅ Resume command generation
- ✅ Callback integration

## Test Fixtures

### `DummyModel` 
Simple PyTorch Lightning model for testing:
- Configurable architecture
- Fast training (small dataset)
- Metrics logging

### `DummyDataModule`
Simple data module with:
- Random synthetic data
- Configurable size
- Train/val splits

## Known Issues

1. **Ray dependency**: Many tests require Ray, which may not be available in all environments
2. **GPU tests**: Currently all tests run on CPU for portability
3. **Long-running tests**: E2E tests can take several minutes

## Future Improvements

- [ ] Add performance benchmarks
- [ ] Add GPU-specific tests
- [ ] Add distributed training tests
- [ ] Add stress tests for pause/resume
- [ ] Add property-based testing with Hypothesis
- [ ] Add mutation testing

## Test Philosophy

1. **Fail Fast**: Tests should fail quickly and clearly
2. **No Mocking Ray**: Integration/E2E tests use real Ray when possible
3. **Minimal Fixtures**: Use simple, fast fixtures
4. **Clear Names**: Test names describe what they test
5. **Independence**: Tests should not depend on each other

## Debugging Tests

```bash
# Run specific test
pytest tests/unit/test_strategies.py::TestBOHBStrategy::test_init -v

# Run with debugging output
pytest tests -v -s

# Run with pdb on failure
pytest tests --pdb

# Run specific marker
pytest tests -m "not requires_ray"
```

## Summary

The test suite provides comprehensive coverage of:
- ✅ Core functionality without dependencies
- ✅ Unit-level component testing
- ✅ Integration between components
- ✅ End-to-end workflows
- ✅ Pause/resume capabilities

This ensures the Lightning BOHB package is robust, reliable, and production-ready.