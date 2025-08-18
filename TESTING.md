# Testing Guide for Lightning BOHB

## Running Tests with pytest

The Lightning BOHB test suite uses **pytest** as the test runner. The Makefile is just a convenience wrapper - all tests are run through pytest.

## Quick Start

### Run Tests Without Dependencies (No Ray/PyTorch Required)
```bash
# Run basic tests that don't need external dependencies
python tests/test_basic.py

# Or with pytest
pytest tests/test_basic.py -v
```

### Run Unit Tests
```bash
# Run all unit tests that work without Ray
pytest tests/unit/test_strategies.py -v

# Run specific test class
pytest tests/unit/test_strategies.py::TestBOHBStrategy -v

# Run specific test
pytest tests/unit/test_strategies.py::TestBOHBStrategy::test_init -v
```

### Run All Available Tests
```bash
# Run all tests that can run in your environment
pytest tests/ -v

# With short traceback for cleaner output
pytest tests/ -v --tb=short

# Skip tests that require missing dependencies
pytest tests/ -v -m "not requires_ray"
```

## Test Organization

Tests are organized by complexity and dependencies:

```
tests/
├── test_basic.py          # No dependencies required ✅
├── unit/                  
│   ├── test_strategies.py # Works without Ray ✅
│   └── test_pause_resume.py # Requires Ray ⚠️
├── integration/           # Requires Ray ⚠️
└── e2e/                   # Requires Ray + PyTorch ⚠️
```

## pytest Commands

### Basic Usage
```bash
# Run all tests
pytest

# Run tests in a directory
pytest tests/unit/

# Run tests in a file
pytest tests/unit/test_strategies.py

# Run tests matching a pattern
pytest -k "test_pickle"

# Run with verbose output
pytest -v

# Run with minimal output
pytest -q
```

### Useful Options
```bash
# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Run with coverage
pytest --cov=lightning_bohb

# Generate HTML coverage report
pytest --cov=lightning_bohb --cov-report=html

# Run in parallel (requires pytest-xdist)
pytest -n auto
```

### Debugging
```bash
# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest -l

# Show full diff on assertion failure
pytest -vv

# Set breakpoint in test
# Add this line in your test: import pdb; pdb.set_trace()
```

## Test Markers

Tests can be marked to run selectively:

```bash
# Skip slow tests
pytest -m "not slow"

# Only run unit tests
pytest -m "unit"

# Skip tests requiring Ray
pytest -m "not requires_ray"

# Skip tests requiring GPU
pytest -m "not requires_gpu"
```

## Test Coverage

```bash
# Run with coverage
pytest --cov=lightning_bohb tests/

# Generate detailed HTML report
pytest --cov=lightning_bohb --cov-report=html tests/
# Open htmlcov/index.html in browser

# Show missing lines in terminal
pytest --cov=lightning_bohb --cov-report=term-missing tests/
```

## Environment-Specific Testing

### Without Ray (Most Tests Pass)
```bash
pytest tests/unit/test_strategies.py tests/test_basic.py -v
```

### With Ray Installed
```bash
pip install ray[tune]
pytest tests/ -v
```

### With Full Dependencies
```bash
pip install ray[tune] pytorch-lightning torch optuna
pytest tests/ -v
```

## Continuous Integration

The GitHub Actions workflow automatically runs tests on multiple Python versions:
- Python 3.8, 3.9, 3.10, 3.11
- Separate jobs for unit, integration, and e2e tests
- Coverage reporting with Codecov

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError: No module named 'ray'`:
- This is expected if Ray isn't installed
- Run tests that don't require Ray: `pytest tests/test_basic.py`
- Or install Ray: `pip install ray[tune]`

### Package Not Found
If you see `ModuleNotFoundError: No module named 'lightning_bohb'`:
- Install the package in development mode: `pip install -e .`

### Test Discovery Issues
If pytest doesn't find tests:
- Make sure you're in the project root directory
- Check that test files start with `test_`
- Verify `__init__.py` files exist in test directories

## Writing New Tests

### Test Structure
```python
import pytest

class TestMyFeature:
    def test_something(self):
        # Arrange
        obj = MyClass()
        
        # Act
        result = obj.method()
        
        # Assert
        assert result == expected
    
    def test_with_fixture(self, tmp_path):
        # pytest provides useful fixtures
        file = tmp_path / "test.txt"
        file.write_text("content")
        assert file.read_text() == "content"
```

### Using Fixtures
```python
@pytest.fixture
def my_fixture():
    # Setup
    obj = create_object()
    yield obj
    # Teardown
    cleanup(obj)

def test_with_fixture(my_fixture):
    assert my_fixture.attribute == expected
```

## Summary

- **pytest** is the test runner (not make)
- Tests work without Ray/PyTorch for core functionality
- Use `pytest tests/test_basic.py` for quick validation
- Use markers to skip tests based on dependencies
- Coverage reports help identify untested code

The test suite is designed to be flexible - core functionality can be tested without heavy dependencies, while full integration tests are available when all dependencies are installed.