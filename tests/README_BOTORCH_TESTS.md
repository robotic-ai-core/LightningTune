# BoTorchSampler Test Coverage

This document describes the test coverage for BoTorchSampler support in LightningTune.

## Test Files

### 1. `test_factories.py`
Tests the factory functions for creating samplers including BoTorch:
- ✅ `test_create_botorch_sampler` - Creates BoTorchSampler when available
- ✅ `test_botorch_not_available_fallback` - Graceful failure when not installed
- ✅ `test_get_sampler_info` - BoTorch appears in info only when available

### 2. `test_botorch_integration.py`
Comprehensive integration tests for BoTorchSampler:
- ✅ Factory integration
- ✅ Custom parameters (n_startup_trials, independent_sampler)
- ✅ PausibleOptunaOptimizer compatibility
- ✅ Optimization loop integration
- ✅ Graceful fallback handling
- ✅ Continuous parameter optimization
- ✅ Different acquisition functions (EI, UCB, PI)

### 3. `test_hpo_botorch_cli.py`
Command-line interface tests:
- ✅ CLI argument acceptance
- ✅ Error messages when not installed
- ✅ Available samplers listing

## Running Tests

```bash
# Run all BoTorch-related tests
pytest -k botorch -v

# Run without BoTorch installed (tests fallback behavior)
pip uninstall optuna-integration
pytest tests/test_factories.py -k botorch

# Run with BoTorch installed (tests full functionality)
pip install optuna-integration[botorch]
pytest tests/test_botorch_integration.py -v
```

## Coverage Matrix

| Feature | Without BoTorch | With BoTorch |
|---------|----------------|--------------|
| Factory creation | ❌ ValueError | ✅ Creates sampler |
| Sampler info | Not listed | Listed with description |
| CLI argument | Accepted | Accepted |
| Optimization | Falls back to TPE | Uses GP-based BO |
| Error handling | Clear message | N/A |

## Conditional Testing

Tests use `pytest.mark.skipif` to handle BoTorch availability:

```python
@pytest.mark.skipif(not BOTORCH_AVAILABLE, reason="BoTorch not installed")
def test_botorch_feature():
    # Test only runs if BoTorch is installed
    pass
```

## Continuous Integration

For CI/CD, consider:

```yaml
# .github/workflows/test.yml
- name: Test without BoTorch
  run: pytest tests/ -v

- name: Test with BoTorch
  run: |
    pip install optuna-integration[botorch]
    pytest tests/ -v
```

## Notes

1. **Graceful Degradation**: When BoTorch is not installed, the system gracefully falls back to other samplers without breaking.

2. **Dynamic Availability**: The sampler list dynamically adjusts based on what's installed.

3. **Clear Error Messages**: Users get helpful instructions if they try to use BoTorch without the required packages.

4. **No Hard Dependencies**: BoTorch remains an optional enhancement, not a requirement.