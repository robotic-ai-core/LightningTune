# Testing Documentation

## Testing Philosophy

LightningTune employs a rigorous testing methodology to ensure that hyperparameter optimization not only runs without errors but actually **improves model performance**. Our tests go beyond simple unit tests to verify the statistical effectiveness of optimization.

## Test Structure

```
tests/
├── test_optuna_simple.py          # Unit tests for components
├── test_optuna_integration.py     # Integration tests
├── test_e2e_fashion_mnist.py      # End-to-end tests on real data
├── test_e2e_optuna_fashion_mnist.py  # Pytest-compatible E2E tests
└── test_hpo_working.py            # Minimal verification test
```

## Testing Levels

### 1. Unit Tests

**Purpose**: Test individual components in isolation

**Coverage**:
- Search space functionality
- Configuration merging
- Parameter suggestion
- Utility functions

**Example**:
```python
def test_simple_search_space():
    """Test SimpleSearchSpace parameter suggestions."""
    search_space = SimpleSearchSpace({
        "learning_rate": ("loguniform", 1e-5, 1e-2),
        "batch_size": ("categorical", [16, 32, 64]),
    })
    
    # Create mock trial
    trial = MockTrial()
    params = search_space.suggest_params(trial)
    
    # Verify structure
    assert "learning_rate" in params
    assert "batch_size" in params
    assert 1e-5 <= params["learning_rate"] <= 1e-2
    assert params["batch_size"] in [16, 32, 64]
```

### 2. Integration Tests

**Purpose**: Test component interactions and Lightning integration

**Coverage**:
- Optimizer initialization
- Trial execution
- Callback integration
- Checkpoint handling

**Example**:
```python
def test_optimizer_with_lightning():
    """Test optimizer integrates with Lightning correctly."""
    optimizer = OptunaDrivenOptimizer(
        base_config=config,
        search_space=search_space,
        model_class=DummyModel,
        sampler=TPESampler(),
        pruner=MedianPruner(),
        n_trials=3,
    )
    
    study = optimizer.optimize()
    
    # Verify optimization ran
    assert len(study.trials) == 3
    assert study.best_value is not None
```

### 3. End-to-End Tests

**Purpose**: Verify HPO actually improves performance on real data

**Coverage**:
- Complete optimization pipeline
- Real model training (Fashion-MNIST CNN)
- Statistical validation
- Different sampler/pruner combinations

**Key Test**: `test_optuna_optimizer_improves_performance`

## Statistical Validation Methodology

### 1. Parameter Exploration Verification

We verify that the optimizer actually explores different parameter values:

```python
# Collect unique values for each parameter
param_variations = {}
for trial in study.trials:
    for key, value in trial.params.items():
        if key not in param_variations:
            param_variations[key] = set()
        param_variations[key].add(value)

# Count parameters with multiple values tried
explored_params = sum(1 for values in param_variations.values() if len(values) > 1)
assert explored_params >= 2, "Insufficient parameter exploration"
```

### 2. Performance Variation Check

We ensure that different parameters lead to different performance:

```python
trial_values = [t.value for t in study.trials if t.value is not None]
std_dev = np.std(trial_values)
assert std_dev > 1e-6, "No variation in results - optimization not working"
```

### 3. Optimization Improvement Validation

We verify that optimization finds better configurations than random:

```python
best_value = min(trial_values)
mean_value = np.mean(trial_values)
improvement = (mean_value - best_value) / mean_value * 100

assert best_value < mean_value, "Best not better than average"
print(f"Best trial {improvement:.1f}% better than average")
```

### 4. Pruning Effectiveness Test

We verify that pruning actually stops bad trials:

```python
pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

if len(pruned_trials) > 0:
    # Pruning is working
    avg_pruned_epochs = mean([t.user_attrs.get("epochs", 0) for t in pruned_trials])
    avg_completed_epochs = mean([t.user_attrs.get("epochs", 0) for t in completed_trials])
    assert avg_pruned_epochs < avg_completed_epochs, "Pruning not stopping trials early"
```

## Test Data

### Fashion-MNIST Dataset

We use Fashion-MNIST for E2E tests because:
- Small enough for fast testing (28x28 grayscale images)
- Complex enough to show optimization benefits
- Standard benchmark dataset
- No external dependencies

**Data Configuration**:
- Training samples: 6,000 (subset for speed)
- Validation samples: 1,000
- Classes: 10 fashion categories
- Preprocessing: Normalization to [-1, 1]

### Test Model Architecture

Simple CNN with tunable hyperparameters:

```python
class FashionMNISTModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.001,      # Tunable
        dropout_rate: float = 0.2,         # Tunable
        hidden_size: int = 128,            # Tunable
        conv_channels: int = 32,           # Tunable
        optimizer_type: str = "adam",      # Tunable
    ):
        # Two conv layers + two FC layers
        # ~400K parameters total
```

## Running Tests

### Quick Tests (Unit + Integration)

```bash
# Run fast tests only
pytest tests/ -m "not slow" -v

# Expected time: < 1 minute
```

### Full Test Suite

```bash
# Run all tests including E2E
pytest tests/ -v

# Expected time: 5-10 minutes (depends on GPU)
```

### Specific Test Categories

```bash
# Only unit tests
pytest tests/test_optuna_simple.py -v

# Only integration tests  
pytest tests/test_optuna_integration.py -v

# Only E2E tests
pytest tests/test_e2e_fashion_mnist.py -v
```

### With Coverage Report

```bash
# Generate coverage report
pytest tests/ --cov=LightningTune --cov-report=html

# View report
open htmlcov/index.html
```

## Test Fixtures and Utilities

### Mock Models

```python
class DummyModel(pl.LightningModule):
    """Minimal model for testing."""
    def __init__(self, learning_rate=0.001, hidden_size=32):
        super().__init__()
        self.layer = nn.Linear(10, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
```

### Mock Data

```python
def create_dummy_dataloader():
    """Create simple synthetic data."""
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32)
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install -e ".[test]"
    - name: Run tests
      run: |
        pytest tests/ -v --cov=LightningTune
```

## Test Metrics and Benchmarks

### Expected Test Results

| Test Type | Expected Time | Success Criteria |
|-----------|--------------|------------------|
| Unit Tests | < 10s | All pass |
| Integration Tests | < 30s | All pass |
| E2E Tests (CPU) | < 5 min | Best > 20% better than average |
| E2E Tests (GPU) | < 2 min | Best > 20% better than average |

### Performance Benchmarks

Fashion-MNIST CNN optimization (5 trials):
- Parameter exploration: ≥ 3 parameters with multiple values
- Performance std dev: > 0.05
- Best vs average improvement: > 15%
- Pruned trials: ≥ 20% of total (with MedianPruner)

## Debugging Failed Tests

### Common Issues and Solutions

**Issue**: Tests fail with "No variation in results"
- **Cause**: All trials returning same value
- **Solution**: Check search space ranges, ensure parameters affect model

**Issue**: "Best not better than average"
- **Cause**: Too few trials or bad search space
- **Solution**: Increase n_trials or expand search ranges

**Issue**: Import errors
- **Cause**: Missing dependencies
- **Solution**: Install with `pip install -e ".[test]"`

**Issue**: CUDA out of memory
- **Cause**: Too many parallel trials or large batch size
- **Solution**: Reduce batch size or set smaller model

### Debug Mode

Run tests with verbose output:

```bash
# Maximum verbosity
pytest tests/ -vvs --tb=long

# With debug logging
pytest tests/ --log-cli-level=DEBUG
```

## Adding New Tests

### Test Template

```python
import pytest
from LightningTune import OptunaDrivenOptimizer

class TestNewFeature:
    """Test suite for new feature."""
    
    def test_basic_functionality(self):
        """Test basic feature works."""
        # Setup
        optimizer = OptunaDrivenOptimizer(...)
        
        # Execute
        result = optimizer.some_method()
        
        # Verify
        assert result.meets_expectation()
    
    @pytest.mark.slow
    def test_e2e_scenario(self):
        """Test feature in realistic scenario."""
        # Run full optimization
        study = optimizer.optimize()
        
        # Statistical validation
        assert self.validate_improvement(study)
```

### Test Checklist

- [ ] Unit test for new component
- [ ] Integration test with existing components
- [ ] E2E test if affects optimization
- [ ] Update test documentation
- [ ] Verify CI passes

## Test Coverage Goals

- **Line Coverage**: > 80%
- **Branch Coverage**: > 70%
- **E2E Scenarios**: All major use cases
- **Error Cases**: All expected exceptions

## Future Testing Improvements

1. **Property-based testing** with Hypothesis
2. **Benchmark suite** comparing different optimizers
3. **Regression tests** for performance
4. **Distributed optimization tests**
5. **Multi-objective optimization tests**