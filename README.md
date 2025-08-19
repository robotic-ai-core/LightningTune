# LightningTune

**Config-driven hyperparameter optimization for PyTorch Lightning using direct Optuna integration**

LightningTune provides a clean, simplified interface for hyperparameter optimization (HPO) that integrates PyTorch Lightning with Optuna's powerful optimization algorithms. No unnecessary abstractions - just direct access to Optuna's samplers and pruners with Lightning's training loop.

## Features

- 🎯 **Direct Dependency Injection**: Use Optuna's samplers and pruners directly without wrapper abstractions
- ⚡ **PyTorch Lightning Integration**: Seamlessly works with any Lightning model and datamodule
- 📝 **Config-Driven**: Define base configurations in YAML/JSON or Python dicts
- 🔍 **Flexible Search Spaces**: Support for all Optuna distribution types
- 📊 **Built-in Visualization**: Optimization history and parameter importance plots
- 🚀 **Parallel Trials**: Support for distributed optimization
- ✂️ **Early Stopping**: Intelligent pruning of unpromising trials
- 🧪 **Well-Tested**: Comprehensive test suite including end-to-end tests

## Installation

```bash
# Basic installation
pip install -e .

# With all optional dependencies
pip install -e ".[full]"

# For development
pip install -e ".[dev,test]"
```

### Requirements

- Python >= 3.8
- PyTorch Lightning >= 1.5.0
- Optuna >= 3.0.0
- PyYAML >= 5.4

## Quick Start

```python
from LightningTune import OptunaDrivenOptimizer, SimpleSearchSpace
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Define your search space
search_space = SimpleSearchSpace({
    "model.learning_rate": ("loguniform", 1e-5, 1e-2),
    "model.hidden_size": ("categorical", [128, 256, 512]),
    "model.dropout": ("uniform", 0.1, 0.5),
    "data.batch_size": ("categorical", [16, 32, 64]),
})

# Base configuration
base_config = {
    "model": {
        "learning_rate": 0.001,
        "hidden_size": 256,
        "dropout": 0.2,
    },
    "data": {
        "batch_size": 32,
    },
    "trainer": {
        "max_epochs": 10,
        "accelerator": "auto",
    }
}

# Create optimizer with direct Optuna components
optimizer = OptunaDrivenOptimizer(
    base_config=base_config,
    search_space=search_space,
    model_class=YourLightningModule,
    datamodule_class=YourDataModule,
    sampler=TPESampler(n_startup_trials=5),  # Direct Optuna sampler
    pruner=MedianPruner(n_warmup_steps=10),  # Direct Optuna pruner
    n_trials=50,
    direction="minimize",
    metric="val_loss",
)

# Run optimization
study = optimizer.optimize()

# Get best hyperparameters
print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")
```

## Architecture

### Simplified Design Philosophy

LightningTune follows a **direct dependency injection** approach rather than unnecessary abstraction layers:

1. **No Strategy Pattern**: Instead of wrapping Optuna's components in strategy classes, we use them directly
2. **Honest Naming**: We don't pretend to have algorithms we don't (e.g., no fake BOHB or ASHA)
3. **Minimal Abstraction**: Only abstract where it adds value (search spaces, config handling)
4. **Direct Access**: Users work directly with Optuna's well-documented samplers and pruners

### Core Components

#### OptunaDrivenOptimizer
The main optimizer class that orchestrates the HPO process:
- Accepts Optuna samplers and pruners directly via constructor injection
- Manages trial execution and Lightning training loops
- Handles configuration merging and checkpoint saving
- Provides progress tracking and result visualization

#### Search Spaces
Two search space implementations for different use cases:

**SimpleSearchSpace**: For straightforward parameter definitions
```python
search_space = SimpleSearchSpace({
    "learning_rate": ("loguniform", 1e-5, 1e-2),
    "batch_size": ("categorical", [16, 32, 64]),
})
```

**AdvancedSearchSpace**: For complex dependencies and conditional parameters
```python
def define_search(trial):
    model_type = trial.suggest_categorical("model_type", ["cnn", "transformer"])
    if model_type == "cnn":
        trial.suggest_int("conv_layers", 2, 5)
    else:
        trial.suggest_int("attention_heads", 4, 16)
    return trial

search_space = AdvancedSearchSpace(define_search)
```

## Available Samplers and Pruners

### Samplers (Parameter Selection Strategies)

| Sampler | Description | Best For |
|---------|-------------|----------|
| `TPESampler` | Tree-structured Parzen Estimator | General purpose, good default choice |
| `RandomSampler` | Random search | Baseline, parallel execution |
| `GridSampler` | Grid search | Small, discrete search spaces |
| `CmaEsSampler` | CMA-ES algorithm | Continuous parameters only |

### Pruners (Early Stopping Strategies)

| Pruner | Description | Best For |
|---------|-------------|----------|
| `MedianPruner` | Stops trials below median | Stable pruning, recommended default |
| `HyperbandPruner` | Successive halving with bands | Large-scale searches |
| `SuccessiveHalvingPruner` | Aggressive early stopping | Limited computational budget |
| `NopPruner` | No pruning | Baseline comparison |

### Note on BOHB and ASHA

Optuna does **not** implement true BOHB or ASHA algorithms:
- **BOHB Alternative**: Use `TPESampler` + `HyperbandPruner` for similar results
- **ASHA Alternative**: `SuccessiveHalvingPruner` provides similar functionality (but synchronous, not async)

## Testing Methodology

### Test Suite Overview

LightningTune employs a comprehensive testing strategy with multiple test levels:

#### 1. Unit Tests
Located in `tests/test_optuna_simple.py`:
- Test individual components in isolation
- Verify search space functionality
- Check configuration merging logic
- Validate sampler/pruner integration

#### 2. Integration Tests  
Located in `tests/test_optuna_integration.py`:
- Test component interactions
- Verify Lightning integration
- Check callback functionality
- Test checkpoint saving/loading

#### 3. End-to-End Tests
Located in `tests/test_e2e_fashion_mnist.py`:
- Complete optimization pipeline on real data (Fashion-MNIST)
- Verify HPO actually improves model performance
- Test different sampler/pruner combinations
- Measure optimization effectiveness

### Testing Methodology

#### Verification Strategy

Our tests verify that HPO is actually working through multiple checks:

1. **Parameter Exploration**: Confirm different hyperparameter values are tried
2. **Performance Variation**: Ensure results vary with different parameters
3. **Optimization Improvement**: Verify best trials outperform average
4. **Statistical Validation**: Check standard deviation of results > 0

#### E2E Test Example

```python
def test_optuna_optimizer_improves_performance():
    """Verify HPO actually improves model performance."""
    
    # Run optimization with 5 trials
    study = optimizer.optimize()
    
    # Verification checks:
    # 1. Multiple parameters explored
    assert explored_params >= 2
    
    # 2. Performance varies
    assert np.std(trial_values) > 1e-6
    
    # 3. Best beats average
    improvement = (mean_val - best_val) / mean_val * 100
    assert improvement > 0
```

#### Test Coverage

- **Functional Coverage**: All major features tested
- **Edge Cases**: Handling of edge cases (empty configs, invalid parameters)
- **Error Handling**: Proper exception handling and recovery
- **Performance**: Tests complete within reasonable time limits

### Running Tests

```bash
# Run all tests
pytest tests/

# Run only fast tests
pytest tests/ -m "not slow"

# Run specific test file
pytest tests/test_e2e_fashion_mnist.py

# Run with coverage
pytest tests/ --cov=LightningTune --cov-report=html

# Run with verbose output
pytest tests/ -v -s
```

## Examples

### Basic HPO with Config File

```python
# config.yaml
model:
  class_path: models.MyModel
  init_args:
    hidden_size: 256
    learning_rate: 0.001

data:
  class_path: data.MyDataModule
  init_args:
    batch_size: 32

trainer:
  max_epochs: 10
  accelerator: gpu
```

```python
from pathlib import Path
from LightningTune import OptunaDrivenOptimizer, SimpleSearchSpace
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Load base config
base_config = Path("config.yaml")

# Define search space
search_space = SimpleSearchSpace({
    "model.init_args.hidden_size": ("categorical", [128, 256, 512]),
    "model.init_args.learning_rate": ("loguniform", 1e-5, 1e-2),
    "data.init_args.batch_size": ("categorical", [16, 32, 64]),
})

# Optimize
optimizer = OptunaDrivenOptimizer(
    base_config=base_config,
    search_space=search_space,
    model_class=MyModel,
    datamodule_class=MyDataModule,
    sampler=TPESampler(),
    pruner=MedianPruner(),
    n_trials=100,
)

study = optimizer.optimize()
```

### Advanced Search with Dependencies

```python
from LightningTune import AdvancedSearchSpace

def advanced_search(trial):
    # Model architecture decisions
    model_type = trial.suggest_categorical("model.type", ["small", "large"])
    
    if model_type == "small":
        hidden_size = trial.suggest_categorical("model.hidden_size", [64, 128])
        num_layers = trial.suggest_int("model.num_layers", 2, 4)
    else:
        hidden_size = trial.suggest_categorical("model.hidden_size", [256, 512])
        num_layers = trial.suggest_int("model.num_layers", 4, 8)
    
    # Learning rate depends on model size
    if hidden_size <= 128:
        lr = trial.suggest_float("model.lr", 1e-4, 1e-2, log=True)
    else:
        lr = trial.suggest_float("model.lr", 1e-5, 1e-3, log=True)
    
    return trial

search_space = AdvancedSearchSpace(advanced_search)
```

### Distributed HPO

```python
# Run on multiple GPUs/nodes
optimizer = OptunaDrivenOptimizer(
    base_config=config,
    search_space=search_space,
    model_class=Model,
    sampler=TPESampler(),
    pruner=HyperbandPruner(),
    n_trials=100,
    storage="postgresql://user:password@localhost/optuna",  # Shared storage
    study_name="distributed_hpo",
    load_if_exists=True,  # Continue existing study
)
```

## API Reference

### OptunaDrivenOptimizer

Main optimizer class for hyperparameter optimization.

```python
OptunaDrivenOptimizer(
    base_config: Union[str, Path, Dict],
    search_space: Union[SimpleSearchSpace, AdvancedSearchSpace],
    model_class: Type[LightningModule],
    datamodule_class: Optional[Type[LightningDataModule]] = None,
    sampler: Optional[BaseSampler] = None,
    pruner: Optional[BasePruner] = None,
    study_name: str = "optuna_study",
    direction: str = "minimize",
    n_trials: int = 100,
    metric: str = "val_loss",
    callbacks: Optional[List] = None,
    experiment_dir: Optional[Path] = None,
    save_checkpoints: bool = True,
    verbose: bool = True,
    **optuna_kwargs
)
```

**Parameters:**
- `base_config`: Base configuration (file path or dict)
- `search_space`: Search space definition
- `model_class`: PyTorch Lightning model class
- `datamodule_class`: Optional Lightning datamodule class
- `sampler`: Optuna sampler for parameter selection
- `pruner`: Optuna pruner for early stopping
- `study_name`: Name for the Optuna study
- `direction`: "minimize" or "maximize"
- `n_trials`: Number of trials to run
- `metric`: Metric to optimize
- `callbacks`: Additional Lightning callbacks
- `experiment_dir`: Directory for saving results
- `save_checkpoints`: Whether to save model checkpoints
- `verbose`: Enable verbose output

### SimpleSearchSpace

Simple search space for basic parameter definitions.

```python
SimpleSearchSpace(param_distributions: Dict[str, Tuple])
```

**Distribution Types:**
- `("uniform", low, high)`: Uniform distribution
- `("loguniform", low, high)`: Log-uniform distribution
- `("discrete_uniform", low, high, q)`: Discrete uniform
- `("int", low, high)`: Integer range
- `("int_log", low, high)`: Log-scale integer
- `("categorical", choices)`: Categorical choices

### AdvancedSearchSpace

Advanced search space for complex parameter dependencies.

```python
AdvancedSearchSpace(search_function: Callable[[Trial], Trial])
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/LightningTune.git
cd LightningTune

# Install in development mode
pip install -e ".[dev,test]"

# Run tests
pytest tests/

# Run linting
black LightningTune tests
flake8 LightningTune tests
mypy LightningTune
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of [Optuna](https://optuna.org/) and [PyTorch Lightning](https://lightning.ai/)
- Inspired by the need for simpler, more direct HPO interfaces
- Thanks to all contributors and users for feedback and improvements

## Citation

If you use LightningTune in your research, please cite:

```bibtex
@software{lightningtune,
  title = {LightningTune: Config-driven HPO for PyTorch Lightning},
  author = {LightningTune Contributors},
  year = {2024},
  url = {https://github.com/yourusername/LightningTune}
}
```