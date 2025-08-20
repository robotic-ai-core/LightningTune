# Strategy Pattern Architecture

## Overview

LightningTune now uses the **Strategy Pattern** to support multiple optimization algorithms with the same interface. This allows you to easily switch between different HPO algorithms based on your needs.

## Available Strategies

| Strategy | Best For | Characteristics |
|----------|----------|-----------------|
| **BOHB** | Expensive evaluations (>30min) | Bayesian optimization + aggressive pruning |
| **Optuna** | Moderate cost (5-30min) | Good balance, handles categoricals well |
| **Random Search** | Fast trials (<5min) | Simple baseline, high parallelism |
| **PBT** | Very long training (>2hrs) | Adaptive schedules during training |
| **Bayesian** | Small search spaces | Pure Gaussian Process, uncertainty estimates |
| **HyperOpt** | Mixed parameters | Alternative TPE implementation |
| **Grid Search** | Exhaustive search | Evaluates all combinations |

## Usage Pattern

```python
from lightning_bohb import ConfigDrivenOptimizer, OptimizationConfig

# Same interface, different strategies
optimizer = ConfigDrivenOptimizer(
    base_config_source="config.yaml",
    search_space=search_space,
    strategy="bohb",  # ← Just change this!
    strategy_config={"grace_period": 10}
)

# Or use Optuna
optimizer = ConfigDrivenOptimizer(
    base_config_source="config.yaml",
    search_space=search_space,
    strategy="optuna",  # ← Different algorithm
    strategy_config={"use_pruner": True}
)

# Or Random Search
optimizer = ConfigDrivenOptimizer(
    base_config_source="config.yaml",
    search_space=search_space,
    strategy="random",  # ← Simple baseline
)
```

## Architecture Benefits

### 1. **Single Interface**
All strategies use the same `ConfigDrivenOptimizer` interface:
```python
results = optimizer.run()
best_config = optimizer.create_production_config()
```

### 2. **Easy Comparison**
Compare strategies on the same problem:
```python
for strategy in ["bohb", "optuna", "random"]:
    optimizer = ConfigDrivenOptimizer(
        base_config_source="config.yaml",
        search_space=search_space,
        strategy=strategy
    )
    results = optimizer.run()
```

### 3. **Strategy-Specific Features**
Each strategy can have its own configuration:
```python
# BOHB with HyperBand scheduling
strategy_config = {
    "grace_period": 10,
    "reduction_factor": 3
}

# Optuna with median pruning
strategy_config = {
    "n_startup_trials": 20,
    "pruner_type": "median"
}

# PBT with adaptive learning rate
strategy_config = {
    "hyperparam_mutations": {
        "learning_rate": tune.loguniform(1e-5, 1e-2)
    }
}
```

### 4. **Custom Strategies**
Easy to add new algorithms:
```python
class MyCustomStrategy(OptimizationStrategy):
    def get_search_algorithm(self):
        return MySearchAlgorithm()
    
    def get_scheduler(self):
        return MyScheduler()

# Register and use
StrategyFactory.register("custom", MyCustomStrategy)
optimizer = ConfigDrivenOptimizer(strategy="custom", ...)
```

## Progressive Optimization Pattern

Start simple, then refine:

```python
# Stage 1: Fast exploration with Random Search (1 hour)
optimizer = ConfigDrivenOptimizer(
    strategy="random",
    optimization_config=OptimizationConfig(
        max_epochs=10,
        time_budget_hrs=1.0
    )
)
random_results = optimizer.run()

# Stage 2: Refine with Optuna (2 hours)
optimizer = ConfigDrivenOptimizer(
    strategy="optuna",
    optimization_config=OptimizationConfig(
        max_epochs=30,
        time_budget_hrs=2.0
    )
)
optuna_results = optimizer.run()

# Stage 3: Final optimization with BOHB (4 hours)
optimizer = ConfigDrivenOptimizer(
    strategy="bohb",
    optimization_config=OptimizationConfig(
        max_epochs=100,
        time_budget_hrs=4.0
    )
)
final_results = optimizer.run()
```

## Decision Guide

### Choose **BOHB** when:
- Training takes >30 minutes per trial
- You need the absolute best results
- You have smooth parameter landscapes
- You can afford 50-200 trials

### Choose **Optuna** when:
- Training takes 5-30 minutes per trial
- You have many categorical parameters
- You want good results quickly
- You need parameter importance analysis

### Choose **Random Search** when:
- Training takes <5 minutes per trial
- You're establishing a baseline
- You have massive parallel resources
- You're doing initial exploration

### Choose **PBT** when:
- Training takes >2 hours per trial
- You want adaptive learning rates
- You have stable training
- You're fine-tuning schedules

## Implementation Details

The strategy pattern is implemented through:

1. **`OptimizationStrategy`** - Abstract base class
2. **`ConfigDrivenOptimizer`** - Uses strategies via dependency injection
3. **`StrategyFactory`** - Creates strategies by name
4. **Strategy-specific classes** - Implement algorithm details

This design ensures:
- **Loose coupling** - Optimizer doesn't know strategy internals
- **High cohesion** - Each strategy is self-contained
- **Easy testing** - Mock strategies for unit tests
- **Extensibility** - Add new algorithms without changing existing code

## Backward Compatibility

The old `ConfigDrivenBOHBOptimizer` still works:
```python
# Legacy (still supported)
from lightning_bohb import ConfigDrivenBOHBOptimizer
optimizer = ConfigDrivenBOHBOptimizer(...)

# New (recommended)
from lightning_bohb import ConfigDrivenOptimizer
optimizer = ConfigDrivenOptimizer(strategy="bohb", ...)
```