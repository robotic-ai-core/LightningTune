# LightningTune - Simplified Optuna Integration

## Direct Dependency Injection Approach

LightningTune now uses direct dependency injection for Optuna's samplers and pruners. No unnecessary abstractions or fake algorithm names.

## What Changed?

### Before (with strategy abstraction):
```python
# Note: Use TPESampler with HyperbandPruner for multi-fidelity optimization

strategy = # TPESampler with HyperbandPruner
optimizer = OptunaDrivenOptimizer(..., strategy=strategy)
```

### Now (direct dependency injection):
```python
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from LightningTune import OptunaDrivenOptimizer

optimizer = OptunaDrivenOptimizer(
    base_config="config.yaml",
    search_space=search_space,
    model_class=MyModel,
    sampler=TPESampler(),      # Direct Optuna sampler
    pruner=HyperbandPruner(),  # Direct Optuna pruner
)
```

## Available Optuna Components

### Samplers (How to choose hyperparameters)
- `TPESampler` - Tree-structured Parzen Estimator (default, most popular)
- `RandomSampler` - Random search
- `GridSampler` - Grid search
- `CmaEsSampler` - CMA-ES for continuous parameters

### Pruners (When to stop unpromising trials)
- `MedianPruner` - Stop if below median (default)
- `HyperbandPruner` - Hyperband algorithm
- `SuccessiveHalvingPruner` - Successive halving (NOT ASHA - synchronous only)
- `NopPruner` - No pruning

## Quick Start

```python
from LightningTune import OptunaDrivenOptimizer, SimpleSearchSpace
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

# Define search space
search_space = SimpleSearchSpace({
    "learning_rate": (1e-5, 1e-3, "log"),
    "batch_size": [16, 32, 64],
})

# Create optimizer with direct components
optimizer = OptunaDrivenOptimizer(
    base_config="config.yaml",
    search_space=search_space,
    model_class=MyLightningModule,
    sampler=TPESampler(n_startup_trials=10),
    pruner=HyperbandPruner(min_resource=1, max_resource=100),
    n_trials=50,
)

# Run optimization
study = optimizer.optimize()
print(f"Best params: {study.best_params}")
```

## Common Configurations

### TPE with Hyperband (multi-fidelity optimization)
```python
sampler = TPESampler()
pruner = HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)
```
Note: This is NOT BOHB. Optuna doesn't have BOHB. This is TPE + Hyperband.

### Random Search without Pruning
```python
sampler = RandomSampler(seed=42)
pruner = NopPruner()
```

### CMA-ES for Continuous Optimization
```python
sampler = CmaEsSampler()
pruner = MedianPruner(n_warmup_steps=5)
```

### Successive Halving (NOT ASHA!)
```python
sampler = TPESampler()
pruner = SuccessiveHalvingPruner(min_resource=1, max_resource=100)
```
Note: This is NOT ASHA. Optuna's SHA is synchronous, not asynchronous.

## Command Line Usage

```bash
# TPE with median pruning (default)
python world_model_hpo_optuna.py

# TPE with Hyperband (efficient multi-fidelity optimization)
python world_model_hpo_optuna.py --sampler tpe --pruner hyperband

# Random search without pruning
python world_model_hpo_optuna.py --sampler random --pruner none

# CMA-ES optimization
python world_model_hpo_optuna.py --sampler cmaes --pruner median

# With WandB integration
python world_model_hpo_optuna.py --wandb my-project --n-trials 100
```

## Why This Approach?

1. **Honest** - We don't pretend to have algorithms Optuna doesn't offer
2. **Simple** - No unnecessary abstraction layers
3. **Flexible** - Full control over sampler/pruner parameters
4. **Clear** - Users see exactly what Optuna components are being used
5. **Maintainable** - Less code, fewer abstractions to maintain

## What We DON'T Have

Be aware that Optuna does NOT provide:
- **BOHB** - Bayesian Optimization with HyperBand (specific algorithm)
- **ASHA** - Asynchronous Successive Halving (Optuna's is synchronous)
- **PBT** - Population Based Training
- **HyperBand** - As a standalone strategy (only as a pruner)

## Migration Guide

If you were using the old strategy-based approach:

```python
# Old way
# Note: Use TPESampler with HyperbandPruner for multi-fidelity optimization
strategy = # TPESampler with HyperbandPruner
optimizer = OptunaDrivenOptimizer(..., strategy=strategy)

# New way
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
optimizer = OptunaDrivenOptimizer(
    ...,
    sampler=TPESampler(),
    pruner=HyperbandPruner()
)
```

## Testing

Run the simplified tests:
```bash
pytest tests/test_optuna_simple.py -v
```

## Philosophy

> "Simple is better than complex." - The Zen of Python

We embrace simplicity and honesty. No fake abstractions, no misleading names, just clean integration with Optuna's actual capabilities.