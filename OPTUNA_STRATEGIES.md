# Optuna Strategies in LightningTune

## Available Strategies

LightningTune provides the following **actual Optuna** strategies:

### 1. TPEStrategy
- **What it is**: Tree-structured Parzen Estimator (Optuna's default)
- **Best for**: General hyperparameter optimization
- **Pruner**: MedianPruner or NopPruner

### 2. RandomStrategy  
- **What it is**: Random sampling
- **Best for**: Baseline comparisons, initial exploration
- **Pruner**: NopPruner (no pruning)

### 3. GridStrategy
- **What it is**: Grid search over discrete values
- **Best for**: Small search spaces, exhaustive search
- **Pruner**: NopPruner (no pruning)

### 4. CMAESStrategy
- **What it is**: Covariance Matrix Adaptation Evolution Strategy
- **Best for**: Continuous parameters, smooth objectives
- **Pruner**: MedianPruner

### 5. TPEWithHyperbandStrategy
- **What it is**: TPE sampler + Hyperband pruner
- **NOT BOHB**: This is not BOHB. Optuna doesn't have BOHB.
- **Best for**: Efficient resource allocation with Bayesian optimization

### 6. SuccessiveHalvingStrategy
- **What it is**: Any sampler + SuccessiveHalvingPruner
- **NOT ASHA**: This is not ASHA. Optuna's SHA is synchronous, not asynchronous.
- **Best for**: Aggressive early stopping

## What We DON'T Have

The following are **NOT** available in Optuna and we don't pretend to offer them:

- **BOHB**: Bayesian Optimization with HyperBand (specific algorithm not in Optuna)
- **ASHA**: Asynchronous Successive Halving (Optuna only has synchronous SHA)
- **PBT**: Population Based Training
- **HyperBand**: As a standalone strategy (only as a pruner)

## Migration from Ray Tune

If you were using Ray Tune strategies, here's the mapping:

| Ray Tune Strategy | Optuna Equivalent |
|------------------|-------------------|
| BOHB | TPEWithHyperbandStrategy (similar but not identical) |
| ASHA | SuccessiveHalvingStrategy (synchronous, not async) |
| Random Search | RandomStrategy |
| Grid Search | GridStrategy |
| TPE (via Optuna) | TPEStrategy |
| Population Based Training | Not available |
| BayesOpt | Not available (use TPE instead) |

## Usage Example

```python
from LightningTune.optuna import (
    TPEStrategy,
    TPEWithHyperbandStrategy,
    RandomStrategy,
)

# Standard TPE optimization
strategy = TPEStrategy()

# TPE with Hyperband pruning (closest to BOHB)
strategy = TPEWithHyperbandStrategy(
    min_resource=1,
    max_resource=100,
    reduction_factor=3
)

# Random search baseline
strategy = RandomStrategy()
```

## Important Notes

1. **No BOHB**: Despite what some documentation might say, Optuna does not have BOHB. TPE + Hyperband is similar but not the same algorithm.

2. **No ASHA**: Optuna's SuccessiveHalvingPruner is synchronous, not asynchronous like ASHA.

3. **Samplers vs Pruners**: In Optuna, strategies combine:
   - **Samplers**: How to choose hyperparameters (TPE, Random, Grid, CMA-ES)
   - **Pruners**: When to stop unpromising trials (Median, Hyperband, SHA, None)

4. **Honesty**: We only offer what Optuna actually provides. No made-up algorithms or misleading names.