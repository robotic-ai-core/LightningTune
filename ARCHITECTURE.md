# Lightning BOHB Architecture

## Overview

Lightning BOHB is a **config-driven** hyperparameter optimization submodule that integrates BOHB with PyTorch Lightning projects through LightningReflow.

## Core Design Philosophy

**Config-First Approach**: The entire training pipeline (model, data, trainer) is defined by a YAML configuration file. BOHB optimizes by suggesting parameter overrides that are merged with the base config.

```
Base Config + BOHB Overrides → LightningReflow → Training
```

## Architecture Components

### 1. ConfigDrivenBOHBOptimizer (`core/optimizer_v2.py`)

The main orchestrator that:
- Manages the BOHB optimization loop
- Creates trial configs by merging base + overrides
- Interfaces with Ray Tune for distributed optimization
- Saves results and generates production configs

### 2. SearchSpace (`core/config.py`)

Abstract base class for defining hyperparameter search spaces:
- `get_search_space()`: Returns Ray Tune search space dict
- `get_metric_config()`: Defines optimization metric and mode
- `validate_config()`: Optional validation of parameter combinations
- `transform_config()`: Optional parameter transformations

### 3. Config Management

**Dot Notation for Nested Parameters**:
```python
"model.init_args.learning_rate": tune.loguniform(1e-5, 1e-2)
```

**Hierarchical Merging**:
```yaml
# Base config
model:
  init_args:
    learning_rate: 0.001  # Default
    hidden_dim: 512       # Default

# BOHB override
model.init_args.learning_rate: 0.0005

# Result
model:
  init_args:
    learning_rate: 0.0005  # Overridden
    hidden_dim: 512        # Kept from base
```

### 4. LightningReflow Integration

Each trial:
1. Creates merged config file
2. Initializes LightningReflow with config
3. LightningReflow instantiates model/data/trainer from config
4. Runs training with pause/resume support
5. Reports metrics back to BOHB

### 5. Callbacks

- **BOHBReportCallback**: Reports metrics to Ray Tune
- **AdaptiveBOHBCallback**: Adaptive reporting frequency
- **PauseCallback** (from LightningReflow): Checkpoint management

## Key Benefits

### 1. Zero Coupling
- No imports of model/data classes needed
- Works with ANY Lightning project that uses configs
- Submodule can be extracted to separate package

### 2. Production Ready
- Best config is immediately usable: `python train.py --config best_config.yaml`
- Every trial's exact config is saved
- Automatic checkpoint management

### 3. Simplicity
```python
# Minimal usage
optimizer = ConfigDrivenBOHBOptimizer(
    base_config_source="config.yaml",
    search_space={"model.init_args.lr": tune.loguniform(1e-5, 1e-2)},
)
results = optimizer.run()
```

## File Structure

```
lightning_bohb/
├── core/
│   ├── optimizer_v2.py      # Main config-driven optimizer
│   ├── config.py            # SearchSpace ABC, ConfigManager
│   ├── optimizer.py         # Legacy class-based optimizer
│   └── trainable.py         # Legacy trainable classes
├── callbacks/
│   └── report.py            # BOHB reporting callbacks
├── examples/
│   ├── minimal_example.py   # Simplest usage
│   └── world_model_optimization.py  # Full example
└── README.md               # User documentation
```

## Usage Pattern

### For Your World Model

```python
from lightning_bohb import ConfigDrivenBOHBOptimizer, SearchSpace

class WorldModelSearchSpace(SearchSpace):
    def get_search_space(self):
        return {
            "model.init_args.no_op_regularizer_weight": tune.loguniform(0.01, 2.0),
            # ... other params
        }

optimizer = ConfigDrivenBOHBOptimizer(
    base_config_source="scripts/world_model_config.yaml",
    search_space=WorldModelSearchSpace(),
)
results = optimizer.run()
production_config = optimizer.create_production_config()
```

## Design Decisions

1. **LightningReflow as Primary**: Since your repos are config-driven, LightningReflow is the default
2. **Dot Notation**: Natural way to specify nested config overrides
3. **SearchSpace ABC**: Encourages reusable, testable search space definitions
4. **Config Merging**: Preserves all base config settings not being tuned
5. **Trial Isolation**: Each trial gets its own directory with config, checkpoints, logs

## Future Extensions

- Monitoring dashboard (started in `monitoring/`)
- Multi-objective optimization support
- Conditional parameter spaces
- Integration with other HPO algorithms (Optuna, etc.)

## Dependencies

- **Required**: ray[tune], lightning-reflow, pytorch-lightning
- **Optional**: streamlit, plotly (for monitoring)

This architecture ensures the submodule remains completely independent of your specific model implementations while providing powerful hyperparameter optimization capabilities.