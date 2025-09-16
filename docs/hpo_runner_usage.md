# HPORunner Usage Guide

## Overview

HPORunner is a high-level class that encapsulates all the CLI handling, checkpoint management, and persistence logic for hyperparameter optimization experiments. It dramatically simplifies HPO scripts by handling all the boilerplate.

## Basic Usage

```python
from LightningTune import HPORunner

# Define your search space
def search_space(trial):
    return {
        "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }

# Create runner
runner = HPORunner(
    model_class=MyModel,
    datamodule_class=MyDataModule,
    search_space=search_space,
    base_config="config.yaml",
    default_study_name="my_study",
)

# Run from CLI - handles everything!
study = runner.run_from_cli()
```

## Features

### Automatic CLI Argument Handling

HPORunner automatically provides these CLI arguments:
- `--n-trials`: Number of trials to run (default: 50)
- `--sampler`: Optuna sampler to use (tpe, random, cmaes, botorch)
- `--pruner`: Optuna pruner to use (median, hyperband, successivehalving, none)
- `--trial-steps`: Max steps per trial
- `--save-every`: Save checkpoint every N trials
- `--wandb`: WandB project name
- `--study-name`: Study name
- `--resume-from`: Resume from checkpoint
- `--experiment-dir`: Directory for results

### Checkpoint Resume with Correct Argument Handling

When resuming from a checkpoint:
- Arguments not specified on CLI are restored from checkpoint
- Arguments explicitly specified override saved values
- Supports extending n_trials (e.g., saved with 50, resume with 100)

Example:
```bash
# Initial run with 50 trials
python my_hpo.py --n-trials 50 --sampler tpe

# Resume without specifying n_trials (continues with 50)
python my_hpo.py --resume-from checkpoint.pkl

# Resume and extend to 100 trials
python my_hpo.py --resume-from checkpoint.pkl --n-trials 100
```

### Configuration Merging

Supports base and override configurations:

```python
runner = HPORunner(
    model_class=MyModel,
    datamodule_class=MyDataModule,
    search_space=search_space,
    base_config="base_config.yaml",
    override_config="override_config.yaml",  # Overrides base
)
```

### Custom CLI Arguments

Add experiment-specific arguments:

```python
runner = HPORunner(
    model_class=MyModel,
    datamodule_class=MyDataModule,
    search_space=search_space,
    base_config="config.yaml",
    additional_cli_args={
        'custom_param': {
            'type': int,
            'default': 42,
            'help': 'My custom parameter'
        }
    }
)
```

### Programmatic Usage

Run without CLI arguments:

```python
study = runner.run(
    n_trials=100,
    sampler='tpe',
    pruner='hyperband',
    wandb='my-project',
)
```

## Migration from Direct Script

Before (700+ lines):
```python
# world_model_hpo_optuna.py
# Lots of argparse boilerplate
# Manual checkpoint handling
# Manual argument restoration logic
# ...700 lines of code...
```

After (~100 lines):
```python
# world_model_hpo_runner.py
from LightningTune import HPORunner

def search_space(trial):
    # Define search space
    pass

runner = HPORunner(
    model_class=WorldModel,
    datamodule_class=LeRobotDataModule,
    search_space=search_space,
    base_config="config.yaml",
)

study = runner.run_from_cli()
```

## Key Benefits

1. **Simplified Code**: ~85% reduction in boilerplate
2. **Robust Resume**: Correctly handles argument persistence and overrides
3. **Checkpoint Compatible**: Works with existing checkpoints
4. **Fully Tested**: Comprehensive test coverage for all scenarios
5. **Clean API**: Intuitive and easy to use

## Implementation Details

HPORunner handles:
- Argument parsing with proper defaults
- Checkpoint loading from file or WandB
- Argument restoration with override detection
- Configuration merging (base + override)
- Study creation and optimization
- Proper cleanup and logging

The implementation correctly detects whether arguments were explicitly specified on the command line, ensuring that:
- Default values don't override saved values when resuming
- Explicit values always take precedence
- n_trials can be extended when resuming