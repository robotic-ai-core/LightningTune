# LightningTune

Optuna-based hyperparameter optimization for PyTorch Lightning. Minimal, direct use of Optuna samplers/pruners with Lightning models.

## Features

- **Simple API** - Define search space as a function, get optimized hyperparameters
- **Pause/Resume** - Press 'p' to pause, resume from local or WandB checkpoints
- **Auto-Restart** - Process restarts after each checkpoint to prevent memory leaks
- **WandB Integration** - Automatic logging and checkpoint storage
- **HPORunner** - High-level CLI wrapper with argument persistence
- **Config Modification** - Search space modifies nested Lightning CLI configs

## Installation

```bash
pip install -e .
```

---

## TL;DR (Quickstart)

### Option 1: Direct Optimizer

```python
from LightningTune import OptunaDrivenOptimizer, TPESampler, MedianPruner

def search_space(trial):
    return {
        "model.learning_rate": trial.suggest_float("model.learning_rate", 1e-4, 1e-2, log=True),
        "trainer.max_epochs": 5,
    }

optimizer = OptunaDrivenOptimizer(
    base_config="config.yaml",
    search_space=search_space,
    model_class=YourLightningModule,
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(),
    n_trials=20,
    metric="val_loss",
)

study = optimizer.optimize()
print(optimizer.get_best_config())
```

### Option 2: HPORunner (Recommended for CLI)

```python
from LightningTune import HPORunner

def search_space(trial, config):
    # Modify config in-place and return it
    config["model"]["init_args"]["learning_rate"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    return config

runner = HPORunner(
    model_class=YourModel,
    datamodule_class=YourDataModule,
    search_space=search_space,
    require_config=True,
)

study = runner.run_from_cli()
```

```bash
# Start optimization
python my_hpo.py --config config.yaml --n-trials 100 --wandb my-project --study-name exp1

# Resume from local checkpoint (most up-to-date)
python my_hpo.py --resume-from local

# Resume from WandB (for cross-machine workflows)
python my_hpo.py --resume-from latest --wandb my-project --study-name exp1
```

---

## Search Space Patterns

### Basic Pattern: Flat Config Modification

```python
def search_space(trial, config):
    """Simple search space that modifies top-level config keys."""
    config["model"]["init_args"]["learning_rate"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    config["model"]["init_args"]["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
    config["data"]["init_args"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    return config
```

### Advanced Pattern: Nested Config with class_path

For Lightning CLI configs with nested `class_path` / `init_args` structures:

```python
def search_space(trial, config):
    """Search space with nested module configurations."""

    # Sample architectural choices
    architecture = trial.suggest_categorical("architecture", ["small", "medium", "large"])
    num_layers = trial.suggest_int("num_layers", 4, 16, step=2)
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 384, 512])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Build nested config for dynamics model
    inner_model_config = {
        "class_path": "myproject.models.TransformerModel",
        "init_args": {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }
    }

    # Wrap in adapter if needed
    adapter_config = {
        "class_path": "myproject.models.AdapterModel",
        "init_args": {
            "wrapped_model": inner_model_config,
            "residual": trial.suggest_categorical("residual", [True, False]),
        }
    }

    # Apply to config
    config["model"]["init_args"]["dynamics_model"] = adapter_config

    return config
```

### Pattern: Conditional Sampling

```python
def search_space(trial, config):
    """Search space with conditional parameter sampling."""

    # Architecture determines which parameters to sample
    arch = trial.suggest_categorical("architecture", ["transformer", "mlp", "hybrid"])

    if arch == "transformer":
        num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])
        num_layers = trial.suggest_int("num_layers", 2, 12)
        config["model"]["init_args"]["transformer_config"] = {
            "num_heads": num_heads,
            "num_layers": num_layers,
        }
    elif arch == "mlp":
        hidden_dims = trial.suggest_categorical("hidden_dims", [[256, 256], [512, 256], [512, 512]])
        config["model"]["init_args"]["mlp_dims"] = hidden_dims
    else:  # hybrid
        # Both transformer and MLP params
        pass

    return config
```

### Pattern: Data Augmentation Search

```python
def search_space(trial, config):
    """Search over data augmentation parameters."""

    # Augmentation probabilities (0.0 = disabled)
    temporal_prob = trial.suggest_float("temporal_prob", 0.0, 0.8)
    mirror_prob = trial.suggest_float("mirror_prob", 0.0, 0.5)
    rotation_prob = trial.suggest_float("rotation_prob", 0.0, 0.5)

    # Augmentation magnitudes (only if enabled)
    augmentation_config = {}

    if temporal_prob > 0:
        augmentation_config["temporal"] = {
            "enabled": True,
            "p": temporal_prob,
            "min_scale": 0.8,
            "max_scale": trial.suggest_float("temporal_scale", 1.1, 1.3),
        }

    if mirror_prob > 0:
        augmentation_config["mirror"] = {
            "enabled": True,
            "p": mirror_prob,
        }

    if rotation_prob > 0:
        augmentation_config["rotation"] = {
            "enabled": True,
            "p": rotation_prob,
            "max_angle": trial.suggest_float("rotation_angle", 5.0, 30.0),
        }

    config["data"]["init_args"]["augmentation"] = augmentation_config
    return config
```

---

## HPORunner Configuration

### Constructor Parameters

```python
runner = HPORunner(
    model_class=YourModel,                    # LightningModule class
    datamodule_class=YourDataModule,          # Optional LightningDataModule class
    search_space=search_space_fn,             # Function(trial, config) -> config
    base_config="config.yaml",                # Optional default config path
    require_config=True,                      # Require --config CLI argument
    default_study_name="my_hpo",              # Default Optuna study name
)
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Base YAML configuration file | Required if `require_config=True` |
| `--n-trials` | Number of trials to run | 100 |
| `--study-name` | Optuna study name | `default_study_name` |
| `--wandb` | W&B project name | None |
| `--sampler` | Optuna sampler (`tpe`, `random`, `cmaes`) | `tpe` |
| `--pruner` | Optuna pruner (`median`, `hyperband`, `none`) | `median` |
| `--trial-steps` | Max steps per trial | None (full training) |
| `--save-every` | Save checkpoint every N trials | 5 |
| `--upload-every` | Upload to W&B every N trials | 10 |
| `--resume-from` | Resume source (see below) | None |
| `--debug-no-restart` | Disable auto-restart | False |
| `--test-mode` | Enable test mode (fast trials) | False |

---

## Resume Options

| `--resume-from` | Source | Use Case |
|-----------------|--------|----------|
| `local` | Local filesystem | After crash/pause (most trials) |
| `latest` | WandB artifact | Cross-machine, collaboration |
| `vN` | WandB version | Specific checkpoint (e.g., `v5`) |
| `/path/file.pkl` | Explicit path | Custom checkpoint location |

**Local vs WandB:** Local checkpoints save after every trial. WandB uploads periodically (`--upload-every`). Use `local` for most up-to-date state.

---

## Auto-Restart

Enabled by default. Process exits with code 42 after checkpoint save, allowing clean memory reclamation between trials.

```bash
# Disable for debugging
python my_hpo.py --debug-no-restart
```

**Why Auto-Restart?**
- Prevents GPU memory accumulation across trials
- Cleans up orphaned CUDA tensors
- Ensures consistent memory baseline for each trial

---

## Result Reporting

### Programmatic Access

```python
# Get best trial
best_trial = study.best_trial
print(f"Best value: {best_trial.value}")
print(f"Best params: {best_trial.params}")

# Generate training command for best config
runner.generate_best_config_command(
    study,
    script="python scripts/train.py fit",
    extra_args={"trainer.max_epochs": 2000},
    excluded_params={"data.batch_size"},
)

# Format results summary
print(runner.format_results(study))
```

### CLI Command Generation

```python
from LightningTune.utils.cli_generation import extract_cli_args_from_config, format_cli_command

# Extract CLI arguments from config
cli_args = extract_cli_args_from_config(config, base_config_path="config.yaml")

# Format as command
command = format_cli_command(
    script="python train.py fit",
    config_path="config.yaml",
    cli_args=cli_args,
)
print(command)
# python train.py fit --config config.yaml --model.init_args.learning_rate 0.001 ...
```

---

## Complete Example: Multi-Architecture HPO

```python
# scripts/model_hpo.py
from LightningTune import HPORunner
from myproject.models import WorldModel
from myproject.data import MyDataModule

ARCHITECTURE_CHOICES = ["transformer", "mlp", "hybrid"]

def search_space(trial, config):
    """Multi-architecture search space."""

    # Architecture selection
    arch = trial.suggest_categorical("architecture", ARCHITECTURE_CHOICES)

    # Common parameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 384, 512])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Architecture-specific parameters
    if arch == "transformer":
        model_config = {
            "class_path": "myproject.models.TransformerModel",
            "init_args": {
                "hidden_dim": hidden_dim,
                "num_heads": trial.suggest_categorical("num_heads", [4, 8, 16]),
                "num_layers": trial.suggest_int("num_layers", 4, 12, step=2),
                "dropout": dropout,
            }
        }
    elif arch == "mlp":
        model_config = {
            "class_path": "myproject.models.MLPModel",
            "init_args": {
                "hidden_dim": hidden_dim,
                "num_layers": trial.suggest_int("mlp_layers", 2, 6),
                "dropout": dropout,
            }
        }
    else:
        model_config = {
            "class_path": "myproject.models.HybridModel",
            "init_args": {
                "hidden_dim": hidden_dim,
                "dropout": dropout,
            }
        }

    # Apply to config
    config["model"]["init_args"]["backbone"] = model_config

    # Data parameters
    config["data"]["init_args"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])

    return config


def main():
    runner = HPORunner(
        model_class=WorldModel,
        datamodule_class=MyDataModule,
        search_space=search_space,
        require_config=True,
        default_study_name="multi_arch_hpo",
    )

    study = runner.run_from_cli()

    # Print results
    print("\n" + runner.format_results(study))

    # Generate command for best trial
    runner.generate_best_config_command(
        study,
        script="python scripts/train.py fit",
        extra_args={"trainer.max_epochs": 2000},
    )


if __name__ == "__main__":
    main()
```

```bash
# Run HPO
python scripts/model_hpo.py --config configs/model.yaml --n-trials 100 --wandb my-project

# Resume after pause
python scripts/model_hpo.py --resume-from local

# Resume on different machine
python scripts/model_hpo.py --resume-from latest --wandb my-project --study-name multi_arch_hpo
```

---

## Integration with LightningReflow

LightningTune works seamlessly with LightningReflow for pause/resume during individual trials:

```yaml
# config.yaml
trainer:
  callbacks:
    - class_path: lightning_reflow.callbacks.PauseCallback
      init_args:
        enable_pause: false  # Disable during HPO (HPO handles pausing)
```

For debugging utilities, LightningTune imports from LightningReflow:

```python
from LightningTune.utils import (
    CrashResistantLogger,
    ThreadMonitor,
    setup_crash_resistant_logging,
)
```

---

## Notes

- Pass `datamodule_class=YourDataModule` if you use a LightningDataModule
- For W&B logging, add `--wandb my-project`
- Use `PausibleOptunaOptimizer` for programmatic pause/resume
- Search space function receives `(trial, config)` and must return modified config
- Nested `class_path` / `init_args` configs are fully supported
