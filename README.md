# LightningTune

Optuna-based hyperparameter optimization for PyTorch Lightning. Minimal, direct use of Optuna samplers/pruners with Lightning models.

## Features

- **Simple API** - Define search space as a function, get optimized hyperparameters
- **Pause/Resume** - Press 'p' to pause, resume from local or WandB checkpoints
- **Auto-Restart** - Process restarts after each checkpoint to prevent memory leaks
- **WandB Integration** - Automatic logging and checkpoint storage
- **HPORunner** - High-level CLI wrapper with argument persistence

## TL;DR (Quickstart)

```bash
pip install -e .
```

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

def search_space(trial):
    return {
        "model.learning_rate": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    }

runner = HPORunner(
    model_class=YourModel,
    datamodule_class=YourDataModule,
    search_space=search_space,
    base_config="config.yaml",
)

study = runner.run_from_cli()
```

```bash
# Start optimization
python my_hpo.py --n-trials 100 --wandb my-project --study-name exp1

# Resume from local checkpoint (most up-to-date)
python my_hpo.py --resume-from local

# Resume from WandB (for cross-machine workflows)
python my_hpo.py --resume-from latest --wandb my-project --study-name exp1
```

## Resume Options

| `--resume-from` | Source | Use Case |
|-----------------|--------|----------|
| `local` | Local filesystem | After crash/pause (most trials) |
| `latest` | WandB artifact | Cross-machine, collaboration |
| `vN` | WandB version | Specific checkpoint (e.g., `v5`) |
| `/path/file.pkl` | Explicit path | Custom checkpoint location |

**Local vs WandB:** Local checkpoints save after every trial. WandB uploads periodically (`--upload-every`). Use `local` for most up-to-date state.

## Auto-Restart

Enabled by default. Process exits with code 42 after checkpoint save, allowing clean memory reclamation between trials.

- `--debug-no-restart` - Disable for debugging

## Notes

- Pass `datamodule_class=YourDataModule` if you use a LightningDataModule
- For W&B logging, add `wandb_project="my-project"`
- Use `PausibleOptunaOptimizer` for programmatic pause/resume
- See [docs/hpo_runner_usage.md](docs/hpo_runner_usage.md) for full HPORunner guide

