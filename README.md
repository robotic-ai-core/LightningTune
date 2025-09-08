# LightningTune

Optuna-based hyperparameter optimization for PyTorch Lightning. Minimal, direct use of Optuna samplers/pruners with Lightning models.

## TL;DR (Quickstart)

```bash
pip install -e .
```

```python
from LightningTune import OptunaDrivenOptimizer, TPESampler, MedianPruner

# Define your Optuna search space (function or LightningTune search space object)
def search_space(trial):
    return {
        "model.learning_rate": trial.suggest_float("model.learning_rate", 1e-4, 1e-2, log=True),
        "trainer.max_epochs": 5,
    }

optimizer = OptunaDrivenOptimizer(
    base_config="config.yaml",        # Lightning-style YAML or dict
    search_space=search_space,         # function or LightningTune OptunaSearchSpace
    model_class=YourLightningModule,   # your pl.LightningModule class
    sampler=TPESampler(seed=42),       # any Optuna sampler (optional)
    pruner=MedianPruner(),             # any Optuna pruner (optional)
    n_trials=20,
    metric="val_loss",
)

study = optimizer.optimize()
best_config = optimizer.get_best_config()
print(best_config)
```

## Notes

- Pass `datamodule_class=YourDataModule` if you use a LightningDataModule.
- For W&B logging, add `wandb_project="my-project"` (checkpoints upload optional via `upload_checkpoints`).
- You can also use `PausibleOptunaOptimizer` for pause/resume workflows.

