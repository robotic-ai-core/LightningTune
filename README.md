# LightningTune

Config-driven hyperparameter optimization for PyTorch Lightning with Ray Tune, supporting multiple strategies (BOHB, Optuna, Random Search, etc.) and interactive pause/resume capabilities.

## What It Does

LightningTune **automates hyperparameter optimization** for your PyTorch Lightning models. Instead of manually trying different learning rates, batch sizes, or architecture choices, it systematically explores the parameter space to find the best configuration for your model.

### Core Capabilities:
- **Automated Search**: Explores hyperparameter combinations intelligently
- **Smart Early Stopping**: Kills bad trials early using algorithms like BOHB (Bayesian Optimization with HyperBand)
- **Parallel Trials**: Runs multiple experiments simultaneously to maximize GPU/CPU utilization
- **Pause & Resume**: Stop optimization anytime and continue later, even on a different machine
- **Config-Based**: Works with your existing Lightning CLI configuration files - no code changes needed

## Why Use LightningTune?

### Manual Hyperparameter Tuning (Without LightningTune):
```python
# Monday: Try learning rate 0.001
python train.py --lr 0.001  # val_loss: 0.45

# Tuesday: Try learning rate 0.0001
python train.py --lr 0.0001  # val_loss: 0.52

# Wednesday: Try batch size 64 with lr 0.001
python train.py --lr 0.001 --batch_size 64  # val_loss: 0.41

# Thursday: Maybe dropout helps?
python train.py --lr 0.001 --batch_size 64 --dropout 0.3  # val_loss: 0.43

# ... weeks later, still guessing ...
```

### Automated Optimization (With LightningTune):
```python
# Define what to search
search_space = {
    "model.init_args.learning_rate": tune.loguniform(1e-5, 1e-1),
    "data.init_args.batch_size": tune.choice([16, 32, 64, 128]),
    "model.init_args.dropout": tune.uniform(0.0, 0.5),
}

# Run optimization - tries 100s of combinations intelligently
optimizer.run()  # Finds best: lr=0.00089, batch_size=48, dropout=0.23
```

### Benefits:
- ⏱️ **Saves Time**: Days of manual tuning → Hours of automated search
- 💰 **Resource Efficient**: Kills bad trials early, focuses compute on promising ones
- 📊 **Systematic**: Explores the space methodically, not randomly
- 🎯 **Better Results**: Often finds configurations you wouldn't have tried manually
- 📝 **Reproducible**: Every trial's config is saved for perfect reproducibility

## Features

- 🎯 **Multiple Optimization Strategies**: BOHB, Optuna, Random Search, PBT, and more
- ⏸️ **Interactive Pause/Resume**: Press 'p' to pause optimization, resume later with one command
- 💉 **Dependency Injection**: Clean strategy pattern for easy customization
- 📦 **Config-Driven**: Works with Lightning CLI configuration files or Python dictionaries
- 💾 **State Preservation**: Complete session state saved for perfect resume
- 🔄 **Zero Data Loss**: All progress preserved across pause/resume cycles

## Installation

```bash
# Required dependencies
pip install ray[tune] lightning-reflow pytorch-lightning

# Optional for additional strategies
pip install optuna hyperopt

# Optional for monitoring
pip install streamlit plotly pandas
```

## Quick Start

### 1. Basic Usage

```python
from LightningTune import ConfigDrivenOptimizer, SearchSpace
from lightning_bohb.core.strategies_v2 import BOHBStrategy
from ray import tune

# Step 1: Define what hyperparameters to optimize
class MySearchSpace(SearchSpace):
    def get_search_space(self):
        return {
            # Model hyperparameters
            "model.init_args.learning_rate": tune.loguniform(1e-5, 1e-2),
            "model.init_args.hidden_dim": tune.choice([256, 512, 1024]),
            "model.init_args.dropout": tune.uniform(0.0, 0.5),
            
            # Training parameters
            "data.init_args.batch_size": tune.choice([16, 32, 64]),
        }
    
    def get_metric_config(self):
        return {"metric": "val_loss", "mode": "min"}  # Minimize validation loss

# Step 2: Choose optimization strategy
strategy = BOHBStrategy(
    grace_period=10,      # Min epochs before pruning bad trials
    reduction_factor=3    # Aggressive pruning of bad trials
)

# Step 3: Run optimization
optimizer = ConfigDrivenOptimizer(
    base_config_source="configs/train_config.yaml",  # Your existing config
    search_space=MySearchSpace(),
    strategy=strategy,
)

results = optimizer.run()

# Step 4: Get the best configuration
best_config = optimizer.create_production_config()
print(f"Best config saved to: {best_config}")
```

### 2. With Interactive Pause/Resume

```python
from LightningTune import ConfigDrivenOptimizer, TuneReflowCLI
from lightning_bohb.core.strategies_v2 import BOHBStrategy

# Wrap with CLI for pause/resume capabilities
cli = TuneReflowCLI(experiment_name="my_model")

strategy = BOHBStrategy(grace_period=10)
optimizer = ConfigDrivenOptimizer(
    base_config_source="config.yaml",
    search_space=MySearchSpace(),
    strategy=strategy,
)

# Run with interactive controls
# Press 'p' to pause, Ctrl+C for graceful stop
results = cli.run(optimizer)

# After pausing, resume with:
# python script.py --resume-session ./experiments/my_model/session_xxx
```

### 3. Interactive Workflow Example

```bash
# Start optimization
$ python optimize.py --experiment-name my_model --base-config config.yaml

🎮 Interactive mode enabled:
   Press 'p' to pause at next validation
   Press Ctrl+C to pause gracefully

Running 4 concurrent trials...
Trial_0: Epoch 45/100 [████████--] val_loss: 0.234  # Good trial, keep running
Trial_1: Epoch 10/100 [██--------] val_loss: 0.891  # Bad trial, will be pruned
Trial_2: Epoch 23/100 [████------] val_loss: 0.456  # Mediocre, might be pruned
Trial_3: Epoch 67/100 [████████--] val_loss: 0.198  # Best so far!

# Press 'p' to pause
⏸️ Pause requested. Trials will pause at next validation...

✅ OPTIMIZATION PAUSED
📝 To resume this EXACT session with all settings, run:
   python optimize.py --resume-session ./experiments/my_model/session_20240101_120000_abc123

# Resume later (even on different machine)
$ python optimize.py --resume-session ./experiments/my_model/session_20240101_120000_abc123

♾️ Resuming session: 20240101_120000_abc123
✅ All configuration restored from session state
Continuing optimization from exact pause point...
```

## Real-World Example: Optimizing a Vision Transformer

```python
from LightningTune import ConfigDrivenOptimizer, BOHBConfig, SearchSpace
from ray import tune

class VisionTransformerSearchSpace(SearchSpace):
    def get_search_space(self):
        return {
            # Architecture tuning
            "model.init_args.num_layers": tune.choice([6, 8, 12, 16]),
            "model.init_args.num_heads": tune.choice([4, 8, 12, 16]),
            "model.init_args.hidden_dim": tune.choice([384, 512, 768]),
            "model.init_args.mlp_ratio": tune.choice([2.0, 3.0, 4.0]),
            
            # Regularization
            "model.init_args.dropout": tune.uniform(0.0, 0.3),
            "model.init_args.attention_dropout": tune.uniform(0.0, 0.3),
            "model.init_args.drop_path_rate": tune.uniform(0.0, 0.2),
            
            # Training hyperparameters
            "model.init_args.learning_rate": tune.loguniform(1e-5, 1e-2),
            "model.init_args.weight_decay": tune.loguniform(1e-6, 1e-2),
            "model.init_args.warmup_epochs": tune.choice([5, 10, 20]),
            
            # Data parameters
            "data.init_args.batch_size": tune.choice([32, 64, 128]),
            "data.init_args.mixup_alpha": tune.uniform(0.0, 1.0),
            "data.init_args.cutmix_alpha": tune.uniform(0.0, 1.0),
        }
    
    def get_metric_config(self):
        return {"metric": "val_accuracy", "mode": "max"}

# Run optimization with time budget
optimizer = ConfigDrivenOptimizer(
    base_config_source="configs/vit_base_config.yaml",
    search_space=VisionTransformerSearchSpace(),
    bohb_config=BOHBConfig(
        max_epochs=100,
        grace_period=20,          # Let models train for 20 epochs before pruning
        max_concurrent_trials=8,  # Use 8 GPUs in parallel
        time_budget_hrs=48,       # Stop after 48 hours
        experiment_name="vit_optimization"
    )
)

results = optimizer.run()

# Analyze results
print(f"Best validation accuracy: {results.best_result['val_accuracy']:.4f}")
print(f"Best hyperparameters: {results.best_config}")
print(f"Tried {results.num_trials} different configurations")
```

## How It Works

### 1. Your Base Config Defines the Training Pipeline

```yaml
# configs/train_config.yaml
model:
  class_path: my_project.models.MyModel
  init_args:
    learning_rate: 0.001  # Default value
    hidden_dim: 512        # Default value
    num_layers: 4          # Fixed value

data:
  class_path: my_project.data.MyDataModule
  init_args:
    batch_size: 32         # Default value
    data_dir: ./data       # Fixed value

trainer:
  max_epochs: 100
  gpus: 1
```

### 2. Ray Tune Suggests Parameter Overrides

For each trial, the optimization algorithm suggests different values:
```python
# Trial 1: Try smaller learning rate, bigger model
{"model.init_args.learning_rate": 0.0005, "model.init_args.hidden_dim": 1024}

# Trial 2: Try larger learning rate, smaller batch
{"model.init_args.learning_rate": 0.005, "data.init_args.batch_size": 16}
```

### 3. Configs Are Merged and Training Runs

The optimizer merges suggestions with your base config and runs training. Results are tracked and used to inform future trials.

### 4. Best Configuration Is Saved

After optimization, you get a production-ready config with the best hyperparameters.

## Common Use Cases

### 1. Finding Optimal Learning Rate and Batch Size
```python
search_space = {
    "model.init_args.learning_rate": tune.loguniform(1e-6, 1e-1),
    "data.init_args.batch_size": tune.choice([8, 16, 32, 64, 128]),
    "trainer.accumulate_grad_batches": tune.choice([1, 2, 4, 8]),
}
```

### 2. Architecture Search
```python
search_space = {
    "model.init_args.num_layers": tune.choice(range(4, 25)),
    "model.init_args.hidden_dim": tune.choice([256, 384, 512, 768, 1024]),
    "model.init_args.num_heads": tune.choice([4, 8, 12, 16]),
}
```

### 3. Regularization Tuning
```python
search_space = {
    "model.init_args.dropout": tune.uniform(0.0, 0.5),
    "model.init_args.weight_decay": tune.loguniform(1e-6, 1e-1),
    "data.init_args.augmentation_strength": tune.uniform(0.0, 1.0),
}
```

## Optimization Strategies

### BOHB (Recommended for Most Cases)
Combines Bayesian optimization with early stopping. Best for:
- Limited compute budget
- When you need good results quickly
- Models that show early indicators of performance

```python
strategy = BOHBStrategy(
    grace_period=10,      # Min epochs before pruning
    reduction_factor=3,   # How aggressively to prune
)
```

### Optuna (Good for Complex Search Spaces)
Tree-structured Parzen Estimator. Best for:
- Complex, high-dimensional search spaces
- When you have domain knowledge to add

```python
strategy = OptunaStrategy(
    n_startup_trials=10,  # Random trials before optimization
    n_ei_candidates=24,   # Candidates for acquisition function
)
```

### Random Search (Baseline)
Simple but surprisingly effective. Best for:
- Initial exploration
- When you have lots of parallel resources
- As a baseline to compare against

```python
strategy = RandomSearchStrategy(
    max_concurrent_trials=16,  # Run many in parallel
)
```

## Tips for Effective Hyperparameter Optimization

1. **Start Small**: Begin with 3-5 key hyperparameters, then expand
2. **Set Reasonable Ranges**: Too wide = wasted compute, too narrow = might miss optimum
3. **Use Grace Period**: Set to 10-20% of max_epochs to avoid premature pruning
4. **Monitor Progress**: Use Ray Tune dashboard or TensorBoard
5. **Save Everything**: All trial configs are saved - you might find the 2nd best is more stable
6. **Use Time Budgets**: Set `time_budget_hrs` to ensure completion within constraints
7. **Leverage Domain Knowledge**: Add constraints in `validate_config()` method

## Analyzing Results

```python
# After optimization completes
analysis = optimizer.analyze_results()

# Get detailed information
print(f"Best {analysis['metric_name']}: {analysis['best_metric_value']:.4f}")
print(f"Total trials: {analysis['total_trials']}")
print(f"Successful trials: {analysis['successful_trials']}")
print(f"Average trial time: {analysis['avg_trial_time_hrs']:.1f} hours")

# Parameter importance (which parameters mattered most)
for param, importance in analysis['parameter_importance'].items():
    print(f"{param}: {importance:.2%} importance")

# Get configs
best_config = optimizer.get_best_config()           # Just the overrides
complete_config = optimizer.get_best_complete_config()  # Full config

# Create production config
production_path = optimizer.create_production_config("configs/production.yaml")
```

## Requirements

- PyTorch Lightning
- LightningReflow
- Ray[tune]
- Your training pipeline must be fully definable via config (Lightning CLI pattern)

## Troubleshooting

### Common Issues

**Issue**: Trials being pruned too early
**Solution**: Increase `grace_period` to give models more time to converge

**Issue**: Not finding good hyperparameters
**Solution**: Expand search ranges or add more parameters to search

**Issue**: Running out of memory
**Solution**: Reduce `max_concurrent_trials` or use smaller batch sizes

**Issue**: Optimization taking too long
**Solution**: Use more aggressive pruning (lower `reduction_factor`) or set `time_budget_hrs`

## FAQ

**Q: How many trials should I run?**
A: Typically 50-200 trials gives good results. BOHB is efficient, so fewer trials are needed than random search.

**Q: What if my model requires complex initialization?**
A: Keep using your existing config - as long as LightningReflow can instantiate from it, it will work.

**Q: Can I tune parameters not in the config?**
A: All tuned parameters must be configurable via the config file. Consider adding them to your model's `__init__`.

**Q: How does this compare to Weights & Biases Sweeps?**
A: LightningTune offers more advanced optimization algorithms (BOHB), better parallelization, and pause/resume capabilities.

**Q: Can I use this with distributed training?**
A: Yes! Each trial can use distributed training. Set resources appropriately in `resources_per_trial`.

## Contributing

Contributions are welcome! Please check the issues page or submit a PR.

## License

[Add your license here]