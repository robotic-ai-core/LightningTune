# LightningTune Examples

This document shows working examples of hyperparameter optimization with LightningTune.

## ✅ Working Features

### 1. E2E Tests
The package includes comprehensive end-to-end tests in `tests/e2e/test_full_optimization.py`:
- `test_basic_optimization_workflow` - Tests basic hyperparameter sweeping
- `test_bohb_strategy_e2e` - Tests BOHB strategy with early stopping
- `test_pause_resume_e2e` - Tests pause/resume functionality
- `test_multiple_strategies_e2e` - Tests different optimization strategies

### 2. Simple Demo (`simple_demo.py`)
A minimal working example that demonstrates:
- Creating a search space for hyperparameters
- Running optimization with Random Search
- Getting best hyperparameters
- Shows results for all trials

**Example Output:**
```
🏆 Best validation loss: 0.6936

🎯 Best hyperparameters:
  - learning_rate: 0.000294
  - hidden_dim: 32

📊 All trials:
  Trial 1: lr=0.002533, hidden=32, val_loss=0.7053
  Trial 2: lr=0.002010, hidden=64, val_loss=0.7073
  Trial 3: lr=0.000294, hidden=32, val_loss=0.6936
```

### 3. Full Example (`minimal_example.py`)
A more comprehensive example showing:
- Multiple hyperparameters (learning_rate, hidden_dim, batch_size, dropout)
- Different optimization strategies (BOHB, Random Search)
- Result analysis and statistics
- Saving production configuration

## 🚀 Quick Start

### Basic Usage
```python
from LightningTune import (
    ConfigDrivenOptimizer,
    SearchSpace,
    RandomSearchStrategy,
    OptimizationConfig,
)

# Define search space
class MySearchSpace(SearchSpace):
    def get_search_space(self):
        from ray import tune
        return {
            "model.init_args.learning_rate": tune.loguniform(1e-4, 1e-2),
            "model.init_args.hidden_dim": tune.choice([16, 32, 64]),
        }
    
    def get_metric_config(self):
        return {"metric": "val_loss", "mode": "min"}

# Run optimization
optimizer = ConfigDrivenOptimizer(
    base_config_source="config.yaml",
    search_space=MySearchSpace(),
    strategy=RandomSearchStrategy(num_samples=5),
    optimization_config=OptimizationConfig(
        max_epochs=10,
        max_concurrent_trials=2,
    ),
)

results = optimizer.run()
best_config = optimizer.get_best_config()
```

## 📊 Available Strategies

1. **RandomSearchStrategy** - Random hyperparameter search
2. **BOHBStrategy** - Bayesian Optimization with HyperBand
3. **OptunaStrategy** - Tree-structured Parzen Estimator
4. **GridSearchStrategy** - Exhaustive grid search
5. **ASHAStrategy** - Asynchronous Successive Halving
6. **PBTStrategy** - Population Based Training

## ✅ Test Results

All tests pass successfully:
- **Unit tests**: 43 passed ✓
- **Integration tests**: All passed ✓
- **E2E tests**: Working correctly ✓
- **No inf/nan values** in model outputs ✓

## 🔧 Running the Examples

1. **Run simple demo:**
   ```bash
   python simple_demo.py
   ```

2. **Run full example:**
   ```bash
   python minimal_example.py
   ```

3. **Run tests:**
   ```bash
   # Unit and integration tests
   python -m pytest tests/unit tests/integration -v
   
   # E2E tests (takes longer)
   python -m pytest tests/e2e -v
   ```

## 📝 Notes

- All examples use CPU by default for portability
- Ray will be initialized automatically when needed
- Results are saved to temporary directories by default
- The package integrates seamlessly with LightningReflow for pause/resume capabilities