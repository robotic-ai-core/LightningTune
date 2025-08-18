# Dependency Injection Pattern

## Overview

Lightning BOHB now supports **clean dependency injection** where optimization strategies are configured as independent objects and injected directly into the optimizer. This provides better separation of concerns, type safety, and flexibility.

## Benefits

### 1. **Better Separation of Concerns**
```python
# Strategy configuration is separate from optimizer
strategy = BOHBStrategy(grace_period=10, reduction_factor=3)
optimizer = ConfigDrivenOptimizer(strategy=strategy, ...)
```

### 2. **Type Safety**
```python
# IDEs can provide autocomplete and type checking
strategy = OptunaStrategy(  # IDE knows all parameters
    n_startup_trials=20,
    use_pruner=True,
)
```

### 3. **No String-Based Selection**
```python
# Old way (string-based)
optimizer = ConfigDrivenOptimizer(
    strategy="bohb",  # String - no type checking
    strategy_config={"grace_period": 10}  # Separate config
)

# New way (dependency injection)
strategy = BOHBStrategy(grace_period=10)  # Self-contained
optimizer = ConfigDrivenOptimizer(strategy=strategy)
```

### 4. **Easy Strategy Swapping**
```python
# Easy to compare strategies
for strategy in [RandomSearchStrategy(), OptunaStrategy(), BOHBStrategy()]:
    optimizer = ConfigDrivenOptimizer(strategy=strategy, ...)
    results = optimizer.run()
```

## Usage Examples

### Basic Usage

```python
from lightning_bohb import ConfigDrivenOptimizer
from lightning_bohb.core.strategies_v2 import BOHBStrategy

# 1. Create and configure strategy
strategy = BOHBStrategy(
    grace_period=10,
    reduction_factor=3,
    max_t=100,
    metric="val_loss",
    mode="min",
)

# 2. Inject into optimizer
optimizer = ConfigDrivenOptimizer(
    base_config_source="config.yaml",
    search_space=search_space,
    strategy=strategy,  # Direct injection!
)

# 3. Run optimization
results = optimizer.run()
```

### Different Strategies

#### BOHB
```python
strategy = BOHBStrategy(
    grace_period=10,
    reduction_factor=3,
    max_t=100,
)
```

#### Optuna
```python
strategy = OptunaStrategy(
    n_startup_trials=20,
    use_pruner=True,
    pruner_type="median",
    num_samples=100,
)
```

#### Random Search
```python
strategy = RandomSearchStrategy(
    num_samples=50,
    use_early_stopping=True,
)
```

#### Population Based Training
```python
strategy = PBTStrategy(
    perturbation_interval=10,
    population_size=8,
    hyperparam_mutations={
        "learning_rate": tune.loguniform(1e-5, 1e-2),
    }
)
```

#### Grid Search
```python
strategy = GridSearchStrategy()  # Exhaustive search
```

#### Bayesian Optimization
```python
strategy = BayesianOptimizationStrategy(
    n_initial_points=10,
    acquisition_function="ucb",
)
```

### Custom Strategies

You can easily create custom strategies:

```python
from lightning_bohb.core.strategies_v2 import OptimizationStrategy
from ray.tune.schedulers import ASHAScheduler

class MyCustomStrategy(OptimizationStrategy):
    def __init__(self, aggressive: bool = True):
        self.aggressive = aggressive
        self.metric = "val_loss"
        self.mode = "min"
    
    def get_search_algorithm(self):
        return None  # Random search
    
    def get_scheduler(self):
        if self.aggressive:
            return ASHAScheduler(
                grace_period=3,
                reduction_factor=4,
            )
        return None
    
    def get_num_samples(self):
        return 100
    
    def describe(self):
        return f"MyCustom(aggressive={self.aggressive})"

# Use custom strategy
strategy = MyCustomStrategy(aggressive=True)
optimizer = ConfigDrivenOptimizer(strategy=strategy, ...)
```

### Dynamic Strategy Selection

Select strategies based on available resources:

```python
def select_strategy(time_budget_hrs: float, n_gpus: int):
    """Select best strategy based on resources."""
    
    if time_budget_hrs < 1.0:
        # Quick exploration
        return RandomSearchStrategy(num_samples=20)
    
    elif time_budget_hrs < 4.0:
        # Balanced approach
        return OptunaStrategy(num_samples=50)
    
    elif n_gpus >= 8:
        # Many GPUs available
        return PBTStrategy(population_size=n_gpus * 2)
    
    else:
        # Thorough optimization
        return BOHBStrategy()

# Select and use
strategy = select_strategy(time_budget=2.0, n_gpus=4)
optimizer = ConfigDrivenOptimizer(strategy=strategy, ...)
```

### Strategy Comparison

Compare multiple strategies easily:

```python
strategies = {
    "random": RandomSearchStrategy(num_samples=50),
    "optuna": OptunaStrategy(num_samples=50),
    "bohb": BOHBStrategy(),
}

results = {}
for name, strategy in strategies.items():
    optimizer = ConfigDrivenOptimizer(
        base_config_source="config.yaml",
        search_space=search_space,
        strategy=strategy,
    )
    results[name] = optimizer.run(time_budget_hrs=1.0)
    
# Analyze results
best_strategy = min(results.items(), 
                   key=lambda x: x[1].get_best_result().metrics["val_loss"])
print(f"Best strategy: {best_strategy[0]}")
```

## Testing Strategies

Strategies are self-contained and easy to test:

```python
def test_bohb_strategy():
    strategy = BOHBStrategy(grace_period=10)
    
    # Test configuration
    assert strategy.grace_period == 10
    assert strategy.describe() == "BOHB(grace=10, reduction=3, max_t=100)"
    
    # Test components
    search_alg = strategy.get_search_algorithm()
    assert search_alg is not None
    
    scheduler = strategy.get_scheduler()
    assert scheduler is not None
    
    # Test with mock optimizer
    mock_optimizer = Mock()
    mock_optimizer.strategy = strategy
    # ... test optimization flow
```

## Migration from String-Based API

### Old Way (Still Supported)
```python
optimizer = ConfigDrivenOptimizer(
    base_config_source="config.yaml",
    search_space=search_space,
    strategy="bohb",  # String name
    strategy_config={  # Separate config dict
        "grace_period": 10,
        "reduction_factor": 3,
    }
)
```

### New Way (Recommended)
```python
from lightning_bohb.core.strategies_v2 import BOHBStrategy

strategy = BOHBStrategy(
    grace_period=10,
    reduction_factor=3,
)

optimizer = ConfigDrivenOptimizer(
    base_config_source="config.yaml",
    search_space=search_space,
    strategy=strategy,  # Direct injection
)
```

## Best Practices

1. **Configure strategies at the top level** of your script for clarity
2. **Use type hints** for better IDE support:
   ```python
   def create_optimizer(strategy: OptimizationStrategy) -> ConfigDrivenOptimizer:
       return ConfigDrivenOptimizer(strategy=strategy, ...)
   ```

3. **Create strategy factories** for common configurations:
   ```python
   class StrategyFactory:
       @staticmethod
       def quick_exploration():
           return RandomSearchStrategy(num_samples=20)
       
       @staticmethod
       def thorough_optimization():
           return BOHBStrategy(grace_period=10)
   ```

4. **Document strategy choices** in your code:
   ```python
   # Using BOHB for expensive model training (>30min per trial)
   # with smooth hyperparameter landscape
   strategy = BOHBStrategy(grace_period=10)
   ```

5. **Test strategies independently** before full optimization runs

## Advanced: Strategy Composition

You can compose strategies for complex workflows:

```python
class ProgressiveStrategy:
    """Progressive optimization: Random → Optuna → BOHB."""
    
    def __init__(self):
        self.stages = [
            (RandomSearchStrategy(num_samples=20), 1.0),  # 1 hour
            (OptunaStrategy(num_samples=50), 2.0),        # 2 hours  
            (BOHBStrategy(), 4.0),                        # 4 hours
        ]
    
    def run(self, base_config_source, search_space):
        best_config = None
        
        for strategy, time_budget in self.stages:
            optimizer = ConfigDrivenOptimizer(
                base_config_source=base_config_source,
                search_space=search_space,
                strategy=strategy,
            )
            
            results = optimizer.run(time_budget_hrs=time_budget)
            best_config = optimizer.get_best_config()
            
            # Narrow search space for next stage
            search_space = self.narrow_search_space(search_space, best_config)
        
        return best_config
```

## Summary

The dependency injection pattern provides:
- ✅ **Type safety** - IDEs understand strategy parameters
- ✅ **Testability** - Strategies are self-contained
- ✅ **Flexibility** - Easy to swap and compare strategies
- ✅ **Clarity** - Configuration is explicit and localized
- ✅ **Extensibility** - Custom strategies integrate seamlessly

Use dependency injection for cleaner, more maintainable hyperparameter optimization code!