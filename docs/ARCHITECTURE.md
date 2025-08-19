# LightningTune Architecture

## Design Philosophy

LightningTune follows a **direct dependency injection** approach, avoiding unnecessary abstractions while providing a clean interface for hyperparameter optimization.

### Core Principles

1. **Simplicity Over Abstraction**
   - No unnecessary wrapper classes or strategy patterns
   - Direct use of well-documented Optuna components
   - Clear, predictable behavior

2. **Honest Implementation**
   - We don't claim to implement algorithms we don't have (no fake BOHB/ASHA)
   - Clear documentation about what each component actually does
   - Transparent about limitations

3. **Composition Over Inheritance**
   - Use dependency injection for flexibility
   - Prefer composition of simple components
   - Avoid deep inheritance hierarchies

4. **Testing-Driven Reliability**
   - Comprehensive test coverage at multiple levels
   - End-to-end tests on real datasets
   - Statistical validation of optimization effectiveness

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Code                            │
├─────────────────────────────────────────────────────────────┤
│                    LightningTune API                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            OptunaDrivenOptimizer                    │   │
│  │  ┌────────────┐  ┌────────────┐  ┌──────────────┐ │   │
│  │  │   Config   │  │   Search   │  │   Training   │ │   │
│  │  │  Manager   │  │   Space    │  │   Manager    │ │   │
│  │  └────────────┘  └────────────┘  └──────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    External Libraries                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │    Optuna    │  │  PyTorch     │  │    YAML/     │     │
│  │  (Samplers,  │  │  Lightning   │  │    JSON      │     │
│  │   Pruners)   │  │              │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### OptunaDrivenOptimizer

The central orchestrator that manages the optimization process.

**Responsibilities:**
- Creates and manages Optuna studies
- Handles trial execution
- Merges configurations
- Manages checkpoints
- Reports results

**Key Design Decisions:**
- Accepts Optuna components directly (no wrappers)
- Stateless between trials (all state in Optuna study)
- Configurable via constructor parameters

### Search Spaces

Two implementations for different complexity levels:

#### SimpleSearchSpace
- For straightforward parameter definitions
- Maps parameter names to distribution tuples
- Handles nested parameter paths (e.g., "model.lr")

#### AdvancedSearchSpace
- For complex, conditional parameters
- Accepts a callable that defines the search logic
- Enables parameter dependencies

### Configuration Management

**Merge Strategy:**
1. Load base configuration (YAML/JSON/dict)
2. Apply trial suggestions as overrides
3. Deep merge preserving structure
4. Validate final configuration

**Key Features:**
- Supports nested configurations
- Handles Lightning CLI format
- Preserves non-tuned parameters
- Type-safe merging

### Training Integration

**Lightning Integration:**
- Uses standard Lightning Trainer
- Supports all Lightning callbacks
- Compatible with any LightningModule
- Optional DataModule support

**Pruning Integration:**
- OptunaPruningCallback for early stopping
- Reports metrics at each validation
- Handles pruned trials gracefully

## Data Flow

### Optimization Pipeline

```
1. User defines search space and base config
   ↓
2. OptunaDrivenOptimizer creates Optuna study
   ↓
3. For each trial:
   a. Optuna suggests parameters
   b. Merge with base config
   c. Instantiate model/data
   d. Train with Lightning
   e. Report metrics to Optuna
   f. Check for pruning
   ↓
4. Return study with best parameters
```

### Trial Execution

```python
def objective(trial):
    # 1. Get parameter suggestions
    params = search_space.suggest_params(trial)
    
    # 2. Merge with base config
    config = merge_configs(base_config, params)
    
    # 3. Create model and data
    model = instantiate_from_config(config["model"])
    datamodule = instantiate_from_config(config["data"])
    
    # 4. Setup trainer with pruning callback
    trainer = Trainer(
        **config["trainer"],
        callbacks=[OptunaPruningCallback(trial, metric)]
    )
    
    # 5. Train and return metric
    trainer.fit(model, datamodule)
    return trainer.callback_metrics[metric]
```

## Testing Architecture

### Test Levels

1. **Unit Tests** (`test_optuna_simple.py`)
   - Test individual components
   - Mock external dependencies
   - Fast execution

2. **Integration Tests** (`test_optuna_integration.py`)
   - Test component interactions
   - Use real Optuna/Lightning
   - Medium execution time

3. **End-to-End Tests** (`test_e2e_fashion_mnist.py`)
   - Complete optimization on real data
   - Verify actual improvement
   - Statistical validation
   - Slow execution

### Verification Strategy

```python
# 1. Parameter Exploration
assert len(unique_params) > threshold

# 2. Performance Variation
assert std(results) > epsilon

# 3. Optimization Effectiveness
assert best_result < mean(results)

# 4. Statistical Significance
assert p_value < 0.05
```

## Extension Points

### Custom Search Spaces

```python
class CustomSearchSpace:
    def suggest_params(self, trial):
        # Custom logic here
        return params
```

### Custom Callbacks

```python
class CustomCallback(Callback):
    def on_trial_end(self, study, trial):
        # Custom logic here
        pass
```

### Custom Metrics

```python
def custom_metric_extractor(trainer):
    # Extract custom metric
    return trainer.callback_metrics["custom_metric"]
```

## Performance Considerations

### Memory Management
- Trials run sequentially or in parallel based on resources
- Each trial gets separate process (no memory leaks)
- Checkpoints saved optionally to reduce I/O

### Computational Efficiency
- Early pruning reduces wasted compute
- Parallel trials maximize resource usage
- Smart sampling focuses on promising regions

### Scalability
- Supports distributed optimization via shared storage
- Can resume from interruptions
- Handles thousands of trials

## Security Considerations

- No execution of arbitrary code from configs
- Validates all inputs before processing
- Secure handling of database credentials for distributed mode
- No network calls except to configured storage

## Future Enhancements

### Planned Features
- Ray Tune backend support (in addition to Optuna)
- Automatic hyperparameter importance analysis
- Integration with experiment tracking (MLflow, W&B)
- Multi-objective optimization support

### Non-Goals
- Not trying to replace Optuna/Ray Tune
- Not implementing custom optimization algorithms
- Not handling non-Lightning training loops