# Lightning BOHB Package Structure

## Directory Organization

```
lightning_bohb/
├── __init__.py                 # Main package exports
├── __version__.py             # Version information
│
├── core/                      # Core optimization components
│   ├── __init__.py
│   ├── optimizer_v4.py       # Main ConfigDrivenOptimizer (with DI)
│   ├── strategies_v2.py      # Strategy implementations (DI-ready)
│   ├── config.py             # SearchSpace, ConfigManager
│   └── [legacy versions]     # optimizer_v2.py, optimizer_v3.py, etc.
│
├── cli/                       # Command-line interfaces
│   ├── __init__.py
│   └── tune_reflow.py        # TuneReflowCLI for pause/resume
│
├── callbacks/                 # Lightning callbacks
│   ├── __init__.py
│   ├── report.py             # BOHBReportCallback, AdaptiveBOHBCallback
│   └── tune_pause_callback.py # TunePauseCallback, TuneResumeCallback
│
├── examples/                  # Usage examples
│   ├── basic_example.py
│   ├── dependency_injection_example.py
│   ├── tune_pause_resume_example.py
│   ├── proper_resume_example.py
│   └── world_model_optimization.py
│
└── docs/                      # Documentation
    ├── README.md
    ├── ARCHITECTURE.md
    ├── DEPENDENCY_INJECTION.md
    ├── PAUSE_RESUME.md
    └── STRATEGY_PATTERN.md
```

## Import Hierarchy

### User-Facing Imports

```python
from lightning_bohb import (
    # Main optimizer
    ConfigDrivenOptimizer,
    
    # Strategies (for DI)
    BOHBStrategy,
    OptunaStrategy,
    RandomSearchStrategy,
    
    # Configuration
    SearchSpace,
    OptimizationConfig,
    
    # CLI
    TuneReflowCLI,
    
    # Callbacks
    TunePauseCallback,
)
```

### Internal Organization

1. **Core** (`lightning_bohb.core`)
   - Optimization logic
   - Strategy implementations
   - Configuration management

2. **CLI** (`lightning_bohb.cli`)
   - Command-line interfaces
   - Interactive controls
   - Session management

3. **Callbacks** (`lightning_bohb.callbacks`)
   - Lightning callbacks
   - Ray Tune integration
   - Pause/resume handlers

## Module Responsibilities

### `core/` - Optimization Engine
- **optimizer_v4.py**: Main optimizer with dependency injection
- **strategies_v2.py**: All strategy implementations
- **config.py**: Search space and config management

### `cli/` - User Interfaces
- **tune_reflow.py**: Interactive pause/resume CLI
  - Keyboard monitoring
  - Session state management
  - Resume command generation

### `callbacks/` - Event Handlers
- **report.py**: Metrics reporting to Ray Tune
- **tune_pause_callback.py**: Pause detection and execution
  - Signal file monitoring
  - Validation boundary detection
  - Checkpoint management

## Usage Patterns

### 1. Basic Usage
```python
from lightning_bohb import ConfigDrivenOptimizer, BOHBStrategy

strategy = BOHBStrategy(grace_period=10)
optimizer = ConfigDrivenOptimizer(
    base_config_source="config.yaml",
    search_space=search_space,
    strategy=strategy,
)
results = optimizer.run()
```

### 2. With Pause/Resume
```python
from lightning_bohb import TuneReflowCLI

cli = TuneReflowCLI(experiment_name="my_opt")
results = cli.run(optimizer)
```

### 3. Direct Callback Usage
```python
from lightning_bohb import TunePauseCallback

callback = TunePauseCallback()
optimizer.additional_callbacks.append(callback)
```

## Version Compatibility

- **v3.x**: Current version with DI and pause/resume
- **v2.x**: Legacy with string-based strategies (backward compatible)
- **v1.x**: Original BOHB-only implementation (deprecated)

## Import Best Practices

1. **Import from top level** when possible:
   ```python
   from lightning_bohb import ConfigDrivenOptimizer
   ```

2. **Import from submodules** for advanced usage:
   ```python
   from lightning_bohb.core.strategies_v2 import CustomStrategy
   from lightning_bohb.cli.tune_reflow import TuneSessionState
   ```

3. **Avoid importing from version-numbered files**:
   ```python
   # Bad
   from lightning_bohb.core.optimizer_v4 import ConfigDrivenOptimizer
   
   # Good
   from lightning_bohb import ConfigDrivenOptimizer
   ```

## Testing Structure

```
tests/
├── unit/
│   ├── test_strategies.py
│   ├── test_optimizer.py
│   └── test_cli.py
├── integration/
│   ├── test_pause_resume.py
│   └── test_ray_tune.py
└── fixtures/
    └── sample_configs.yaml
```