# Complete Pause/Resume Implementation Guide

## Overview

This guide documents the complete implementation of LightningReflow-style pause/resume functionality for Ray Tune hyperparameter optimization in the LightningTune package.

## Architecture

### Component Structure

```
lightning_bohb/
├── cli/
│   └── tune_reflow.py          # Main CLI with keyboard monitoring
├── callbacks/
│   └── tune_pause_callback.py  # Pause detection in trials
└── core/
    └── optimizer_v4.py         # Optimizer with callback support
```

### Key Classes

1. **`TuneReflowCLI`** (`cli/tune_reflow.py`)
   - Monitors keyboard input in main process
   - Saves complete session state (search space, strategy, config)
   - Generates resume commands
   - Handles signal-based communication with trials

2. **`TunePauseCallback`** (`callbacks/tune_pause_callback.py`)
   - Runs in each trial process
   - Checks for pause signals at validation boundaries
   - Saves trial checkpoints before pausing
   - Reports pause status to Ray Tune

3. **`TuneSessionState`** (`cli/tune_reflow.py`)
   - Dataclass storing complete session information
   - Pickles search space and strategy objects
   - Enables exact restoration of optimization state

## Installation Requirements

```bash
# Required dependencies
pip install ray[tune]           # Ray Tune for optimization
pip install lightning            # PyTorch Lightning
pip install lightning-reflow    # For keyboard monitoring (optional but recommended)

# Optional for specific strategies
pip install optuna              # For OptunaStrategy
pip install hyperopt            # For HyperOptStrategy
```

## Usage Examples

### 1. Basic Usage with Pause/Resume

```python
from lightning_bohb import (
    ConfigDrivenOptimizer,
    TuneReflowCLI,
    SearchSpace,
    OptimizationConfig,
)
from lightning_bohb.core.strategies_v2 import BOHBStrategy
from ray import tune

# Define your search space
class MySearchSpace(SearchSpace):
    def get_search_space(self):
        return {
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([16, 32, 64]),
            "num_layers": tune.choice([2, 4, 6]),
        }
    
    def get_metric_config(self):
        return {"metric": "val_loss", "mode": "min"}

# Create strategy
strategy = BOHBStrategy(
    grace_period=10,
    reduction_factor=3,
    max_t=100,
)

# Create optimizer
optimizer = ConfigDrivenOptimizer(
    base_config_source="config.yaml",
    search_space=MySearchSpace(),
    strategy=strategy,
    optimization_config=OptimizationConfig(
        max_epochs=100,
        max_concurrent_trials=4,
        experiment_name="my_model",
    ),
)

# Wrap with CLI for pause/resume
cli = TuneReflowCLI(
    experiment_name="my_model",
    experiment_dir="./experiments",
)

# Run with interactive controls
try:
    results = cli.run(optimizer)
    print(f"Best config: {results.get_best_result().config}")
except KeyboardInterrupt:
    print("Optimization paused. Resume command printed above.")
```

### 2. Command-Line Script

```python
#!/usr/bin/env python
"""optimize.py - Optimization with pause/resume support."""

import argparse
from pathlib import Path
from lightning_bohb import ConfigDrivenOptimizer, TuneReflowCLI
from lightning_bohb.core.strategies_v2 import BOHBStrategy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--resume-session", type=str, help="Resume from session")
    
    args = parser.parse_args()
    
    if args.resume_session:
        # Resume from saved session
        cli = TuneReflowCLI.resume(session_dir=args.resume_session)
        results = cli.run(resume=True)
    else:
        # Start new optimization
        strategy = BOHBStrategy(grace_period=10)
        optimizer = ConfigDrivenOptimizer(
            base_config_source=args.base_config,
            search_space=create_search_space(),
            strategy=strategy,
        )
        
        cli = TuneReflowCLI(experiment_name=args.experiment_name)
        results = cli.run(optimizer)
    
    print(f"Best result: {results.get_best_result().config}")

if __name__ == "__main__":
    main()
```

### 3. Usage Workflow

```bash
# Start new optimization
$ python optimize.py --experiment-name my_model --base-config config.yaml

Output:
🎮 Interactive mode enabled:
   Press 'p' to pause at next validation
   Press 'q' to quit immediately
   Press Ctrl+C to pause gracefully

💾 Session state saved to: ./experiments/my_model/session_20240101_120000_abc123/session_state.pkl
Running optimization...

# Press 'p' or Ctrl+C to pause
⏸️ Pause requested. Waiting for trials to reach validation boundaries...

✅ OPTIMIZATION PAUSED
📊 Session: 20240101_120000_abc123
💾 State saved to: ./experiments/my_model/session_20240101_120000_abc123/session_state.pkl

📝 To resume this EXACT session with all settings, run:
   python optimize.py --resume-session ./experiments/my_model/session_20240101_120000_abc123

# Resume later (even on different machine with same code)
$ python optimize.py --resume-session ./experiments/my_model/session_20240101_120000_abc123

♻️ Resuming session: 20240101_120000_abc123
✅ All configuration restored from session state:
   • Search space: ✓
   • Strategy: ✓
   • Base config: ✓
   • Optimization config: ✓
Continuing optimization...
```

## How It Works

### 1. Session State Serialization

When starting a new optimization, TuneReflowCLI saves:
- **Search space object** (pickled)
- **Strategy object with all parameters** (pickled)
- **Optimization config** (pickled)
- **Base config path**
- **Session metadata**

```python
# Automatically saved to session directory
./experiments/my_model/session_20240101_120000_abc123/
├── session_state.pkl           # Complete pickled state
├── session_state.summary.yaml  # Human-readable summary
└── [Ray Tune checkpoints]      # Individual trial states
```

### 2. Pause Signal Propagation

```
Main Process (TuneReflowCLI)
    ↓
    Writes pause signal file
    ↓
/tmp/tune_pause_my_model_xxx.signal
    ↑
    Read by each trial
    ↑
Trial Processes (TunePauseCallback)
```

### 3. Validation Boundary Pausing

Each trial checks for pause signals at validation end:
```python
def on_validation_end(self, trainer, pl_module):
    if pause_signal_exists():
        save_checkpoint()
        report_to_ray_tune(paused=True)
        trainer.should_stop = True
```

### 4. Resume Process

1. Load session state from pickle
2. Restore optimizer with exact configuration
3. Ray Tune resumes from experiment checkpoint
4. Each trial loads its individual checkpoint
5. Training continues from exact pause point

## Advanced Features

### Custom Pause Conditions

```python
class CustomTuneReflowCLI(TuneReflowCLI):
    def should_auto_pause(self, metrics):
        # Auto-pause based on metrics
        if metrics.get("val_loss", float("inf")) < 0.1:
            return True
        # Auto-pause based on time
        if time.time() - self.start_time > 3600:  # 1 hour
            return True
        return False
```

### Pause with Notifications

```python
def on_pause_callback():
    send_email("Optimization paused", f"Resume: {resume_command}")
    save_to_database(session_info)

cli = TuneReflowCLI(
    experiment_name="my_model",
    on_pause=on_pause_callback,
)
```

### Scheduled Resume

```bash
# Cron job to resume at night
0 22 * * * cd /project && python optimize.py --resume-session /path/to/session
```

## Troubleshooting

### Issue: "Ray is required for TuneReflowCLI"

**Solution:**
```bash
pip install ray[tune]
```

### Issue: "Keyboard monitoring unavailable"

**Solution:**
```bash
pip install lightning-reflow
```

Note: Pause still works with Ctrl+C even without keyboard monitoring.

### Issue: "Session state not found"

**Cause:** Wrong session directory path

**Solution:** Check the exact path printed when pausing:
```bash
ls ./experiments/*/session_*/session_state.pkl
```

### Issue: "Cannot unpickle search space"

**Cause:** Code changes between pause and resume

**Solution:** Ensure the same code version is used for resume

## API Reference

### TuneReflowCLI

```python
class TuneReflowCLI:
    def __init__(
        self,
        experiment_name: str,
        experiment_dir: str = "./experiments",
        session_id: Optional[str] = None,
        enable_pause: bool = True,
        verbose: bool = True,
    )
    
    def run(
        self,
        optimizer: Optional[ConfigDrivenOptimizer] = None,
        resume: bool = False,
        **kwargs
    ) -> tune.ResultGrid
    
    @classmethod
    def resume(cls, session_dir: Union[str, Path]) -> 'TuneReflowCLI'
```

### TunePauseCallback

```python
class TunePauseCallback(Callback):
    def __init__(
        self,
        pause_signal_file: Optional[Path] = None,
        check_interval: float = 1.0,
        cli_instance: Optional[TuneReflowCLI] = None,
        verbose: bool = False,
    )
```

## Best Practices

1. **Always use session directories** for organization:
   ```python
   cli = TuneReflowCLI(
       experiment_name=f"{model_name}_{timestamp}",
       experiment_dir="./experiments",
   )
   ```

2. **Save session paths** for easy resume:
   ```python
   with open("active_sessions.txt", "a") as f:
       f.write(f"{cli.session_dir}\n")
   ```

3. **Version control your search space**:
   ```python
   class MySearchSpace(SearchSpace):
       VERSION = "1.0.0"  # Increment when changing
   ```

4. **Use descriptive experiment names**:
   ```python
   experiment_name = f"model_v2_lr_search_{date}"
   ```

5. **Clean up old sessions** periodically:
   ```bash
   find ./experiments -name "session_*" -mtime +30 -exec rm -rf {} \;
   ```

## Limitations

1. **Validation Boundary Only**: Pause occurs at validation end, not mid-epoch
2. **Global Pause**: All trials pause together (no individual trial pause)
3. **Code Consistency**: Same code version needed for resume
4. **Shared Filesystem**: Required for multi-node setups

## Future Enhancements

- [ ] Web UI for pause control
- [ ] Individual trial pause/resume
- [ ] Automatic pause scheduling
- [ ] Cloud storage for session states
- [ ] Checkpoint compression

## Summary

The pause/resume implementation provides:

✅ **Complete State Preservation**: Everything needed to resume is saved
✅ **Simple Interface**: Same 'p' key as LightningReflow
✅ **Robust Recovery**: Handles interruptions gracefully
✅ **Zero Configuration**: Works out of the box
✅ **Production Ready**: Suitable for long-running optimizations

This brings the intuitive pause/resume experience of LightningReflow to entire Ray Tune hyperparameter optimization sessions!