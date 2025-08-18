# Interactive Pause/Resume for Ray Tune

## Overview

Lightning BOHB now provides **LightningReflow-style pause/resume capabilities** for Ray Tune hyperparameter optimization sessions. Press 'p' during optimization to pause all trials at their next validation boundaries, exit safely, and resume later with a single command.

## Features

- ⏸️ **Interactive Pause**: Press 'p' to pause at validation boundaries
- 💾 **Safe Checkpointing**: All trials save state before pausing
- 🔄 **Easy Resume**: Single command to continue from where you left off
- 🎮 **Familiar Interface**: Same experience as LightningReflow's pause/resume
- 🌍 **Global Control**: Pause/resume entire optimization session

## Quick Start

### 1. Basic Usage

```python
from lightning_bohb import ConfigDrivenOptimizer
from lightning_bohb.core.strategies_v2 import BOHBStrategy
from lightning_bohb.core.tune_reflow_cli import TuneReflowCLI

# Create CLI wrapper
cli = TuneReflowCLI(
    experiment_name="my_optimization",
    experiment_dir="./experiments"
)

# Create optimizer as usual
strategy = BOHBStrategy(grace_period=10, reduction_factor=3)
optimizer = ConfigDrivenOptimizer(
    base_config_source="config.yaml",
    search_space=search_space,
    strategy=strategy,
)

# Run with pause/resume support
results = cli.run(optimizer)
```

### 2. Interactive Controls

While optimization is running:
- Press **'p'** - Pause at next validation boundary
- Press **'q'** - Quit immediately (saves checkpoint)
- Press **Ctrl+C** - Graceful pause (same as 'p')

### 3. Resume Command

After pausing, the system prints the exact resume command:

```bash
✅ OPTIMIZATION PAUSED SUCCESSFULLY
📊 Experiment: my_optimization
📁 Saved to: ./experiments/my_optimization
🔄 Trials paused: 4/4

📝 To resume this optimization session, run:

python optimize.py --experiment-name my_optimization --resume
```

## Command Line Interface

### Start New Optimization

```bash
python optimize.py \
    --experiment-name world_model \
    --base-config config.yaml \
    --strategy bohb \
    --max-epochs 100 \
    --max-trials 4
```

### Resume Paused Optimization

```bash
python optimize.py \
    --experiment-name world_model \
    --base-config config.yaml \
    --resume
```

### Disable Interactive Mode

```bash
python optimize.py \
    --experiment-name world_model \
    --base-config config.yaml \
    --no-pause
```

## Architecture

### How It Works

1. **Main Process Monitoring**
   - `TuneReflowCLI` monitors keyboard in the main Tuner process
   - Uses LightningReflow's `UnifiedKeyboardHandler` for non-blocking input

2. **Signal Propagation**
   - Pause request written to signal file
   - All trials check signal file at validation boundaries

3. **Coordinated Pause**
   - Each trial pauses at its next validation
   - Saves checkpoint before stopping
   - Reports pause status to Ray Tune

4. **Resume Mechanism**
   - Ray Tune's `Tuner.restore()` loads experiment state
   - Each trial resumes from its checkpoint
   - Optimization continues seamlessly

### Components

```
TuneReflowCLI (Main Process)
    ├── Keyboard Monitor
    ├── Signal File Writer
    └── Resume Command Generator

TunePauseCallback (Each Trial)
    ├── Signal File Reader
    ├── Validation Boundary Detector
    └── Checkpoint Saver
```

## Integration with Existing Code

### Method 1: CLI Wrapper

```python
from lightning_bohb.core.tune_reflow_cli import TuneReflowCLI

# Wrap your existing optimizer
cli = TuneReflowCLI.from_args()  # Parse from command line
results = cli.run(optimizer)
```

### Method 2: Manual Integration

```python
from lightning_bohb.callbacks.tune_pause_callback import TunePauseCallback

# Add callback to your optimizer
pause_callback = TunePauseCallback(
    pause_signal_file="/tmp/tune_pause.signal"
)
optimizer.additional_callbacks.append(pause_callback)
```

### Method 3: Full Example Script

```python
#!/usr/bin/env python
"""optimize.py - Optimization with pause/resume."""

from lightning_bohb import ConfigDrivenOptimizer
from lightning_bohb.core.strategies_v2 import BOHBStrategy
from lightning_bohb.core.tune_reflow_cli import TuneReflowCLI

def main():
    # Create CLI from command line args
    cli = TuneReflowCLI.from_args()
    
    # Your optimization setup
    strategy = BOHBStrategy(grace_period=10)
    optimizer = ConfigDrivenOptimizer(
        base_config_source="config.yaml",
        search_space=create_search_space(),
        strategy=strategy,
    )
    
    # Run with CLI wrapper
    results = cli.run(
        optimizer,
        resume=cli._should_resume,
    )
    
    print(f"Best config: {results.get_best_result().config}")

if __name__ == "__main__":
    main()
```

## Use Cases

### 1. Long-Running Optimizations

Perfect for optimizations that run for days:
- Pause before meetings or end of day
- Resume when resources are available
- No lost progress

### 2. Shared Resources

Coordinate resource usage:
- Pause to free GPUs for other tasks
- Resume during off-peak hours
- Flexible scheduling

### 3. Iterative Development

Inspect intermediate results:
- Pause to analyze current best configs
- Adjust search space if needed
- Resume with insights

### 4. Cost Management

Control cloud computing costs:
- Pause before budget limits
- Resume when new budget available
- Optimize spending

## Advanced Features

### Custom Pause Conditions

```python
class ConditionalTuneReflowCLI(TuneReflowCLI):
    def should_pause(self, results):
        # Pause if best loss is good enough
        if results.best_metric < 0.1:
            return True
        # Pause after certain time
        if time.time() - self.start_time > 3600:
            return True
        return False
```

### Pause with Notification

```python
cli = TuneReflowCLI(
    experiment_name="my_opt",
    on_pause=lambda: send_email("Optimization paused")
)
```

### Automated Resume

```bash
# Cron job to resume at night
0 22 * * * cd /path/to/project && python optimize.py --resume
```

## Comparison with LightningReflow

| Feature | LightningReflow | TuneReflowCLI |
|---------|-----------------|---------------|
| Pause key | 'p' | 'p' |
| Pause boundary | Validation | Validation |
| Resume command | ✅ Printed | ✅ Printed |
| Checkpoint | Single model | All trials |
| Scope | One training | Full optimization |

## Technical Details

### Signal File Format

```json
{
    "pause_requested": true,
    "timestamp": 1704067200.0,
    "experiment_name": "world_model_opt"
}
```

### Checkpoint Structure

```
experiments/
└── world_model_opt/
    ├── trial_0/
    │   └── pause_checkpoints/
    │       ├── pause_epoch_45.ckpt
    │       └── pause_metadata.json
    ├── trial_1/
    │   └── pause_checkpoints/
    │       ├── pause_epoch_23.ckpt
    │       └── pause_metadata.json
    └── experiment_state.json
```

### Resume Process

1. `Tuner.restore()` loads experiment state
2. Each trial finds its pause checkpoint
3. Training resumes from exact pause point
4. Optimization continues with all history preserved

## Limitations

- **Validation Boundary**: Pause only occurs at validation, not mid-epoch
- **Global Pause**: All trials pause together (no individual trial pause yet)
- **Signal File**: Requires shared filesystem for multi-node setups

## Future Enhancements

- [ ] Individual trial pause/resume
- [ ] Web UI for pause control
- [ ] Pause scheduling (e.g., pause at 6 PM)
- [ ] Resource-aware pause (e.g., pause when GPU memory low)
- [ ] Checkpoint compression for large models

## Summary

TuneReflowCLI brings the intuitive pause/resume experience of LightningReflow to Ray Tune optimization. This enables:

- ✅ **Better resource management** - Pause/resume based on availability
- ✅ **Cost control** - Stop when needed without losing progress  
- ✅ **Flexibility** - Adapt to changing priorities
- ✅ **Peace of mind** - Safe to interrupt at any time

The same simple 'p' key that pauses LightningReflow training now pauses entire hyperparameter searches!