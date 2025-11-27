# LightningTune Architecture - Quick Reference Guide

## Component Hierarchy

```
┌─ HPORunner ────────────────────────────────┐
│  - CLI argument parsing                    │
│  - Config merging                          │
│  - Checkpoint loading                      │
│  - Argument restoration                    │
│  └─ creates and calls                      │
│                                            │
│  PausibleOptunaOptimizer ─────────────────┐
│  - Trial-by-trial loop                    │
│  - Pause/resume management                │
│  - Study integrity checking               │
│  - WandB + local checkpointing            │
│  └─ creates and calls                     │
│                                           │
│  OptunaDrivenOptimizer / ReflowOptuna....│
│  - Single trial objective function        │
│  - Config merging per trial               │
│  - Callback composition                   │
│  - Lightning trainer creation             │
│  └─ calls                                 │
│                                           │
│  LightningReflow or Trainer.fit()        │
│  - Model instantiation                    │
│  - Training loop execution                │
│  - Checkpoint/callback handling           │
└────────────────────────────────────────────┘
```

## Key Classes and Responsibilities

### HPORunner
- **File:** `hpo_runner.py`
- **Main Methods:**
  - `run_from_cli(argv=None)` - Main entry point
  - `_parse_dot_notation_args()` - Parse `--model.lr 0.1` style args
  - `_load_checkpoint()` - Load from WandB or local file
  - `_restore_args_from_checkpoint()` - Restore saved arguments
- **Outputs:** `optuna.Study` with results
- **Config Flow:** Base → Override → CLI args → Dot-notation overrides

### PausibleOptunaOptimizer
- **File:** `pausible_optimizer.py`
- **Main Method:** `optimize(n_trials, resume_from=None, config_overrides=None)`
- **Key Responsibilities:**
  1. Trial loop: `while total < n_trials and not pause`
  2. Pause detection: Check keyboard 'p', 'q' keys
  3. Study integrity: Only save COMPLETE/PRUNED trials
  4. Checkpointing: WandB artifacts + local filesystem
  5. Resume handling: Load study, restore trial count, extend n_trials
- **Architecture:** Runs optimizer one trial at a time for clean pause boundaries

### OptunaDrivenOptimizer / ReflowOptunaDrivenOptimizer
- **Files:** `optimizer.py`, `optimizer_reflow.py`
- **Main Method:** `create_objective() -> Callable[[Trial], float]`
- **Single Trial Workflow:**
  1. Merge base config + overrides + search space suggestions
  2. Instantiate model and datamodule from merged config
  3. Create callbacks (pruning, NaN detection, checkpointing)
  4. Create trainer with callbacks and WandB logger
  5. Call `trainer.fit()` or `reflow.fit()`
  6. Return monitored metric value
- **Differences:**
  - OptunaDrivenOptimizer: Vanilla Lightning
  - ReflowOptunaDrivenOptimizer: LightningReflow (env vars, torch.compile, better pause)

## Configuration System

### Loading and Merging
```
load_config(source)           # source: str|Path|dict → dict
  ├─ YAML file
  ├─ JSON file
  └─ Already a dict

deep_merge_configs(base, override)  # Recursive merge
  └─ Base preserved, override values applied
  
apply_dotted_updates(config, updates)  # "model.lr": 0.1 → {model: {lr: 0.1}}
  └─ Creates nested structure as needed
```

### Per-Trial Application Order
```
1. Start with base_config
2. Apply fixed config_overrides (from HPORunner/PausibleOptuna)
3. Apply search space suggestions (from trial.suggest_*)
4. Instantiate model/datamodule with final config
```

## Pause/Resume Architecture

### Pause Detection
```
Background Thread (_pause_input_loop)
  └─ Continuously calls keyboard_handler.get_key()
     ├─ 'p' pressed → Toggle _pause_requested flag
     ├─ 'q' pressed → Set _quit_after_current flag
     └─ Messages printed directly (bypass progress bar)

Main Loop
  └─ Before/after each trial: Check _update_pause_from_keyboard()
     └─ If pause requested: Break and save checkpoint
```

### Study Integrity Verification
```
_verify_study_integrity(study):
  ├─ Count COMPLETE + PRUNED → finished_count
  ├─ Check no RUNNING or WAITING trials → valid
  └─ Return (is_valid, finished_count, message)
  
Only save if valid (all trials finished)
```

### Checkpoint Format
```python
session_info = {
    "study": pickled_optuna_study,
    "total_trials_completed": int,
    "sampler_name": str,
    "pruner_name": str,
    "study_name": str,
    "config_overrides": {
        "args.trial_steps": 1000,
        "model.init_args.learning_rate": 0.001,
        # ... other persistent config
    }
}
```

### Resume Strategy
```
resume_from options:
  ├─ 'local'   → Load from local checkpoint (most up-to-date)
  │              Path: checkpoints/{wandb_project}/{study_name}/study.pkl
  ├─ 'latest'  → Load from WandB artifact (when --wandb is set)
  │              Artifact: {wandb_project}/{study_name}_checkpoint:latest
  ├─ 'vN'      → Load specific WandB version (e.g., 'v5')
  └─ '/path'   → Load from explicit file path

Why local vs WandB matters:
  - Local checkpoints are saved after EVERY trial
  - WandB artifacts are uploaded every N trials (--upload-every)
  - After a crash, local may have more trials than WandB

Restoration:
  1. Load study, extract sampler/pruner states
  2. Restore config_overrides
  3. Restore saved CLI arguments (unless explicitly overridden)
  4. Handle n_trials extension if requested > saved
  5. Resume from exact trial count
```

### Auto-Restart for Memory Management
```
Problem: GPU/CPU memory accumulates over many trials
Solution: Process restart after each checkpoint save

Enabled by default (via CLI):
  - HPORunner sets restart_on_save=True unless --debug-no-restart

Flow:
  1. Complete trial N
  2. Check if save_every_n_trials reached
  3. Save checkpoint locally
  4. Exit with code 42
  5. Wrapper script relaunches with --resume-from local
  6. Continue from trial N+1 with fresh memory

CLI Flags:
  --debug-no-restart    # Disable auto-restart (for debugging)

Exit Code 42:
  - Special code signals "restart needed" to wrapper scripts
  - Distinguishes from errors (non-zero) and success (0)
```

## Search Space Abstractions

### SimpleSearchSpace (Declarative)
```python
space = SimpleSearchSpace({
    "param_name": ("type", arg1, arg2, ...),
    # Types: "uniform", "loguniform", "int", "categorical", "discrete_uniform"
})
# Example:
# "model.init_args.learning_rate": ("loguniform", 1e-5, 1e-3)
# "model.init_args.hidden_size": ("categorical", [128, 256, 512])
```

### ConditionalSearchSpace (Logic-Based)
```python
class MySpace(ConditionalSearchSpace):
    def suggest_params(self, trial):
        arch = trial.suggest_categorical("model.arch", ["cnn", "rnn"])
        if arch == "cnn":
            params = {"model.filters": trial.suggest_int("model.filters", 32, 256)}
        else:
            params = {"model.hidden": trial.suggest_int("model.hidden", 64, 512)}
        return params
```

### CompositeSearchSpace (Modular)
```python
composite = CompositeSearchSpace({
    "model": SimpleSearchSpace({...}),
    "data": SimpleSearchSpace({...}),
})
```

### DynamicSearchSpace (Adaptive)
```python
space = DynamicSearchSpace(initial_params)
space.update_search_space({"new_param": ("uniform", 0, 1)})
space.narrow_search_space("learning_rate", factor=0.5)  # Halve range
```

## Callback Architecture

### Standard Optuna Callbacks
```
OptunaPruningCallback
  └─ on_validation_end: Report metric and check prune decision

OptunaCheckpointCallback
  └─ on_validation_end: Save best checkpoint metadata

OptunaProgressCallback
  └─ on_train_epoch_end: Display progress

OptunaEarlyStoppingCallback
  └─ on_validation_end: Early stopping with patience

PruneOnExceptionCallback
  └─ on_exception: Convert exception to TrialPruned (preserves Ctrl+C)
```

### Enhanced Callbacks
```
EnhancedOptunaPruningCallback
  └─ Pruning with NaN detection

NaNDetectionCallback
  └─ Detects NaN/Inf at training steps and terminates early
```

### Callback Factory
```python
callbacks = build_optuna_callbacks(trial, monitor="val_loss")
# Returns: [EnhancedOptunaPruning, NaNDetection, PruneOnException]
# With graceful fallbacks if enhanced versions unavailable
```

## Optuna Integration

### Sampler Creation
```python
from LightningTune import create_sampler

sampler = create_sampler(
    "tpe",                    # "tpe", "random", "grid", "cmaes", "botorch"
    seed=42,                  # Optional, for reproducibility
    n_startup_trials=10       # Kwargs passed to sampler constructor
)
```

### Pruner Creation
```python
from LightningTune import create_pruner

pruner = create_pruner(
    "hyperband",              # "median", "hyperband", "successivehalving", "none"
    max_resource=100,         # Override defaults
    reduction_factor=3
)
```

### Study Management
```python
study = optuna.create_study(
    study_name=name,
    sampler=sampler,
    pruner=pruner,
    direction="minimize",     # or "maximize"
    storage=None,             # or "sqlite:///db.sqlite" for distributed
    load_if_exists=True       # Resume if study already exists
)

study.optimize(objective, n_trials=1, gc_after_trial=True)
```

## Resource Management

### Memory Cleanup (Per-Trial)
```python
cleanup_trial_resources():
  1. gc.collect()                    # Python GC
  2. torch.cuda.empty_cache()        # GPU cache
  3. torch.cuda.synchronize()        # Sync CUDA
  4. torch.cuda.reset_peak_memory_stats()  # Reset peaks
  5. plt.close('all')                # Close matplotlib
  6. gc.collect()                    # GC again
```

### Torch Compile State Reset (ReflowOptuna Only)
```python
_reset_torch_compile_state():
  1. torch._dynamo.reset()           # Reset dynamo cache
  2. torch.cuda.empty_cache() x2     # GPU cache before/after sync
  3. torch.cuda.manual_seed_all()    # Reset RNG states
```

## Extension Points

### Add Custom Search Space
```python
class MySearchSpace(ConditionalSearchSpace):
    def suggest_params(self, trial):
        return {...}
    
    @property
    def param_names(self):
        return [...]

runner.search_space = MySearchSpace()
```

### Add Custom Callbacks
```python
class MyCallback(Callback):
    def __init__(self, trial):
        self.trial = trial
    
    def on_validation_end(self, trainer, pl_module):
        value = trainer.callback_metrics["val_loss"].item()
        self.trial.report(value, trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()

runner.callbacks.append(MyCallback(trial))
```

### Direct Sampler/Pruner Injection
```python
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

optimizer = PausibleOptunaOptimizer(
    base_config=...,
    search_space=...,
    sampler=TPESampler(n_startup_trials=20),
    pruner=HyperbandPruner(min_resource=1, max_resource=50),
    ...
)
```

### Extend HPORunner CLI
```python
runner = HPORunner(
    ...,
    additional_cli_args={
        'my_arg': {
            'type': int,
            'default': 10,
            'help': 'My custom argument'
        }
    }
)
```

## Common Patterns

### Minimal HPO Setup
```python
from LightningTune import HPORunner
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

runner = HPORunner(
    model_class=MyModel,
    datamodule_class=MyDataModule,
    search_space=search_space_fn,
    base_config="config.yaml"
)

study = runner.run_from_cli()
```

### CLI Usage
```bash
# Start fresh
python train_hpo.py --n-trials 100 --wandb my-project --study-name exp1

# Resume from local checkpoint (most up-to-date, saved after every trial)
python train_hpo.py --resume-from local

# Resume from WandB (for cross-machine workflows)
python train_hpo.py --resume-from latest --wandb my-project --study-name exp1

# Resume and extend trials
python train_hpo.py --resume-from local --n-trials 200

# Resume from specific WandB version
python train_hpo.py --resume-from v5 --wandb my-project --study-name exp1

# Resume from explicit file path
python train_hpo.py --resume-from /path/to/study.pkl

# Override config on CLI
python train_hpo.py --model.init_args.dropout 0.2 --data.init_args.batch_size 64

# Disable auto-restart (for debugging)
python train_hpo.py --n-trials 10 --debug-no-restart
```

### Programmatic Usage (no CLI)
```python
runner = HPORunner(...)
study = runner.run(
    n_trials=100,
    wandb="my-project",
    study_name="exp1",
)
```

## Architecture Strengths

1. **Clean Dependency Injection** - No strategy abstraction, direct Optuna use
2. **Pause/Resume at Boundaries** - Never mid-trial, clean checkpointing
3. **Layered Configuration** - Base + override + CLI args, all composable
4. **Dual Checkpointing** - WandB + local filesystem, with fallback chain
5. **Resource Management** - Explicit cleanup between trials, Torch compile reset
6. **Extension Points** - Search spaces, callbacks, samplers/pruners, CLI args
7. **LightningReflow Integration** - Optional, enables torch.compile and better pause handling

## Known Limitations

1. **Full Study Serialization** - Pickles entire study, becomes large over time
2. **Config Per-Trial Copying** - Deep copies configs on every trial
3. **Same-Process Isolation** - No subprocess isolation between trials
4. **Generic Error Handling** - Failed trials indistinguishable from bad hyperparams
5. **Test Injection Hooks** - `underlying_optimizer` and `create_objective()` method patching

## Related Files

- Configuration: `utils/config_utils.py`, `arg_persistence.py`
- Persistence: `persistence.py`
- Utilities: `memory_cleanup.py`, `keyboard_monitor.py`
- Callbacks: `optuna/callbacks.py`, `optuna/callback_factory.py`, `optuna/nan_detection_callback.py`
- WandB: `utils/wandb_logger.py`, `optuna/wandb_integration.py`
- Torch Compile: `utils/torch_compile.py`

