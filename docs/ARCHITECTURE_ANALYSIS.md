# LightningTune Codebase Architecture Analysis

## Executive Summary

LightningTune is a modern hyperparameter optimization (HPO) framework for PyTorch Lightning that wraps Optuna with pause/resume capabilities, WandB integration, and LightningReflow support. It provides a clean, dependency-injection-based architecture with minimal abstraction overhead while adding essential production features like checkpointing, resource management, and interactive pause/resume functionality.

**Current Version:** 0.4.0  
**Key Philosophy:** Direct dependency injection - no unnecessary strategy abstractions, just use Optuna's samplers and pruners directly.

---

## Core Architecture Overview

### Layered Component Stack

```
┌─────────────────────────────────────────────────────────────┐
│ HPORunner (CLI & Persistence Layer)                         │
│ - Parses CLI arguments                                      │
│ - Manages checkpoint loading/saving                         │
│ - Orchestrates pause/resume state restoration              │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ PausibleOptunaOptimizer (Trial Loop & Pause Management)    │
│ - Manages study.optimize() loop (one trial at a time)      │
│ - Handles 'p' key detection for pause requests             │
│ - WandB artifact checkpointing (periodic saves)            │
│ - Trial integrity verification (no incomplete trials)      │
│ - Local checkpoint mirroring                               │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ OptunaDrivenOptimizer / ReflowOptunaDrivenOptimizer        │
│ - Single trial objective function                          │
│ - Config merging & parameter suggestion                    │
│ - Lightning trainer instantiation                          │
│ - Callback wiring (pruning, NaN detection, etc.)          │
│ - WandB trial logging                                      │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ LightningReflow / Vanilla Lightning                        │
│ - Model instantiation and compilation                      │
│ - Training loop execution (fit)                            │
│ - Trainer callbacks & checkpoint saving                    │
│ - DataModule lifecycle management                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Core Abstractions and Interfaces

### 1.1 Base Optimizer Classes

#### **OptunaDrivenOptimizer** (`optimizer.py`)
- **Purpose:** Simple, clean Optuna wrapper with direct dependency injection
- **Key Methods:**
  - `create_objective()` → Returns the trial objective function
  - `optimize()` → Runs N trials with progress reporting
  - `get_best_config()` → Extracts best trial hyperparameters
  
- **Key Responsibilities:**
  - Configuration loading from YAML/JSON/dict
  - Config merging with dotted notation (`model.lr`, `data.batch_size`)
  - Objective function construction
  - Model/DataModule instantiation
  - Callback management (pruning, checkpointing, progress)
  - WandB logging setup
  
- **Design Pattern:** Pure dependency injection
  - Sampler and pruner passed directly (not string-based factory)
  - Config overrides applied at multiple stages (base → overrides → search space suggestions)
  - Clean separation between config management and training logic

#### **ReflowOptunaDrivenOptimizer** (`optimizer_reflow.py`)
- **Purpose:** Optuna optimizer with LightningReflow integration
- **Differences from OptunaDrivenOptimizer:**
  - Uses LightningReflow for proper environment variable setup
  - Enables torch.compile() for better performance
  - Handles PauseCallback removal (HPO manages pause at trial boundaries)
  - Provides fallback to vanilla Lightning if Reflow unavailable
  
- **Key Method:** `create_objective()` uses LightningReflow.fit() instead of Trainer.fit()
  
- **Environment Management:**
  - Sets CUDA environment variables via Reflow
  - Manages torch._dynamo state between trials
  - Resets peak memory stats to prevent accumulation
  - Synchronizes CUDA state cleanly

### 1.2 High-Level Runner Interface

#### **PausibleOptunaOptimizer** (`pausible_optimizer.py`)
- **Purpose:** Wraps OptunaDrivenOptimizer/ReflowOptunaDrivenOptimizer with pause/resume
- **Key Responsibilities:**
  - Trial-by-trial loop management (runs one trial per `study.optimize()` call)
  - Keyboard monitoring for 'p' pause and 'q' quit requests
  - WandB artifact checkpointing (periodic and on pause)
  - Local filesystem checkpoint mirroring
  - Study integrity verification (only saves COMPLETE/PRUNED trials)
  - Resume logic: loads study, restores trial counts, extends n_trials if needed
  - Config override persistence and restoration
  - Argument persistence (via args extractor)
  
- **Pause Architecture:**
  - Keyboard handler optionally loads from LightningReflow (`create_improved_keyboard_handler`)
  - Fallback to internal KeyboardMonitor for cross-platform support
  - Polling thread for non-blocking keyboard input detection
  - Pause only occurs at trial boundaries (clean state)
  - Saves study to WandB/local before pausing

#### **HPORunner** (`hpo_runner.py`)
- **Purpose:** High-level CLI interface that encapsulates all argument handling and resumption logic
- **Key Responsibilities:**
  - CLI argument parsing (including dot-notation for config overrides)
  - Config file loading and merging
  - Checkpoint loading and argument restoration
  - EarlyStoppingSteps callback injection (if trial_steps specified)
  - Trainer config standardization for HPO (validation intervals, progress bar)
  - Resume state management
  - Study name auto-generation
  
- **Workflow:**
  1. Parse CLI args (known + dot-notation unknown args)
  2. Load base + override configs, merge them
  3. If resume_from: load checkpoint, restore saved args (respecting explicit overrides)
  4. Build final config overrides from args and dot-notation
  5. Create and run PausibleOptunaOptimizer with restored state
  
- **Non-Persistent Args:** `{resume_from, study_name, n_trials}`
  - These are intentionally NOT persisted to allow resuming with different targets
  - n_trials can be extended on resume

---

## 2. Optuna Integration Strategy

### 2.1 Sampler and Pruner Management

#### **Factories** (`factories.py`)
- `create_sampler(name, seed=None, **kwargs)` → Returns BaseSampler instance
  - Supports: TPE, Random, Grid, CMA-ES, BoTorch (optional)
  - Seed parameter for reproducibility
  - Direct kwargs passthrough for flexibility
  
- `create_pruner(name, **kwargs)` → Returns BasePruner instance
  - Supports: Median, Hyperband, SuccessiveHalving, Nop
  - Built-in defaults for common parameters
  - Kwargs override defaults

#### **Study Creation Pattern**
```python
sampler = create_sampler(sampler_name, seed=seed_from_config)
pruner = create_pruner(pruner_name)
study = optuna.create_study(
    study_name=study_name,
    sampler=sampler,
    pruner=pruner,
    direction=direction,
    storage=storage,  # Optional for distributed optimization
    load_if_exists=True  # Resume existing studies
)
```

### 2.2 Callback Integration

#### **Optuna Callbacks** (`callbacks.py`)
1. **OptunaPruningCallback** - Monitors metric and prunes unpromising trials
2. **OptunaCheckpointCallback** - Saves checkpoint metadata with trial info
3. **OptunaProgressCallback** - Real-time progress display
4. **OptunaEarlyStoppingCallback** - Early stopping with patience
5. **PruneOnExceptionCallback** - Converts unexpected errors to pruned trials (preserves Ctrl+C)

#### **Enhanced Callbacks** (`nan_detection_callback.py`)
- **EnhancedOptunaPruningCallback** - Pruning with NaN detection
- **NaNDetectionCallback** - Detects NaN/Inf in training and terminates early

#### **Callback Factory** (`callback_factory.py`)
- `build_optuna_callbacks(trial, monitor)` → Returns standard callback set
- Always includes: EnhancedOptunaPruning, NaNDetection, PruneOnException
- Gracefully falls back if enhanced callbacks unavailable

### 2.3 Search Space Abstraction

#### **OptunaSearchSpace** (`search_space.py`)
Abstract base for defining parameter spaces:
- `suggest_params(trial)` → Dict[str, value]
- `param_names` → List[str] of parameter names

#### **Implementations**

**SimpleSearchSpace** - Declarative parameter ranges
```python
space = SimpleSearchSpace({
    "model.init_args.learning_rate": ("loguniform", 1e-5, 1e-3),
    "model.init_args.dropout": ("uniform", 0.1, 0.5),
    "model.init_args.hidden_size": ("categorical", [128, 256, 512]),
    "data.init_args.batch_size": ("int", 16, 128, 8),  # with step
})
```

**ConditionalSearchSpace** - Parameters that depend on other parameters
```python
class MySpace(ConditionalSearchSpace):
    def suggest_params(self, trial):
        arch = trial.suggest_categorical("model.arch", ["cnn", "transformer"])
        if arch == "cnn":
            return {"model.arch": arch, "model.filters": trial.suggest_int(...)}
        else:
            return {"model.arch": arch, "model.heads": trial.suggest_int(...)}
```

**CompositeSearchSpace** - Combines multiple search spaces with prefixes

**DynamicSearchSpace** - Search space that can be adapted during optimization

---

## 3. LightningReflow Integration

### 3.1 Purpose and Role

LightningReflow provides:
- **Environment Configuration:** Proper CUDA/environment variable setup
- **Torch Compilation:** Automatic `torch.compile()` application
- **Callback Management:** Handles PauseCallback removal for HPO
- **Robust Keyboard Handling:** Better terminal state restoration

### 3.2 Integration Points

#### **In ReflowOptunaDrivenOptimizer**

```python
if self.use_reflow:
    reflow = LightningReflow(
        model_class=...,
        datamodule_class=...,
        model_init_args=...,
        datamodule_init_args=...,
        trainer_defaults=trainer_config,  # Includes logger
        callbacks=callbacks,
        disable_pause_callback=True,  # Critical for HPO
        auto_configure_logging=False,
    )
    result = reflow.fit()
```

#### **Keyboard Handler**
```python
if create_improved_keyboard_handler is not None:
    self.keyboard_handler = create_improved_keyboard_handler(test_mode=test_mode)
```

- Imports from `lightning_reflow.callbacks.pause.improved_keyboard_handler`
- Better TTY handling than cross-platform alternatives
- Test mode support for automated testing

### 3.3 Fallback Strategy

```
LightningReflow available?
├─ Yes → Use ReflowOptunaDrivenOptimizer (full features)
└─ No  → Use OptunaDrivenOptimizer (vanilla Lightning, still works)
```

---

## 4. Pause Functionality Architecture

### 4.1 Pause Request Detection

#### **Keyboard Handler** (`keyboard_monitor.py`)
- Cross-platform keyboard input with non-blocking reads
- Unix: Uses `select` + `termios`/`tty` for terminal mode
- Windows: Uses `msvcrt.kbhit()` + `msvcrt.getch()`
- Context manager for proper terminal state restoration

#### **Detection Flow in PausibleOptunaOptimizer**
```
Main Loop (while total_trials < n_trials):
  ├─ Before Trial: Check _update_pause_from_keyboard()
  ├─ Run Trial: study.optimize(objective, n_trials=1)
  ├─ After Trial: Check _update_pause_from_keyboard()
  │   └─ If pause requested: break
  └─ Periodic: Save to WandB every save_every_n_trials

_update_pause_from_keyboard():
  ├─ Get key from keyboard_handler
  ├─ 'p' → Toggle _pause_requested
  ├─ 'q' → Set _quit_after_current flag
  └─ Ctrl+C → Set should_pause = True, raise KeyboardInterrupt eventually
```

### 4.2 Background Polling Thread

For improved UX with Lightning's progress bar:
- Daemon thread (`_pause_poll_thread`) continuously monitors keyboard
- Non-blocking so main loop remains responsive
- Updates `_pause_requested` flag in thread-safe manner
- Messages printed directly to stdout to bypass progress bar

### 4.3 Pause State Saving

#### **Study Integrity Verification**
Before saving, `_verify_study_integrity()` checks:
- No RUNNING trials (incomplete, cannot save)
- No WAITING trials (incomplete, cannot save)
- Counts COMPLETE + PRUNED as valid finished trials
- Counts FAILED trials separately (not included in finished count)

#### **Checkpoint Format**
```python
session_info = {
    "study": study,  # Pickled Optuna study object
    "total_trials_completed": int,  # Count of finished trials
    "sampler_name": str,
    "pruner_name": str,
    "study_name": str,
    "config_overrides": Dict[str, Any],  # Persistent config for resumption
}
```

#### **Dual Checkpointing**
- **WandB Artifacts:** Public, versioned (v1, v2, ...), aliased as "latest"
- **Local Filesystem:** Mirrored at `checkpoints/{wandb_project}/{study_name}/`
  - Faster fallback if WandB unavailable
  - Absolute paths ensure reliable resume

### 4.4 Resume Logic

#### **In PausibleOptunaOptimizer.optimize()**
```
if resume_from:
  └─ Load checkpoint (prefer local → WandB → generic fallback)
     ├─ Extract study
     ├─ Extract total_trials_completed
     ├─ Restore config_overrides (including persistent args)
     ├─ Handle n_trials extension (saved < requested)
     └─ Verify finished trial count matches
```

#### **In HPORunner._restore_args_from_checkpoint()**
```
For each saved arg:
  ├─ Skip if non-persistent (resume_from, study_name, n_trials)
  ├─ Check if explicitly specified on current CLI
  │  ├─ Yes → Keep current value (override)
  │  └─ No  → Restore saved value
  └─ Log restoration/override decisions
```

---

## 5. Configuration Management

### 5.1 Configuration Loading

#### **load_config()** (`config_utils.py`)
- Unified loader for YAML, JSON, and dict configs
- Type-safe with FileNotFoundError on missing files
- Returns plain dict for easy manipulation

#### **Config Structure**
```yaml
seed_everything: 42
model:
  class_path: my_module.MyModel
  init_args:
    learning_rate: 0.001
    dropout: 0.1
data:
  class_path: my_module.MyDataModule
  init_args:
    batch_size: 32
    num_workers: 4
trainer:
  max_epochs: 10
  accelerator: auto
  devices: auto
```

### 5.2 Configuration Merging

#### **deep_merge_configs()**
- Recursive merge of nested dicts
- Preserves base structure, overrides values
- Handles Lightning CLI format (class_path + init_args)

#### **apply_dotted_updates()**
- Applies dotted-notation config overrides
- Creates missing intermediate dicts
- Examples:
  - `"model.init_args.learning_rate": 0.0001`
  - `"trainer.max_epochs": 20`

#### **Config Application Order**
```
1. Load base_config (YAML/dict)
2. Apply config_overrides (from search space suggestions)
3. Apply dotted_updates (from CLI args or persistent overrides)
4. For each trial:
   ├─ Run base_config → Apply overrides → Apply search space suggestions
   └─ Instantiate model and datamodule with final config
```

### 5.3 Argument Persistence

#### **_extract_persistable_args()** (PausibleOptunaOptimizer)
- Minimal blacklist-based extraction: `{resume_from, study_name, ...}`
- Only extracts simple types: str, int, float, bool, Path→str
- Defensive handling of argparse.Namespace, dict, or dynamic attributes

#### **Arguments Stored in Config Overrides**
```python
config_overrides = {
    "args.trial_steps": 1000,
    "args.batch_size": 32,
    "args.learning_rate": 0.001,
    # ... other persistent args
}
```

#### **Non-Persistent Args Excluded**
- `resume_from` - only valid in original context
- `study_name` - may change on resume
- `n_trials` - intentionally extensible (can increase on resume)

---

## 6. Trial Execution and Resource Management

### 6.1 Single-Trial Optimization Loop

#### **PausibleOptunaOptimizer.optimize()**
```python
while total_trials_completed < n_trials and not should_pause:
    # Count finished trials BEFORE this trial
    trials_before = len([t for t in study.trials 
                        if t.state in [COMPLETE, PRUNED]])
    
    # Run exactly one trial
    study.optimize(objective, n_trials=1, show_progress_bar=False, 
                   gc_after_trial=True)
    
    # Count finished trials AFTER
    trials_after = len([t for t in study.trials 
                       if t.state in [COMPLETE, PRUNED]])
    
    if trials_after > trials_before:
        # Trial finished (completed or pruned successfully)
        total_trials_completed = trials_after
        # Save checkpoint, log results
    else:
        # Trial failed (exception that wasn't pruned)
        pass
```

#### **Why One Trial Per Loop Iteration**
- Clean pause boundaries - never mid-trial
- Accurate checkpoint timing - only save after complete trials
- Resource cleanup - `cleanup_trial_resources()` called after each trial
- Keyboard responsiveness - checks pause request between trials

### 6.2 Resource Management

#### **cleanup_trial_resources()** (`memory_cleanup.py`)
1. Force Python garbage collection
2. Clear PyTorch CUDA cache
3. Synchronize CUDA operations
4. Reset peak memory stats (prevent accumulation)
5. Close matplotlib figures (if any)
6. Additional GC pass
7. Log memory usage (if psutil available)

#### **Optuna's gc_after_trial=True**
```python
study.optimize(objective, n_trials=1, gc_after_trial=True)
```
- Built-in Optuna garbage collection after trial completes
- Combined with our explicit cleanup for robustness

#### **Torch Compile State Reset** (ReflowOptunaDrivenOptimizer)
```python
def _reset_torch_compile_state(self):
    # Reset torch._dynamo cache
    if hasattr(torch, '_dynamo'):
        torch._dynamo.reset()
    
    # Reset config to defaults
    # Clear CUDA cache and RNG state
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.manual_seed_all(torch.initial_seed())
```

### 6.3 Trial Failure Handling

#### **Exception Handling in Objective**
```python
try:
    result = reflow.fit()  # or trainer.fit()
    return metric_value
except optuna.TrialPruned:
    # Expected - trial was pruned by pruner
    # WandB cleanup, then re-raise
    raise
except Exception as e:
    # Unexpected error
    # Log, cleanup WandB
    # Return worst possible value (not pruned)
    return float('inf')  # if minimize, else float('-inf')
```

#### **PruneOnExceptionCallback**
- Catches non-KeyboardInterrupt exceptions
- Converts to TrialPruned to free resources quickly
- Preserves KeyboardInterrupt for user interruption

---

## 7. Extension Points and Plugin Mechanisms

### 7.1 Search Space Extension

#### **Custom Conditional Space**
```python
class MySearchSpace(ConditionalSearchSpace):
    def suggest_params(self, trial):
        # Define conditional logic
        return {...}
    
    @property
    def param_names(self):
        return [...]
```

#### **Dynamic Space Adaptation**
```python
space = DynamicSearchSpace(initial_params)
# After some trials, adapt the search space:
space.update_search_space({"new_param": ("uniform", 0, 1)})
space.narrow_search_space("learning_rate", factor=0.5)
```

### 7.2 Callback Extension

#### **Custom Optuna Callback**
```python
from lightning.pytorch.callbacks import Callback
import optuna

class MyCallback(Callback):
    def __init__(self, trial: optuna.Trial):
        self.trial = trial
    
    def on_validation_end(self, trainer, pl_module):
        value = trainer.callback_metrics["val_loss"].item()
        self.trial.report(value, trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
```

### 7.3 Sampler/Pruner Composition

#### **Direct Dependency Injection**
```python
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import HyperbandPruner

sampler = TPESampler(n_startup_trials=20, seed=42)
pruner = HyperbandPruner(min_resource=1, max_resource=100)

optimizer = PausibleOptunaOptimizer(
    base_config=...,
    search_space=...,
    sampler=sampler,  # Direct injection
    pruner=pruner,    # Direct injection
    ...
)
```

### 7.4 Config Overrides System

#### **Layered Overrides**
1. **Base Config** - From file (YAML/JSON) or dict
2. **Override Config** - Additional layer merged on top
3. **Config Overrides Dict** - Dotted-notation updates from CLI/search space
4. **Search Space Suggestions** - Per-trial hyperparameter suggestions

#### **Override Points**
- `PausibleOptunaOptimizer.__init__()` - override_config merging
- `HPORunner._merge_configs()` - Base + override config merging
- `HPORunner._build_config_overrides()` - CLI args → dotted updates
- `OptunaDrivenOptimizer.create_objective()` - Apply overrides in sequence

### 7.5 CLI Extension (HPORunner)

#### **Additional CLI Arguments**
```python
runner = HPORunner(
    ...,
    additional_cli_args={
        'my_arg': {
            'type': str,
            'default': 'value',
            'help': 'My custom argument'
        }
    }
)
```

#### **Dot-Notation Arguments**
CLI supports arbitrary config overrides:
```bash
python train_hpo.py --model.init_args.dropout 0.2 --data.init_args.batch_size 64
```

---

## 8. Current Coupling Between Components

### 8.1 Strong Couplings

#### **PausibleOptunaOptimizer ↔ OptunaDrivenOptimizer**
- PausibleOptunaOptimizer creates and calls OptunaDrivenOptimizer
- Directly accesses: `optimizer.create_objective()`
- Shares: Study object, sampler, pruner, config_overrides
- **Why:** Single-trial abstraction requires objective creation

#### **HPORunner ↔ PausibleOptunaOptimizer**
- HPORunner creates PausibleOptunaOptimizer with resolved config
- Passes: config_overrides from CLI, callbacks, args
- Retrieves: study object for results
- **Why:** CLI argument handling and resume logic must precede optimizer creation

#### **OptunaDrivenOptimizer ↔ Optuna**
- Direct use of optuna.Trial, optuna.Study, samplers, pruners
- No abstraction layer
- **Why:** Clean dependency injection philosophy

#### **ReflowOptunaDrivenOptimizer ↔ LightningReflow**
- Conditional import (fallback to vanilla Lightning)
- Passes all model/data/trainer configs to Reflow
- **Why:** Enables torch.compile and environment configuration

### 8.2 Weak Couplings

#### **PausibleOptunaOptimizer ↔ WandB**
- Optional - if wandb_project is None, no WandB functionality
- Uses persistence module functions for upload/download
- **Why:** WandB is optional for local-only workflows

#### **PausibleOptunaOptimizer ↔ Keyboard Handler**
- Loose coupling via interface: `keyboard_handler.get_key()`
- Can inject different implementations
- **Why:** Cross-platform and testability

#### **Callbacks ↔ Trial**
- Each callback receives trial object
- Callbacks don't need to know about optimizer or runner
- **Why:** Composable, Lightning-standard callback interface

### 8.3 Architectural Debt and Limitations

#### **Issue: Config Instantiation in Objective**
- Each trial instantiates model/datamodule from config
- Deep copy and dotted update of configs on every trial
- **Impact:** Slower trials, memory overhead
- **Potential Fix:** Pre-instantiation factory or lazy evaluation

#### **Issue: Study Object Serialization**
- Entire study pickled for checkpointing
- Large memory footprint for long optimizations
- **Impact:** Slower WandB upload, larger disk usage
- **Potential Fix:** Custom study serialization (trials only, not full state)

#### **Issue: Bidirectional PausibleOptunaOptimizer Dependencies**
- Has `underlying_optimizer` attribute for test injection
- Tests patch `create_objective()` method
- **Impact:** Less composable, harder to test in isolation
- **Potential Fix:** Dependency injection of objective factory

#### **Issue: HPORunner ↔ LightningTune Import Cycle Risk**
- HPORunner imports PausibleOptunaOptimizer
- Could form import cycle if callbacks import HPORunner
- **Current:** No cycle because callbacks don't import HPORunner
- **Fragile:** Adding new callbacks requires care

---

## 9. Design Patterns and Best Practices

### 9.1 Patterns Used

1. **Dependency Injection**
   - Samplers/pruners injected directly
   - Config merged rather than constructed
   - Callbacks registered with optimizer

2. **Factory Pattern**
   - `create_sampler()`, `create_pruner()` - Named factories
   - `build_optuna_callbacks()` - Callback composition
   - `create_wandb_logger()` - Logger factory

3. **Strategy Pattern**
   - Different samplers (TPE, Random, CMA-ES, BoTorch)
   - Different pruners (Median, Hyperband, etc.)
   - Vanilla Lightning vs. Reflow

4. **Adapter Pattern**
   - OptunaDrivenOptimizer adapts Optuna to Lightning
   - ReflowOptunaDrivenOptimizer adapts LightningReflow
   - KeyboardMonitor adapts cross-platform keyboard input

5. **Template Method**
   - `create_objective()` defines trial template
   - Search space suggests parameters
   - Model/datamodule instantiation follows standard pattern

6. **Decorator/Wrapper**
   - PausibleOptunaOptimizer wraps optimizer with pause logic
   - HPORunner wraps PausibleOptunaOptimizer with CLI logic
   - Keyboard handler wraps input reading with state management

7. **State Machine**
   - Study states: WAITING → RUNNING → COMPLETE|PRUNED|FAIL
   - Pause state: requested → pausing → paused
   - Resume state: checkpoint loaded → restored → resumed

### 9.2 Configuration Best Practices

1. **Layered Configuration**
   - Base config + override config + CLI args
   - Each layer clearly defined and traceable
   - Persistent config stored for resume

2. **Dotted Notation**
   - Supports arbitrary nesting depth
   - Compatible with Lightning CLI format
   - Clear, human-readable overrides

3. **Type Safety**
   - Config files validated at load time
   - Type conversion in auto_convert_type()
   - Error messages for invalid types

### 9.3 Resource Management Best Practices

1. **Clean Trial Boundaries**
   - Only save at trial end, not mid-training
   - Resource cleanup between trials
   - No partial states in checkpoints

2. **Dual Checkpointing**
   - Local filesystem for speed
   - WandB for versioning and sharing
   - Fallback chain: local → WandB → generic

3. **Memory Management**
   - Explicit garbage collection passes
   - CUDA cache clearing + synchronization
   - Peak memory stat resets (prevent accumulation)
   - DataLoader lifecycle management via DataModule.teardown()

---

## 10. Current Limitations and Architectural Concerns

### 10.1 Serialization and Persistence

**Problem:** Pickled study objects become large
- Entire study history, all trial data, sampler state
- Slows down WandB upload, increases storage
- No incremental checkpointing

**Potential Solutions:**
- Delta-based checkpointing (new trials only)
- Custom study serialization format
- Separate trial history from sampler state

### 10.2 Config Instantiation

**Problem:** Configs are deep-copied and merged on every trial
- Performance impact for large configs
- Memory overhead from duplicates
- No lazy evaluation

**Potential Solutions:**
- Pre-compute merged config once
- Use config as template, only instantiate differences
- Lazy instantiation of model/datamodule

### 10.3 Trial-Level Isolation

**Problem:** Subprocess isolation not enforced
- All trials run in same process
- Potential memory leaks if model has side effects
- State pollution between trials possible

**Current Workaround:** Manual cleanup calls and config reset

**Better Solution:**
- Optional subprocess isolation per trial (via spawn/fork)
- But adds complexity and overhead

### 10.4 Error Recovery

**Problem:** Failed trials return worst possible value
- No distinction between "bad hyperparams" and "crashed"
- Pruners may select regions with systematic failures
- User can't easily identify crash causes

**Potential Solutions:**
- Trial user attributes for failure metadata
- Automatic crash pattern detection
- Graceful degradation (skip trial, try next)

### 10.5 Testing Testability

**Problem:** PausibleOptunaOptimizer has test injection hooks
- `underlying_optimizer` attribute for injection
- `create_objective()` method patch point
- Not ideal for unit testing

**Better Approach:**
- Explicit dependency injection of objective factory
- Cleaner test seams
- Easier mock/stub injection

---

## 11. Key Files and Their Roles

| File | Purpose | Key Exports |
|------|---------|-------------|
| `hpo_runner.py` | CLI interface and argument handling | `HPORunner` |
| `optuna/pausible_optimizer.py` | Trial loop and pause management | `PausibleOptunaOptimizer` |
| `optuna/optimizer.py` | Single trial objective | `OptunaDrivenOptimizer` |
| `optuna/optimizer_reflow.py` | Reflow integration | `ReflowOptunaDrivenOptimizer` |
| `optuna/callbacks.py` | Trial callbacks (pruning, checkpoints) | Various callback classes |
| `optuna/search_space.py` | Search space abstractions | `OptunaSearchSpace`, implementations |
| `optuna/factories.py` | Named factories for samplers/pruners | `create_sampler()`, `create_pruner()` |
| `optuna/keyboard_monitor.py` | Cross-platform keyboard input | `KeyboardMonitor` |
| `optuna/memory_cleanup.py` | Resource management utilities | `cleanup_trial_resources()` |
| `optuna/callback_factory.py` | Standard callback composition | `build_optuna_callbacks()` |
| `optuna/nan_detection_callback.py` | NaN/Inf detection and early termination | Enhanced pruning callbacks |
| `persistence.py` | WandB/local checkpoint I/O | Save/load functions |
| `arg_persistence.py` | CLI argument extraction and merging | Argument utilities |
| `utils/config_utils.py` | Config loading, merging, dotted updates | `apply_dotted_updates()`, etc. |
| `utils/torch_compile.py` | Torch compilation mode settings | Compile utilities |
| `utils/wandb_logger.py` | WandB logger factory | `create_wandb_logger()` |

---

## 12. Recommended Architecture Improvements

### 12.1 High Priority

1. **Objective Factory Injection**
   - Replace method patching with explicit factory injection
   - Improves testability and composability
   - One-line change in most cases

2. **Incremental Checkpointing**
   - Only save new trials since last checkpoint
   - Significant speedup for long optimizations
   - Requires custom study serialization

3. **Trial Metadata Enrichment**
   - Add failure reasons to trial user_attrs
   - Track crash patterns
   - Enable crash-aware sampling

### 12.2 Medium Priority

1. **Config Pre-merging**
   - Merge configs once, reuse across trials
   - Template/diff-based instantiation
   - ~10-20% trial speedup potential

2. **Error Classification**
   - Distinguish OOM, crash, timeout, etc.
   - Per-error-type callbacks
   - Better failure diagnostics

3. **Resource Profiling**
   - Per-trial memory/GPU usage tracking
   - Warnings for increasing memory
   - Auto-detection of resource leaks

### 12.3 Low Priority

1. **Subprocess Isolation** (optional, adds complexity)
2. **Distributed Study Support** (already supported via storage URL)
3. **Custom Samplers** (extensible via dependency injection)

---

## Conclusion

LightningTune achieves a clean, extensible architecture through:
- **Minimal abstraction** - Direct Optuna dependency injection
- **Clear separation of concerns** - CLI, optimizer, objective, training
- **Production readiness** - Checkpointing, pause/resume, resource management
- **Reflow integration** - Enabling advanced features like torch.compile
- **Plugin extensibility** - Via callbacks, search spaces, samplers, pruners

The main limitations stem from **serialization overhead** and **config instantiation pattern**, both solvable with targeted refactoring without architectural changes. The codebase demonstrates good design practices and is well-positioned for production HPO workflows.

