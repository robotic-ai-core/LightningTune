# LightningTune Migration Plan: Ray Tune → Optuna

## Executive Summary
Ray Tune's BOHB integration is broken with modern Python ML stacks (2025). ConfigSpace compatibility issues and lack of maintenance make it unsuitable for production use. Optuna is actively maintained, has better integration with modern ML frameworks, and provides similar functionality with a cleaner API.

## Why Migrate to Optuna?

### Problems with Ray Tune (as of 2025)
1. **Broken BOHB Integration**: ConfigSpace version conflicts (requires ancient 0.6.x, incompatible with numpy 2.x)
2. **Deprecated APIs**: Multiple deprecation warnings in core functionality
3. **Slow Development**: Ray Tune appears to have slowed development, focusing more on Ray Train
4. **Complex Dependency Chain**: Ray → Ray Tune → ConfigSpace → HpBandSter creates fragile dependencies
5. **Heavy Runtime**: Ray requires cluster management overhead even for single-machine HPO

### Advantages of Optuna
1. **Active Development**: Regular releases, modern Python support
2. **Native BOHB Support**: Built-in Successive Halving and Hyperband pruners
3. **Lightweight**: No cluster management overhead for single-machine use
4. **Better Integration**: First-class PyTorch Lightning support
5. **Simpler API**: More Pythonic, easier to debug
6. **Storage Backends**: SQLite, PostgreSQL, MySQL for experiment tracking
7. **Visualization**: Built-in visualization tools

## Feature Parity Analysis

### Current LightningTune Features (Ray Tune based)
- [x] Multiple optimization strategies (BOHB, ASHA, Random, Grid, PBT, Optuna bridge)
- [x] Config-driven optimization
- [x] PyTorch Lightning integration
- [x] Pause/resume capability
- [x] Distributed training support
- [x] Search space abstraction
- [x] Metric tracking and reporting
- [x] Checkpoint management
- [x] Trial resource allocation

### Optuna Equivalents
| Ray Tune Feature | Optuna Equivalent | Implementation Effort |
|-----------------|-------------------|----------------------|
| BOHB | `SuccessiveHalvingPruner` + `TPESampler` | Low |
| ASHA | `SuccessiveHalvingPruner` | Low |
| Random Search | `RandomSampler` | Low |
| Grid Search | `GridSampler` | Low |
| PBT | `PopulationBasedTraining` (contrib) | Medium |
| Hyperopt/Tree-structured Parzen | `TPESampler` (default) | Low |
| Distributed Training | Optuna + PyTorch DDP | Medium |
| Pause/Resume | Storage backend + trial state | Medium |
| Config Management | Same (YAML/dict based) | Low |
| Search Space | `suggest_*` methods | Low |

## Architecture Design

### Current Architecture (Ray Tune)
```
ConfigDrivenOptimizer
    ├── SearchSpace (abstract)
    ├── Strategy (Ray Tune specific)
    │   ├── BOHBStrategy → TuneBOHB
    │   ├── ASHAStrategy → ASHAScheduler
    │   └── OptunaStrategy → OptunaSearch
    ├── Trainable (Ray Tune Actor)
    └── Tuner (Ray Tune orchestrator)
```

### Proposed Architecture (Optuna Native)
```
ConfigDrivenOptimizer
    ├── SearchSpace (abstract)
    ├── Strategy (Optuna native)
    │   ├── BOHBStrategy → SuccessiveHalvingPruner
    │   ├── TPEStrategy → TPESampler
    │   └── RandomStrategy → RandomSampler
    ├── Objective (callable function)
    └── Study (Optuna orchestrator)
```

## Migration Strategy

### Phase 1: Core Infrastructure (Week 1)
1. Create `OptunaDrivenOptimizer` alongside existing `ConfigDrivenOptimizer`
2. Implement Optuna-native strategies
3. Port search space abstractions
4. Implement objective function wrapper for PyTorch Lightning

### Phase 2: Feature Parity (Week 2)
1. Implement pause/resume with storage backends
2. Port metric tracking and reporting
3. Implement checkpoint management
4. Add distributed training support

### Phase 3: Testing & Validation (Week 3)
1. Port existing tests to Optuna
2. Validate against benchmark problems
3. Performance comparison with Ray Tune
4. Documentation updates

### Phase 4: Gradual Deprecation
1. Mark Ray Tune components as deprecated
2. Provide migration guide for users
3. Support both backends temporarily
4. Remove Ray Tune after grace period

## Implementation Details

### 1. Search Space Translation
```python
# Ray Tune style
search_space = {
    "lr": tune.loguniform(1e-5, 1e-3),
    "batch_size": tune.choice([16, 32, 64])
}

# Optuna style
def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
```

### 2. Strategy Pattern
```python
class OptunaStrategy(ABC):
    @abstractmethod
    def create_sampler(self) -> optuna.samplers.BaseSampler:
        pass
    
    @abstractmethod
    def create_pruner(self) -> optuna.pruners.BasePruner:
        pass

class BOHBStrategy(OptunaStrategy):
    def create_sampler(self):
        return optuna.samplers.TPESampler()
    
    def create_pruner(self):
        return optuna.pruners.SuccessiveHalvingPruner()
```

### 3. PyTorch Lightning Integration
```python
class OptunaLightningObjective:
    def __call__(self, trial):
        # Create config from trial
        config = self.create_config(trial)
        
        # Initialize Lightning module
        model = self.model_class(**config["model"])
        
        # Add Optuna callback for pruning
        callbacks = [
            OptunaPruningCallback(trial, monitor="val_loss")
        ]
        
        # Train with Lightning
        trainer = pl.Trainer(callbacks=callbacks, **config["trainer"])
        trainer.fit(model)
        
        return trainer.callback_metrics["val_loss"].item()
```

### 4. Pause/Resume Implementation
```python
# Use SQLite storage for persistence
storage = "sqlite:///optuna_study.db"
study = optuna.create_study(
    study_name="world_model_hpo",
    storage=storage,
    load_if_exists=True,  # Resume if exists
    direction="minimize"
)
```

## Breaking Changes

### API Changes
1. `SearchSpace.get_search_space()` → `SearchSpace.suggest_params(trial)`
2. Strategy initialization will change
3. Results format will differ
4. Checkpoint paths will change

### Migration Guide for Users
```python
# Old (Ray Tune)
optimizer = ConfigDrivenOptimizer(
    strategy=BOHBStrategy(),
    search_space=MySearchSpace()
)
results = optimizer.run()

# New (Optuna)
optimizer = OptunaDrivenOptimizer(
    strategy=BOHBStrategy(),  # Same name, different implementation
    search_space=MySearchSpace()
)
results = optimizer.run()  # Compatible API
```

## Risk Assessment

### Low Risk
- Core optimization algorithms (well understood)
- Config management (unchanged)
- PyTorch Lightning integration (Optuna has official support)

### Medium Risk
- Distributed training (different paradigm)
- Pause/resume state management
- Performance regression for specific workloads

### Mitigation
- Maintain both backends during transition
- Extensive testing on real workloads
- Performance benchmarks before switching

## Timeline

- **Week 1**: Core implementation
- **Week 2**: Feature parity
- **Week 3**: Testing & validation
- **Week 4**: Documentation & examples
- **Month 2**: User feedback & refinement
- **Month 3**: Deprecate Ray Tune backend

## Success Metrics

1. **Functionality**: All existing features work with Optuna
2. **Performance**: No regression in optimization efficiency
3. **Reliability**: No ConfigSpace/numpy compatibility issues
4. **Simplicity**: Reduced code complexity
5. **Maintainability**: Fewer dependencies, cleaner architecture

## Next Steps

1. Create `optuna` branch ✓
2. Implement proof-of-concept with basic TPE optimization
3. Validate on world model training task
4. Get user feedback on API design
5. Proceed with full implementation