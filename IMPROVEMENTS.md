# LightningTune - Future Improvements Roadmap

This document tracks potential improvements and refactoring opportunities identified during code reviews. These are not critical issues but could improve maintainability and code quality in future iterations.

## 1. Architecture Refactoring

### 1.1 Merge Optimizer Classes (Breaking Change)
**Priority**: High  
**Complexity**: High  
**Files Affected**: 
- `LightningTune/optuna/optimizer.py`
- `LightningTune/optuna/optimizer_reflow.py`

**Current State**: 
- Two separate optimizer classes with ~80% code duplication
- `OptunaDrivenOptimizer` for standard Lightning
- `ReflowOptunaDrivenOptimizer` for LightningReflow integration
- **ReflowOptunaDrivenOptimizer is the DEFAULT in production** (world_model_hpo_optuna.py)
- Both are actively used via `PausibleOptunaOptimizer` with `use_reflow` flag

**Proposed Solution**:
```python
class OptunaDrivenOptimizer:
    def __init__(self, ..., use_reflow: bool = False):
        self.use_reflow = use_reflow
        if use_reflow:
            self._setup_reflow()
        else:
            self._setup_standard()
```

**Benefits**:
- Eliminate ~400 lines of duplicate code
- Single source of truth for optimizer logic
- Easier maintenance

**Challenges**:
- Breaking change for existing code using `ReflowOptunaDrivenOptimizer`
- Need to update all tests
- Risk of introducing regressions

**Migration Strategy**:
1. Create unified class with `use_reflow` parameter
2. Keep old classes as deprecated wrappers for 1-2 versions
3. Add deprecation warnings
4. Remove old classes in major version update

---

## 2. Dead Code Removal

### 2.1 Remove Deprecated keyboard_monitor Module
**Priority**: Low  
**Complexity**: Low  
**Files Affected**:
- `LightningTune/optuna/keyboard_monitor.py`
- Tests importing this module

**Current State**:
- Module replaced by LightningReflow's improved keyboard handler
- Still imported by tests for backward compatibility
- Contains ~200 lines of unused code

**Proposed Solution**:
1. Remove `keyboard_monitor.py`
2. Update tests to use new keyboard handler
3. Remove backward compatibility shim in `pausible_optimizer.py`

**Benefits**:
- Remove 200+ lines of dead code
- Cleaner codebase

**Challenges**:
- Need to update all affected tests
- Possible external dependencies we're not aware of

---

## 3. Code Organization

### 3.1 Consolidate Callback Modules
**Priority**: Low  
**Complexity**: Medium  
**Files Affected**:
- `LightningTune/callbacks/` directory
- `LightningTune/optuna/callbacks.py`
- `LightningTune/optuna/nan_detection_callback.py`

**Current State**:
- Callbacks split across multiple locations
- Inconsistent organization

**Proposed Solution**:
```
LightningTune/callbacks/
├── __init__.py
├── optuna/
│   ├── __init__.py
│   ├── pruning.py
│   ├── nan_detection.py
│   └── metrics.py
└── lightning/
    ├── __init__.py
    └── training.py
```

**Benefits**:
- Clear separation of concerns
- Easier to find related code
- Better import organization

**Challenges**:
- Breaking change for imports
- Need to update all references

---

## 4. Test Improvements

### 4.1 Parameterize Memory Tests
**Priority**: Medium  
**Complexity**: Low  
**Files Affected**:
- `tests/test_memory_accumulation.py`
- `tests/test_system_memory_accumulation.py`

**Current State**:
- Separate test files for different memory aspects
- Some code duplication in test setup

**Proposed Solution**:
```python
@pytest.mark.parametrize("memory_type,cleanup_func", [
    ("system", cleanup_trial_resources),
    ("gpu", torch.cuda.empty_cache),
    ("process", gc.collect),
])
def test_memory_cleanup(memory_type, cleanup_func):
    # Unified test logic
```

**Benefits**:
- Reduce test code duplication
- Easier to add new memory test scenarios
- More maintainable

---

## 5. Performance Optimizations

### 5.1 Lazy Import Heavy Dependencies
**Priority**: Medium  
**Complexity**: Low  
**Files Affected**:
- All modules importing torch, wandb, optuna

**Current State**:
- All imports at module level
- Slow initial import time

**Proposed Solution**:
```python
def get_torch():
    """Lazy import torch only when needed."""
    global _torch
    if _torch is None:
        import torch as _torch
    return _torch
```

**Benefits**:
- Faster CLI startup
- Reduced memory footprint for simple operations

---

## 6. Feature Enhancements

### 6.1 Add n_jobs Support for Parallel Trials
**Priority**: High  
**Complexity**: High  
**Files Affected**:
- `LightningTune/optuna/pausible_optimizer.py`

**Current State**:
- Only sequential trial execution
- Memory cleanup works well for sequential trials

**Proposed Solution**:
- Implement parallel trial execution using Optuna's n_jobs
- Ensure memory cleanup works with parallel execution
- Handle GPU distribution across parallel workers

**Note**: This was partially implemented in the "isolation" branch but removed for simplicity. The implementation can be referenced from git history.

---

## 7. Documentation Improvements

### 7.1 Add Architecture Diagram
**Priority**: Medium  
**Complexity**: Low  

**Proposed Content**:
- Component interaction diagram
- Data flow during optimization
- Pause/resume mechanism visualization

### 7.2 Add Migration Guide
**Priority**: High (if breaking changes implemented)  
**Complexity**: Low  

**Proposed Content**:
- Step-by-step migration from old to new APIs
- Code examples
- Common pitfalls

---

## 8. Process Isolation (Advanced)

### 8.1 Optional Process Isolation for Memory-Intensive Trials
**Priority**: Low  
**Complexity**: Very High  
**Branch Reference**: `isolation`

**Current State**:
- Simple memory cleanup is sufficient for most cases
- Process isolation implementation exists in `isolation` branch

**When Might Be Needed**:
- Trials that create global state that can't be cleaned
- C extensions with memory leaks
- Trials that modify system-wide settings

**Implementation Available In**:
```bash
git checkout isolation
# See LightningTune/optuna/command_based_trial_runner.py
# See LightningTune/optuna/gpu_manager.py
```

---

## Implementation Priority Matrix

| Priority | Complexity | Items |
|----------|------------|-------|
| High | Low | - |
| High | Medium | - |
| High | High | 6.1 n_jobs Support |
| Medium | Low | 4.1 Parameterize Tests, 5.1 Lazy Imports, 7.1 Architecture Diagram |
| Medium | Medium | - |
| Medium | High | 1.1 Merge Optimizer Classes |
| Low | Low | 2.1 Remove keyboard_monitor |
| Low | Medium | 3.1 Consolidate Callbacks |
| Low | High | 8.1 Process Isolation |

---

## Notes

- **Breaking Changes**: Items marked as breaking changes should be bundled together in a major version release
- **Testing**: All changes should maintain or improve test coverage
- **Documentation**: Update docs for any user-facing changes
- **Backward Compatibility**: Consider deprecation periods for breaking changes

---

*Last Updated: 2025-09-12*
*Generated during cleanup of memory management implementation*