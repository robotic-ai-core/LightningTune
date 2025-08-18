# Migration from lightning_bohb to LightningTune

## Package Renamed
The package has been renamed from `lightning_bohb` to `LightningTune` to better reflect its use of Ray Tune as the core optimization engine.

## Repository Information
- **GitHub Repository**: https://github.com/neil-tan/LightningTune
- **Default Branch**: master
- **Location in ProtoWorld**: `/external/LightningTune` (as git submodule)

## Key Changes
1. **Package Name**: `lightning_bohb` → `LightningTune`
2. **Import Statements**: 
   ```python
   # Old
   from lightning_bohb import ConfigDrivenOptimizer
   
   # New
   from LightningTune import ConfigDrivenOptimizer
   ```
3. **Documentation**: All references updated to reflect Ray Tune as the primary optimization framework

## Features
- ✅ Multiple optimization strategies (BOHB, Optuna, Random Search, PBT, Grid Search)
- ✅ Interactive pause/resume capabilities
- ✅ Dependency injection pattern for clean strategy usage
- ✅ Complete state serialization for perfect resume
- ✅ Comprehensive test suite with pytest
- ✅ Full documentation

## Testing
Run tests with pytest:
```bash
# Basic tests (no dependencies required)
pytest tests/test_basic.py -v

# Unit tests
pytest tests/unit/test_strategies.py -v

# All tests (requires Ray and PyTorch)
pytest tests/ -v
```

## Installation
```bash
# From ProtoWorld
pip install -e external/LightningTune

# With all dependencies
pip install -e "external/LightningTune[full]"

# For development
pip install -e "external/LightningTune[full,test,dev]"
```

## Quick Start
```python
from LightningTune import ConfigDrivenOptimizer, SearchSpace
from LightningTune.core.strategies_v2 import BOHBStrategy

# Define search space
class MySearchSpace(SearchSpace):
    def get_search_space(self):
        return {
            "learning_rate": [0.001, 0.01, 0.1],
            "batch_size": [16, 32, 64],
        }
    
    def get_metric_config(self):
        return {"metric": "val_loss", "mode": "min"}

# Create strategy
strategy = BOHBStrategy(grace_period=10, reduction_factor=3)

# Run optimization
optimizer = ConfigDrivenOptimizer(
    base_config_source="config.yaml",
    search_space=MySearchSpace(),
    strategy=strategy,
)

results = optimizer.run()
```

## Submodule Management
```bash
# Update submodule
git submodule update --remote external/LightningTune

# Clone ProtoWorld with submodules
git clone --recursive git@github.com:your-repo/ProtoWorld.git

# Initialize submodules after clone
git submodule update --init --recursive
```