# LightningTune Installation Guide

## Basic Installation

Install LightningTune with core dependencies:

```bash
pip install -e external/LightningTune
```

This includes:
- ✅ Optuna (TPE, Random, CMA-ES samplers)
- ✅ optuna-integration (base package)
- ✅ PyTorch Lightning
- ✅ YAML configuration support

## Installation Options

### 1. With BoTorch Support (Recommended for expensive HPO)

```bash
pip install -e "external/LightningTune[botorch]"
```

This adds:
- ✅ BoTorchSampler for Gaussian Process-based Bayesian Optimization
- ✅ GPyTorch for GP computations
- Best for: Expensive evaluations (>10 min/trial) with continuous parameters

### 2. Full Installation

```bash
pip install -e "external/LightningTune[full]"
```

Includes everything plus:
- ✅ WandB for experiment tracking
- ✅ Plotly for visualization
- ✅ BoTorch and GPyTorch
- ✅ scikit-optimize compatibility

### 3. Development Installation

```bash
pip install -e "external/LightningTune[dev,test]"
```

Adds development tools:
- ✅ pytest for testing
- ✅ black, isort for formatting
- ✅ flake8, mypy for linting

## Sampler Availability

| Sampler | Basic Install | With [botorch] | Use Case |
|---------|--------------|----------------|----------|
| TPE | ✅ | ✅ | General purpose (default) |
| Random | ✅ | ✅ | Baseline, parallel trials |
| CMA-ES | ✅ | ✅ | Continuous parameters |
| Grid | ✅ | ✅ | Exhaustive search |
| BoTorch | ❌ | ✅ | Expensive evaluations |

## Verifying Installation

Check available samplers:

```python
from LightningTune.optuna.factories import get_sampler_info

for name, description in get_sampler_info().items():
    print(f"{name}: {description}")
```

## Troubleshooting

### BoTorch Not Available

If BoTorch sampler isn't showing up:

```bash
# Install BoTorch dependencies explicitly
pip install botorch gpytorch

# Or reinstall with extras
pip install -e "external/LightningTune[botorch]"
```

### CUDA Compatibility

BoTorch requires PyTorch with CUDA support for GPU acceleration:

```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install BoTorch
pip install botorch gpytorch
```

### Import Errors

If you get import errors for optuna-integration:

```bash
# Ensure optuna-integration is installed
pip install optuna-integration>=3.0.0

# For specific integrations
pip install optuna-integration[botorch]  # Just BoTorch
```

## Minimal Requirements

- Python >= 3.8
- PyTorch >= 1.11 (for BoTorch)
- CUDA >= 11.3 (optional, for GPU acceleration)

## Usage After Installation

### Basic (TPE)
```bash
python scripts/world_model_hpo_optuna.py --sampler tpe
```

### With BoTorch (after installing [botorch])
```bash
python scripts/world_model_hpo_optuna.py --sampler botorch
```