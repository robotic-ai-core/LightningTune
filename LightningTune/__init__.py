"""
LightningTune: Optuna-based hyperparameter optimization for PyTorch Lightning.

This module provides modern hyperparameter optimization using Optuna
for Lightning projects with direct dependency injection, pause/resume
capabilities, and WandB integration.

No unnecessary abstractions - just use Optuna's samplers and pruners directly.
"""

import warnings

# Primary interface - Optuna-based optimizer
try:
    from .optuna.optimizer import OptunaDrivenOptimizer
    from .optuna.optimizer import OptunaDrivenOptimizer as ConfigDrivenOptimizer  # Alias for compatibility
except ImportError as e:
    warnings.warn(f"OptunaDrivenOptimizer not available: {e}")
    OptunaDrivenOptimizer = None
    ConfigDrivenOptimizer = None

# Optuna components
try:
    from .optuna.wandb_integration import WandBOptunaOptimizer
    from .optuna.search_space import (
        OptunaSearchSpace,
        SimpleSearchSpace,
        ConditionalSearchSpace,
        CompositeSearchSpace,
        DynamicSearchSpace,
    )
    from .optuna.search_space_dependent import DependentSearchSpace
    from .optuna.callbacks import (
        OptunaPruningCallback,
        OptunaCheckpointCallback,
        OptunaProgressCallback,
        OptunaEarlyStoppingCallback,
    )
    from .optuna.factories import (
        create_sampler,
        create_pruner,
        get_sampler_info,
        get_pruner_info,
    )
    # Re-export Optuna's actual components for convenience
    from optuna.samplers import (
        TPESampler,
        RandomSampler,
        GridSampler,
        CmaEsSampler,
    )
    from optuna.pruners import (
        MedianPruner,
        HyperbandPruner,
        SuccessiveHalvingPruner,
        NopPruner,
    )
except ImportError as e:
    warnings.warn(f"Optuna components not available: {e}")
    WandBOptunaOptimizer = None
    OptunaSearchSpace = None
    SimpleSearchSpace = None
    ConditionalSearchSpace = None
    CompositeSearchSpace = None
    DynamicSearchSpace = None
    DependentSearchSpace = None
    OptunaPruningCallback = None
    OptunaCheckpointCallback = None
    OptunaProgressCallback = None
    OptunaEarlyStoppingCallback = None
    TPESampler = None
    RandomSampler = None
    GridSampler = None
    CmaEsSampler = None
    MedianPruner = None
    HyperbandPruner = None
    SuccessiveHalvingPruner = None
    NopPruner = None
    create_sampler = None
    create_pruner = None
    get_sampler_info = None
    get_pruner_info = None

__version__ = "0.4.0"  # Bumped version for major refactor

__all__ = [
    # Main interface
    "ConfigDrivenOptimizer",  # Alias for backward compatibility
    "OptunaDrivenOptimizer",
    "WandBOptunaOptimizer",
    
    # Search spaces
    "OptunaSearchSpace",
    "SimpleSearchSpace",
    "ConditionalSearchSpace",
    "CompositeSearchSpace",
    "DynamicSearchSpace",
    "DependentSearchSpace",
    
    # Callbacks
    "OptunaPruningCallback",
    "OptunaCheckpointCallback",
    "OptunaProgressCallback",
    "OptunaEarlyStoppingCallback",
    
    # Factory functions
    "create_sampler",
    "create_pruner",
    "get_sampler_info",
    "get_pruner_info",
    
    # Optuna Samplers (actual Optuna components)
    "TPESampler",
    "RandomSampler",
    "GridSampler",
    "CmaEsSampler",
    
    # Optuna Pruners (actual Optuna components)
    "MedianPruner",
    "HyperbandPruner",
    "SuccessiveHalvingPruner",
    "NopPruner",
]