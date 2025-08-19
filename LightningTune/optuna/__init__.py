"""
Optuna-based optimization for PyTorch Lightning.

This module provides a modern, maintainable alternative to Ray Tune
for hyperparameter optimization with PyTorch Lightning.

Direct dependency injection is used - no unnecessary abstractions.
Just pass Optuna's samplers and pruners directly to the optimizer.
"""

from .optimizer import OptunaDrivenOptimizer
from .search_space import (
    OptunaSearchSpace,
    SimpleSearchSpace,
    ConditionalSearchSpace,
    CompositeSearchSpace,
    DynamicSearchSpace,
)
from .callbacks import (
    OptunaPruningCallback,
    OptunaCheckpointCallback,
    OptunaProgressCallback,
    OptunaEarlyStoppingCallback,
)
from .wandb_integration import (
    WandBOptunaOptimizer,
    save_optuna_session,
    load_optuna_session,
)

# Import Optuna components for convenience
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

__all__ = [
    # Core optimizer
    "OptunaDrivenOptimizer",
    # Search spaces
    "OptunaSearchSpace",
    "SimpleSearchSpace",
    "ConditionalSearchSpace",
    "CompositeSearchSpace",
    "DynamicSearchSpace",
    # Callbacks
    "OptunaPruningCallback",
    "OptunaCheckpointCallback",
    "OptunaProgressCallback",
    "OptunaEarlyStoppingCallback",
    # WandB integration
    "WandBOptunaOptimizer",
    "save_optuna_session",
    "load_optuna_session",
    # Optuna samplers (for convenience)
    "TPESampler",
    "RandomSampler",
    "GridSampler",
    "CmaEsSampler",
    # Optuna pruners (for convenience)
    "MedianPruner",
    "HyperbandPruner",
    "SuccessiveHalvingPruner",
    "NopPruner",
]