"""
Optuna-based optimization for PyTorch Lightning.

This module provides a modern, maintainable alternative to Ray Tune
for hyperparameter optimization with PyTorch Lightning.

Direct dependency injection is used - no unnecessary abstractions.
Just pass Optuna's samplers and pruners directly to the optimizer.
"""

from .optimizer import OptunaDrivenOptimizer
from .optimizer_reflow import ReflowOptunaDrivenOptimizer
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

# Import NaN detection callbacks if available
try:
    from .nan_detection_callback import (
        NaNDetectionCallback,
        EnhancedOptunaPruningCallback,
    )
except ImportError:
    # NaN detection not available
    NaNDetectionCallback = None
    EnhancedOptunaPruningCallback = None
from .wandb_integration import (
    WandBOptunaOptimizer,
    save_optuna_session,
    load_optuna_session,
)
from .factories import (
    create_sampler,
    create_pruner,
    get_sampler_info,
    get_pruner_info,
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
    "ReflowOptunaDrivenOptimizer",
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
    "NaNDetectionCallback",
    "EnhancedOptunaPruningCallback",
    # WandB integration
    "WandBOptunaOptimizer",
    "save_optuna_session",
    "load_optuna_session",
    # Factory functions
    "create_sampler",
    "create_pruner",
    "get_sampler_info",
    "get_pruner_info",
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