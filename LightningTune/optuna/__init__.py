"""
Optuna-based optimization for PyTorch Lightning.

This module provides a modern, maintainable alternative to Ray Tune
for hyperparameter optimization with PyTorch Lightning.
"""

from .optimizer import OptunaDrivenOptimizer
from .strategies import (
    OptunaStrategy,
    BOHBStrategy,
    TPEStrategy,
    RandomStrategy,
    GridStrategy,
)
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

__all__ = [
    # Core optimizer
    "OptunaDrivenOptimizer",
    # Strategies
    "OptunaStrategy",
    "BOHBStrategy",
    "TPEStrategy", 
    "RandomStrategy",
    "GridStrategy",
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
]