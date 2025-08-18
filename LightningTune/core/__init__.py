"""
Core components for Lightning hyperparameter optimization.

DEPRECATION NOTICE: The Ray Tune-based core optimizer has been deprecated.
Please migrate to the Optuna-based optimizer for better performance and features.
"""

import warnings

# Import deprecated optimizer (raises helpful error)
try:
    from .deprecated_optimizer import DeprecatedConfigDrivenOptimizer as ConfigDrivenOptimizer
except ImportError as e:
    warnings.warn(f"Deprecated optimizer not available: {e}")
    ConfigDrivenOptimizer = None

# Import strategies
try:
    from .strategies import (
        OptimizationStrategy,
        OptimizationConfig,
        BOHBStrategy,
        OptunaStrategy,
        ASHAStrategy,
        RandomSearchStrategy,
        GridSearchStrategy,
        PBTStrategy,
    )
except ImportError as e:
    warnings.warn(f"Strategies not available: {e}")
    OptimizationStrategy = None
    OptimizationConfig = None
    BOHBStrategy = None
    OptunaStrategy = None
    ASHAStrategy = None
    RandomSearchStrategy = None
    GridSearchStrategy = None
    PBTStrategy = None

# Import config management
try:
    from .config import SearchSpace, ConfigManager
except ImportError as e:
    warnings.warn(f"Config classes not available: {e}")
    SearchSpace = None
    ConfigManager = None

# Import trainable
try:
    from .trainable import LightningBOHBTrainable
except ImportError as e:
    warnings.warn(f"LightningBOHBTrainable not available: {e}")
    LightningBOHBTrainable = None

__all__ = [
    "ConfigDrivenOptimizer",
    "OptimizationStrategy",
    "OptimizationConfig",
    "BOHBStrategy",
    "OptunaStrategy",
    "ASHAStrategy",
    "RandomSearchStrategy",
    "GridSearchStrategy",
    "PBTStrategy",
    "SearchSpace",
    "ConfigManager",
    "LightningBOHBTrainable",
]