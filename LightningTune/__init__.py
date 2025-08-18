"""
LightningTune: Optuna-based hyperparameter optimization for PyTorch Lightning.

This module provides modern hyperparameter optimization using Optuna
for Lightning projects with comprehensive strategy support, pause/resume
capabilities, and WandB integration.
"""

# Import conditionally to handle missing dependencies
import warnings

# Primary interface - Optuna-based optimizer
try:
    from .optuna.optimizer import OptunaDrivenOptimizer as ConfigDrivenOptimizer
except ImportError as e:
    warnings.warn(f"OptunaDrivenOptimizer not available: {e}")
    ConfigDrivenOptimizer = None

# Optuna-specific imports
try:
    from .optuna.optimizer import OptunaDrivenOptimizer
    from .optuna.wandb_integration import WandBOptunaOptimizer
except ImportError as e:
    warnings.warn(f"Optuna optimizers not available: {e}")
    OptunaDrivenOptimizer = None
    WandBOptunaOptimizer = None

# Optuna Strategies
try:
    from .optuna.strategies import (
        OptunaStrategy,
        BOHBStrategy,
        TPEStrategy,
        RandomStrategy,
        GridStrategy,
        ASHAStrategy,
        CMAESStrategy,
    )
except ImportError as e:
    warnings.warn(f"Optuna strategies not available: {e}")
    OptunaStrategy = None
    BOHBStrategy = None
    TPEStrategy = None
    RandomStrategy = None
    GridStrategy = None
    ASHAStrategy = None
    CMAESStrategy = None

# Optuna Configuration
try:
    from .optuna.search_space import OptunaSearchSpace
except ImportError as e:
    warnings.warn(f"Optuna search space not available: {e}")
    OptunaSearchSpace = None

# Legacy configuration support (for backward compatibility)
try:
    from .core.config import (
        SearchSpace,
        StandardSearchSpace,
        ConfigManager,
    )
except ImportError as e:
    warnings.warn(f"Legacy configuration classes not available: {e}")
    SearchSpace = None
    StandardSearchSpace = None
    ConfigManager = None

# Optuna Callbacks
try:
    from .optuna.callbacks import (
        OptunaPruningCallback,
        OptunaCheckpointCallback,
    )
except ImportError as e:
    warnings.warn(f"Optuna callbacks not available: {e}")
    OptunaPruningCallback = None
    OptunaCheckpointCallback = None

# Legacy callbacks (for backward compatibility)
try:
    from .callbacks.report import (
        BOHBReportCallback,
        AdaptiveBOHBCallback,
    )
    from .callbacks.tune_pause_callback import (
        TunePauseCallback,
        TuneResumeCallback,
    )
except ImportError as e:
    warnings.warn(f"Legacy callbacks not available: {e}")
    BOHBReportCallback = None
    AdaptiveBOHBCallback = None
    TunePauseCallback = None
    TuneResumeCallback = None

# CLI - Legacy support
try:
    from .cli.tune_reflow import TuneReflowCLI
except ImportError as e:
    warnings.warn(f"TuneReflowCLI not available: {e}. Legacy CLI no longer supported after Ray Tune removal.")
    TuneReflowCLI = None

# Legacy aliases for backward compatibility
ConfigDrivenBOHBOptimizer = ConfigDrivenOptimizer  # Alias for compatibility

__version__ = "0.3.0"

__all__ = [
    # Main interface
    "ConfigDrivenOptimizer",
    "OptunaDrivenOptimizer",
    "WandBOptunaOptimizer",
    
    # Optuna Strategies
    "OptunaStrategy",
    "BOHBStrategy",
    "TPEStrategy",
    "RandomStrategy", 
    "GridStrategy",
    "ASHAStrategy",
    "CMAESStrategy",
    
    # Configuration
    "OptunaSearchSpace",
    
    # Optuna Callbacks
    "OptunaPruningCallback",
    "OptunaCheckpointCallback",
    
    # Legacy support (for backward compatibility)
    "SearchSpace",
    "StandardSearchSpace", 
    "ConfigManager",
    "BOHBReportCallback",
    "AdaptiveBOHBCallback",
    "TunePauseCallback",
    "TuneResumeCallback",
    "TuneReflowCLI",
    
    # Legacy/Aliases
    "ConfigDrivenBOHBOptimizer",
]