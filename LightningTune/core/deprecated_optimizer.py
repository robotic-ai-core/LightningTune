"""
Deprecated Ray Tune optimizer with migration guidance.

This module provides a deprecation wrapper that guides users to migrate
from Ray Tune to Optuna-based optimization.
"""

import warnings
from typing import Dict, Any, Optional, Union, List
from pathlib import Path


class DeprecatedConfigDrivenOptimizer:
    """
    Deprecated Ray Tune-based optimizer.
    
    This class now raises a deprecation error and provides guidance
    on migrating to the new Optuna-based optimizer.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Raise deprecation error with migration guidance.
        """
        migration_message = """
        
        ========================================================
        BREAKING CHANGE: Ray Tune Support Removed
        ========================================================
        
        LightningTune has migrated from Ray Tune to Optuna for better
        performance, reliability, and maintainability.
        
        MIGRATION GUIDE:
        ================
        
        OLD (Ray Tune):
        ```python
        from LightningTune.core.optimizer import ConfigDrivenOptimizer
        
        optimizer = ConfigDrivenOptimizer(
            base_config_path="config.yaml",
            search_space=search_space,
            strategy="bohb"
        )
        ```
        
        NEW (Optuna):
        ```python
        from LightningTune import OptunaDrivenOptimizer
        from LightningTune.optuna.search_space import OptunaSearchSpace
        from LightningTune.optuna.strategies import BOHBStrategy
        
        search_space = OptunaSearchSpace()
        search_space.add_float("learning_rate", 1e-4, 1e-2, log=True)
        
        optimizer = OptunaDrivenOptimizer(
            base_config="config.yaml",
            search_space=search_space,
            model_class=YourLightningModule,
            strategy=BOHBStrategy()
        )
        ```
        
        ALTERNATIVE - Use backward-compatible interface:
        ```python
        from LightningTune import ConfigDrivenOptimizer  # Now Optuna-based
        
        # ConfigDrivenOptimizer is now an alias to OptunaDrivenOptimizer
        optimizer = ConfigDrivenOptimizer(...)
        ```
        
        KEY CHANGES:
        ============
        
        1. Optuna replaces Ray Tune for better performance
        2. New search space API with OptunaSearchSpace
        3. Better strategy system with Optuna samplers/pruners
        4. WandB integration with pause/resume capabilities
        5. Improved error handling and logging
        
        BENEFITS OF OPTUNA:
        ===================
        
        - Better optimization algorithms (TPE, CMA-ES, etc.)
        - More reliable pruning and early stopping
        - Native WandB integration
        - Easier pause/resume functionality
        - Better visualization tools
        - Lower memory footprint
        - More active development and community
        
        For detailed migration examples, see:
        - LightningTune/examples/
        - Documentation at: [URL]
        
        ========================================================
        """
        
        raise DeprecationError(migration_message)


class DeprecationError(Exception):
    """Custom exception for deprecation with migration guidance."""
    pass