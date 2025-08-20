"""
Factory functions for creating Optuna samplers and pruners.

These utilities provide convenient ways to create Optuna components by name,
with sensible defaults. Useful for CLI applications and quick experimentation.
"""

from typing import Dict, Any, Optional
import optuna
from optuna.samplers import (
    TPESampler, 
    RandomSampler, 
    GridSampler, 
    CmaEsSampler,
    BaseSampler
)
from optuna.pruners import (
    MedianPruner,
    HyperbandPruner, 
    SuccessiveHalvingPruner,
    NopPruner,
    BasePruner
)


def create_sampler(
    sampler_name: str, 
    seed: Optional[int] = None,
    **kwargs
) -> BaseSampler:
    """
    Create an Optuna sampler by name with sensible defaults.
    
    Args:
        sampler_name: Name of the sampler ('tpe', 'random', 'grid', 'cmaes')
        seed: Random seed for reproducibility
        **kwargs: Additional arguments passed to the sampler constructor
        
    Returns:
        Optuna sampler instance
        
    Examples:
        >>> sampler = create_sampler("tpe", seed=42)
        >>> sampler = create_sampler("random", seed=42)
        >>> sampler = create_sampler("cmaes", seed=42, n_startup_trials=5)
        
    Raises:
        ValueError: If sampler_name is not recognized
    """
    # Add seed if provided and not already in kwargs
    if seed is not None and 'seed' not in kwargs:
        # Not all samplers support seed, so we check
        if sampler_name in ['tpe', 'random', 'cmaes']:
            kwargs['seed'] = seed
    
    samplers = {
        "tpe": lambda: TPESampler(**kwargs),
        "random": lambda: RandomSampler(**kwargs),
        "grid": lambda: GridSampler(kwargs.pop('search_space', {}), **kwargs),
        "cmaes": lambda: CmaEsSampler(**kwargs),
    }
    
    # Add BoTorchSampler (requires optuna-integration with botorch extra)
    try:
        from optuna.integration.botorch import BoTorchSampler
        samplers["botorch"] = lambda: BoTorchSampler(**kwargs)
    except (ImportError, ModuleNotFoundError):
        # BoTorch has heavy dependencies (torch, gpytorch, botorch)
        # so we keep it optional even though optuna-integration is installed
        pass
    
    if sampler_name not in samplers:
        raise ValueError(
            f"Unknown sampler: {sampler_name}. "
            f"Choose from: {list(samplers.keys())}"
        )
    
    return samplers[sampler_name]()


def create_pruner(
    pruner_name: str,
    **kwargs
) -> BasePruner:
    """
    Create an Optuna pruner by name with sensible defaults.
    
    Args:
        pruner_name: Name of the pruner 
                    ('median', 'hyperband', 'successivehalving', 'none')
        **kwargs: Additional arguments passed to the pruner constructor
        
    Returns:
        Optuna pruner instance
        
    Examples:
        >>> pruner = create_pruner("median")  # Uses default n_warmup_steps=5
        >>> pruner = create_pruner("hyperband", max_resource=100)
        >>> pruner = create_pruner("none")  # No pruning
        
    Raises:
        ValueError: If pruner_name is not recognized
    """
    # Default parameters for each pruner
    defaults = {
        "median": {
            "n_startup_trials": 5,
            "n_warmup_steps": 5,
        },
        "hyperband": {
            "min_resource": 1,
            "max_resource": 30,
            "reduction_factor": 3,
        },
        "successivehalving": {
            "min_resource": 1,
            "reduction_factor": 3,
            # Note: SuccessiveHalvingPruner doesn't have max_resource parameter
        },
        "none": {},
    }
    
    # Merge defaults with provided kwargs (kwargs override defaults)
    if pruner_name in defaults:
        params = {**defaults[pruner_name], **kwargs}
    else:
        params = kwargs
    
    pruners = {
        "median": lambda: MedianPruner(**params),
        "hyperband": lambda: HyperbandPruner(**params),
        "successivehalving": lambda: SuccessiveHalvingPruner(**params),
        "none": lambda: NopPruner(**params),
    }
    
    if pruner_name not in pruners:
        raise ValueError(
            f"Unknown pruner: {pruner_name}. "
            f"Choose from: {list(pruners.keys())}"
        )
    
    return pruners[pruner_name]()


def get_sampler_info() -> Dict[str, str]:
    """
    Get information about available samplers.
    
    Returns:
        Dictionary mapping sampler names to descriptions
    """
    info = {
        "tpe": "Tree-structured Parzen Estimator - Good general purpose sampler",
        "random": "Random Search - Simple baseline, good for parallel execution",
        "grid": "Grid Search - Exhaustive search over discrete space",
        "cmaes": "CMA-ES - Evolution strategy, good for continuous parameters",
    }
    
    # Check if BoTorchSampler is actually available
    try:
        from optuna.integration.botorch import BoTorchSampler
        info["botorch"] = "BoTorch GP-based BO - Best for expensive evaluations with continuous params"
    except (ImportError, ModuleNotFoundError):
        pass
    
    return info


def get_pruner_info() -> Dict[str, str]:
    """
    Get information about available pruners.
    
    Returns:
        Dictionary mapping pruner names to descriptions
    """
    return {
        "median": "Median Pruner - Prunes trials below median at each step",
        "hyperband": "Hyperband - Successive halving with multiple brackets",
        "successivehalving": "Successive Halving - Aggressive early stopping",
        "none": "No Pruning - All trials run to completion",
    }