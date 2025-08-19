"""
Search space with dependent parameters support.

This module provides a search space that can handle parameters
that depend on each other (e.g., dimensions that must match).
"""

from typing import Dict, Any, List, Tuple
import optuna
from .search_space import OptunaSearchSpace


class DependentSearchSpace(OptunaSearchSpace):
    """
    Search space that handles dependent parameters.
    
    Example:
        # Define latent_dim once and use it for both transformer and adapter
        search_space = DependentSearchSpace({
            "model.init_args.learning_rate": ("loguniform", 1e-5, 1e-3),
            "latent_dim": ("categorical", [62, 126, 190, 254, 318]),
        }, dependencies={
            "model.init_args.transformer_hparams.latent_dim": "latent_dim",
            "model.init_args.adapter_hparams.internal_latent_dim": "latent_dim",
        })
    """
    
    def __init__(
        self, 
        param_dict: Dict[str, tuple],
        dependencies: Dict[str, str] = None
    ):
        """
        Initialize search space with dependencies.
        
        Args:
            param_dict: Dictionary of parameters to optimize
            dependencies: Dict mapping dependent param names to their source param
        """
        self.param_dict = param_dict
        self.dependencies = dependencies or {}
    
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters, handling dependencies."""
        params = {}
        
        # First, suggest all primary parameters
        for name, spec in self.param_dict.items():
            param_type = spec[0]
            
            if param_type == "uniform":
                value = trial.suggest_float(name, spec[1], spec[2])
            elif param_type == "loguniform":
                value = trial.suggest_float(name, spec[1], spec[2], log=True)
            elif param_type == "int":
                if len(spec) == 3:
                    value = trial.suggest_int(name, spec[1], spec[2])
                else:
                    value = trial.suggest_int(name, spec[1], spec[2], step=spec[3])
            elif param_type == "categorical":
                value = trial.suggest_categorical(name, spec[1])
            elif param_type == "discrete_uniform":
                value = trial.suggest_discrete_uniform(name, spec[1], spec[2], spec[3])
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
            
            params[name] = value
        
        # Then, apply dependencies
        for dependent_name, source_name in self.dependencies.items():
            if source_name in params:
                params[dependent_name] = params[source_name]
            else:
                raise ValueError(f"Dependency source '{source_name}' not found in params")
        
        # Remove intermediate parameters that were only used for dependencies
        # (keep only params that start with model., data., trainer., etc.)
        final_params = {}
        for key, value in params.items():
            if "." in key or key not in self.dependencies.values():
                final_params[key] = value
        
        return final_params
    
    @property
    def param_names(self) -> List[str]:
        """Return list of all parameter names including dependencies."""
        names = list(self.param_dict.keys())
        names.extend(self.dependencies.keys())
        return names