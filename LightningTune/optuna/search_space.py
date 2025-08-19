"""
Search space abstractions for Optuna optimization.

This module provides a clean abstraction for defining search spaces
that can be used with Optuna trials.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional
import optuna


class OptunaSearchSpace(ABC):
    """Abstract base class for defining Optuna search spaces."""
    
    @abstractmethod
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        pass
    
    @property
    @abstractmethod
    def param_names(self) -> List[str]:
        """Return list of parameter names in the search space."""
        pass
    
    def get_config_updates(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Get configuration updates from trial suggestions.
        
        This method calls suggest_params and formats the results
        for merging with base configuration.
        """
        return self.suggest_params(trial)


class SimpleSearchSpace(OptunaSearchSpace):
    """
    Simple search space defined by a dictionary of parameter ranges.
    
    Supported parameter types:
        - ("uniform", low, high): Uniform float distribution
        - ("loguniform", low, high): Log-uniform float distribution
        - ("int", low, high): Integer range
        - ("int", low, high, step): Integer range with step
        - ("categorical", [choices]): Categorical choices
        - ("discrete_uniform", low, high, q): Discrete uniform with step q
    
    Example:
        search_space = SimpleSearchSpace({
            "model.init_args.learning_rate": ("loguniform", 1e-5, 1e-3),
            "model.init_args.dropout": ("uniform", 0.1, 0.5),
            "model.init_args.hidden_size": ("categorical", [128, 256, 512]),
            "data.init_args.batch_size": ("int", 16, 128, 8),  # min, max, step
        })
    """
    
    def __init__(self, param_dict: Dict[str, tuple]):
        """
        Initialize search space with parameter dictionary.
        
        Args:
            param_dict: Dictionary mapping parameter names to tuples defining
                       the parameter type and range/choices
        """
        self.param_dict = param_dict
    
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters based on the defined ranges."""
        params = {}
        
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
        
        return params
    
    @property
    def param_names(self) -> List[str]:
        """Return list of parameter names."""
        return list(self.param_dict.keys())


class ConditionalSearchSpace(OptunaSearchSpace):
    """
    Search space with conditional parameters.
    
    Allows defining parameters that depend on values of other parameters.
    
    Example:
        class MySearchSpace(ConditionalSearchSpace):
            def suggest_params(self, trial):
                params = {}
                
                # Choose architecture
                arch = trial.suggest_categorical("model.architecture", ["cnn", "transformer"])
                params["model.architecture"] = arch
                
                # Conditional parameters based on architecture
                if arch == "cnn":
                    params["model.num_filters"] = trial.suggest_int("model.num_filters", 32, 256)
                    params["model.kernel_size"] = trial.suggest_int("model.kernel_size", 3, 7, step=2)
                else:  # transformer
                    params["model.num_heads"] = trial.suggest_int("model.num_heads", 4, 16)
                    params["model.hidden_dim"] = trial.suggest_int("model.hidden_dim", 128, 512)
                
                return params
    """
    
    @property
    def param_names(self) -> List[str]:
        """
        Return list of all possible parameter names.
        
        Note: For conditional spaces, this returns all possible parameters
        even if not all are used in every trial.
        """
        # This should be overridden in subclasses
        return []


class CompositeSearchSpace(OptunaSearchSpace):
    """
    Composite search space combining multiple search spaces.
    
    Useful for modular configuration where different components
    have their own search spaces.
    
    Example:
        model_space = SimpleSearchSpace({...})
        data_space = SimpleSearchSpace({...})
        trainer_space = SimpleSearchSpace({...})
        
        composite = CompositeSearchSpace({
            "model": model_space,
            "data": data_space,
            "trainer": trainer_space,
        })
    """
    
    def __init__(self, spaces: Dict[str, OptunaSearchSpace]):
        """
        Initialize composite search space.
        
        Args:
            spaces: Dictionary mapping prefixes to search spaces
        """
        self.spaces = spaces
    
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters from all component spaces."""
        params = {}
        
        for prefix, space in self.spaces.items():
            space_params = space.suggest_params(trial)
            
            # Add prefix to parameter names if not already present
            for key, value in space_params.items():
                if not key.startswith(prefix + "."):
                    full_key = f"{prefix}.{key}"
                else:
                    full_key = key
                params[full_key] = value
        
        return params
    
    @property
    def param_names(self) -> List[str]:
        """Return list of all parameter names from all spaces."""
        names = []
        for prefix, space in self.spaces.items():
            for name in space.param_names:
                if not name.startswith(prefix + "."):
                    names.append(f"{prefix}.{name}")
                else:
                    names.append(name)
        return names


class DynamicSearchSpace(OptunaSearchSpace):
    """
    Dynamic search space that can be modified during optimization.
    
    Useful for adaptive optimization strategies where the search
    space changes based on results.
    """
    
    def __init__(self, initial_params: Dict[str, tuple]):
        """Initialize with initial parameter definitions."""
        self.current_params = initial_params
        self.history = []
    
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters from current search space."""
        # Use SimpleSearchSpace logic for current parameters
        temp_space = SimpleSearchSpace(self.current_params)
        params = temp_space.suggest_params(trial)
        
        # Record history
        self.history.append({
            "trial": trial.number,
            "params": params.copy()
        })
        
        return params
    
    def update_search_space(self, updates: Dict[str, tuple]) -> None:
        """
        Update the search space with new parameter definitions.
        
        Args:
            updates: Dictionary of parameter updates
        """
        self.current_params.update(updates)
    
    def narrow_search_space(self, param: str, factor: float = 0.5) -> None:
        """
        Narrow the search range for a parameter.
        
        Args:
            param: Parameter name
            factor: Factor to narrow the range (0.5 = half the range)
        """
        if param not in self.current_params:
            raise ValueError(f"Parameter {param} not in search space")
        
        spec = self.current_params[param]
        if spec[0] in ["uniform", "loguniform", "discrete_uniform"]:
            low, high = spec[1], spec[2]
            center = (low + high) / 2
            new_range = (high - low) * factor / 2
            self.current_params[param] = (spec[0], center - new_range, center + new_range)
    
    @property
    def param_names(self) -> List[str]:
        """Return current parameter names."""
        return list(self.current_params.keys())