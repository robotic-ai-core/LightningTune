"""
Configuration management for Lightning BOHB optimization.

This module provides flexible configuration handling that works with any
Lightning module and data module combination.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import yaml
import json
from abc import ABC, abstractmethod

from ray import tune


class SearchSpace(ABC):
    """
    Abstract base class for defining hyperparameter search spaces.
    
    Users should inherit from this class to define their own search spaces
    in a structured, reusable way.
    """
    
    @abstractmethod
    def get_search_space(self) -> Dict[str, Any]:
        """
        Return the search space dictionary for Ray Tune.
        
        Returns:
            Dictionary mapping parameter names to tune sampling functions
        """
        pass
    
    @abstractmethod
    def get_metric_config(self) -> Dict[str, Any]:
        """
        Return metric configuration for optimization.
        
        Returns:
            Dictionary with 'metric' and 'mode' keys
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Optional validation of sampled configurations.
        
        Args:
            config: Sampled configuration from BOHB
            
        Returns:
            True if configuration is valid, False otherwise
        """
        return True
    
    def transform_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optional transformation of sampled configurations.
        
        Useful for derived parameters or complex dependencies.
        
        Args:
            config: Raw sampled configuration
            
        Returns:
            Transformed configuration
        """
        return config


class StandardSearchSpace(SearchSpace):
    """
    Standard search space with common hyperparameters for neural networks.
    
    This provides a good starting point that can be extended or modified.
    """
    
    def __init__(
        self,
        include_architecture: bool = True,
        include_optimization: bool = True,
        include_regularization: bool = True,
        custom_params: Optional[Dict[str, Any]] = None
    ):
        self.include_architecture = include_architecture
        self.include_optimization = include_optimization
        self.include_regularization = include_regularization
        self.custom_params = custom_params or {}
        
    def get_search_space(self) -> Dict[str, Any]:
        """Build the search space from components."""
        space = {}
        
        if self.include_optimization:
            space.update({
                "learning_rate": tune.loguniform(1e-5, 1e-1),
                "batch_size": tune.choice([8, 16, 32, 64]),
                "gradient_clip_val": tune.loguniform(0.1, 10.0),
                "warmup_steps": tune.choice([0, 500, 1000, 2000]),
            })
        
        if self.include_architecture:
            space.update({
                "hidden_dim": tune.choice([128, 256, 512, 1024]),
                "num_layers": tune.choice([2, 4, 6, 8]),
                "dropout": tune.uniform(0.0, 0.5),
            })
        
        if self.include_regularization:
            space.update({
                "weight_decay": tune.loguniform(1e-6, 1e-2),
                "label_smoothing": tune.uniform(0.0, 0.2),
            })
        
        # Add custom parameters
        space.update(self.custom_params)
        
        return space
    
    def get_metric_config(self) -> Dict[str, Any]:
        """Return default metric configuration."""
        return {
            "metric": "val_loss",
            "mode": "min"
        }


@dataclass
class BOHBConfig:
    """
    Configuration for BOHB optimization runs.
    
    This configuration is independent of the specific model being optimized,
    focusing only on the optimization process itself.
    """
    
    # BOHB algorithm settings
    max_epochs: int = 100
    grace_period: int = 10
    reduction_factor: int = 3
    max_concurrent_trials: int = 4
    
    # Resource allocation
    resources_per_trial: Dict[str, Union[int, float]] = field(default_factory=lambda: {
        "cpu": 4,
        "gpu": 1.0
    })
    
    # Experiment management
    experiment_name: str = "lightning_bohb"
    experiment_dir: Path = field(default_factory=lambda: Path("./bohb_experiments"))
    checkpoint_freq: int = 1
    keep_checkpoints_num: int = 2
    
    # Optimization settings
    num_samples: int = -1  # -1 means let BOHB decide
    time_budget_hrs: Optional[float] = None
    
    # Monitoring
    verbose: int = 1
    log_to_file: bool = True
    enable_dashboard: bool = False
    dashboard_port: int = 8080
    
    # Resume settings
    resume_from: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Path):
                result[field_name] = str(field_value)
            else:
                result[field_name] = field_value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BOHBConfig":
        """Create config from dictionary."""
        # Convert string paths back to Path objects
        if "experiment_dir" in config_dict and isinstance(config_dict["experiment_dir"], str):
            config_dict["experiment_dir"] = Path(config_dict["experiment_dir"])
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "BOHBConfig":
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save config to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        if self.max_epochs <= 0:
            issues.append("max_epochs must be positive")
        
        if self.grace_period <= 0:
            issues.append("grace_period must be positive")
        
        if self.grace_period >= self.max_epochs:
            issues.append("grace_period must be less than max_epochs")
        
        if self.reduction_factor <= 1:
            issues.append("reduction_factor must be greater than 1")
        
        if self.max_concurrent_trials <= 0:
            issues.append("max_concurrent_trials must be positive")
        
        if self.resources_per_trial.get("cpu", 1) <= 0:
            issues.append("CPU resources must be positive")
        
        if self.resources_per_trial.get("gpu", 0) < 0:
            issues.append("GPU resources cannot be negative")
        
        return issues


class ConfigManager:
    """
    Manages configuration merging and validation for BOHB trials.
    
    This class handles the complexity of merging base configurations with
    BOHB-suggested hyperparameters.
    """
    
    def __init__(self, base_config_source: Optional[Union[str, Path, Dict[str, Any]]] = None):
        """
        Initialize ConfigManager with a base configuration.
        
        Args:
            base_config_source: Can be:
                - Path/str to a YAML/JSON config file
                - Dict containing the configuration directly
                - None for empty base config
        """
        self.base_config_source = base_config_source
        self.base_config_path = None
        
        # Handle different input types
        if base_config_source is None:
            self.base_config = {}
        elif isinstance(base_config_source, dict):
            self.base_config = base_config_source
        else:
            # It's a path
            self.base_config_path = Path(base_config_source)
            self.base_config = self._load_base_config()
        
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration from file."""
        if not self.base_config_path or not self.base_config_path.exists():
            return {}
            
        suffix = self.base_config_path.suffix.lower()
        
        with open(self.base_config_path, 'r') as f:
            if suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif suffix == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {suffix}")
    
    def merge_configs(
        self,
        base: Dict[str, Any],
        overrides: Dict[str, Any],
        separator: str = "."
    ) -> Dict[str, Any]:
        """
        Recursively merge configurations with support for nested keys.
        
        Args:
            base: Base configuration dictionary
            overrides: Override dictionary (can use dot notation)
            separator: Separator for nested keys
            
        Returns:
            Merged configuration dictionary
        """
        import copy
        result = copy.deepcopy(base)
        
        for key, value in overrides.items():
            if separator in key:
                # Handle nested keys like "model.learning_rate"
                keys = key.split(separator)
                target = result
                
                # Navigate to the nested location
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                
                # Set the final value
                target[keys[-1]] = value
            else:
                # Direct key assignment
                result[key] = value
        
        return result
    
    def create_trial_config(
        self,
        trial_params: Dict[str, Any],
        trial_dir: Path
    ) -> Path:
        """
        Create a trial-specific configuration file.
        
        Args:
            trial_params: Parameters suggested by BOHB
            trial_dir: Directory for this trial
            
        Returns:
            Path to the created configuration file
        """
        # Merge with base config
        if self.base_config:
            merged_config = self.merge_configs(self.base_config, trial_params)
        else:
            merged_config = trial_params
        
        # Save to trial directory
        config_path = trial_dir / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(merged_config, f, default_flow_style=False)
        
        return config_path
    
    def extract_flat_params(
        self,
        nested_dict: Dict[str, Any],
        prefix: str = "",
        separator: str = "."
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionary to dot-notation keys.
        
        Args:
            nested_dict: Nested dictionary
            prefix: Prefix for keys
            separator: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        flat = {}
        
        for key, value in nested_dict.items():
            full_key = f"{prefix}{separator}{key}" if prefix else key
            
            if isinstance(value, dict) and not value.get("_type"):
                # Recursively flatten nested dicts (unless they have _type field)
                flat.update(self.extract_flat_params(value, full_key, separator))
            else:
                flat[full_key] = value
        
        return flat