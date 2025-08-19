"""
Configuration utilities for merging and manipulating configs.
"""

from typing import Dict, Any
import copy


def deep_merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override config into base config.
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
    
    Returns:
        Merged configuration (base is not modified)
    
    Example:
        >>> base = {"model": {"lr": 0.1}, "data": {"batch_size": 32}}
        >>> override = {"model": {"lr": 0.01}}
        >>> result = deep_merge_configs(base, override)
        >>> result == {"model": {"lr": 0.01}, "data": {"batch_size": 32}}
        True
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge_configs(result[key], value)
        else:
            # Override the value
            result[key] = value
    
    return result


def apply_dotted_updates(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply dotted-notation updates to a nested config.
    
    Args:
        config: Base configuration dictionary
        updates: Dictionary with dotted keys (e.g., "model.lr": 0.01)
    
    Returns:
        Updated configuration (config is not modified)
    
    Example:
        >>> config = {"model": {"lr": 0.1}, "data": {"batch_size": 32}}
        >>> updates = {"model.lr": 0.01, "data.batch_size": 64}
        >>> result = apply_dotted_updates(config, updates)
        >>> result == {"model": {"lr": 0.01}, "data": {"batch_size": 64}}
        True
    """
    result = copy.deepcopy(config)
    
    for key, value in updates.items():
        if '.' in key:
            # Handle nested keys like "model.learning_rate"
            parts = key.split('.')
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            result[key] = value
    
    return result


def merge_with_dotted_updates(
    base: Dict[str, Any], 
    override: Dict[str, Any] = None,
    dotted_updates: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Convenience function to apply both deep merge and dotted updates.
    
    Args:
        base: Base configuration
        override: Optional nested config to deep merge
        dotted_updates: Optional dotted-notation updates
    
    Returns:
        Merged and updated configuration
    
    Example:
        >>> base = {"model": {"lr": 0.1}, "data": {"batch_size": 32}}
        >>> override = {"model": {"weight_decay": 0.01}}
        >>> dotted = {"model.lr": 0.001}
        >>> result = merge_with_dotted_updates(base, override, dotted)
        >>> result["model"] == {"lr": 0.001, "weight_decay": 0.01}
        True
    """
    result = copy.deepcopy(base)
    
    if override:
        result = deep_merge_configs(result, override)
    
    if dotted_updates:
        result = apply_dotted_updates(result, dotted_updates)
    
    return result