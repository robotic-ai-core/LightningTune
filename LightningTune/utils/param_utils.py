"""Utilities for parameter name simplification and formatting."""

from typing import Dict, Any
import re


def simplify_param_names(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplify parameter names for cleaner WandB/logging display.
    
    Removes all 'init_args' segments and simplifies nested paths.
    
    Examples:
    - 'model.init_args.learning_rate' -> 'learning_rate'
    - 'model.init_args.transformer_hparams.num_layers' -> 'transformer.num_layers'
    - 'data.init_args.batch_size' -> 'batch_size'
    - 'model.init_args.adapter.init_args.hidden_dim' -> 'adapter.hidden_dim'
    
    Args:
        params: Dictionary with full config paths as keys
        
    Returns:
        Dictionary with simplified keys for better readability
    """
    simplified = {}
    
    for key, value in params.items():
        # Split the key into parts
        parts = key.split('.')
        
        # Keep only parts that are not 'init_args'
        clean_parts = [part for part in parts if part != 'init_args']
        
        # Remove the first part if it's a top-level category like 'model', 'data', 'trainer'
        if clean_parts and clean_parts[0] in ['model', 'data', 'trainer']:
            clean_parts = clean_parts[1:]
        
        # Simplify common nested names
        if clean_parts:
            # Replace long names with shorter versions
            clean_parts = [
                part.replace('transformer_hparams', 'transformer')
                    .replace('adapter_hparams', 'adapter')
                    .replace('image_processor_hparams', 'image')
                    .replace('_hparams', '')  # Remove any other _hparams suffixes
                for part in clean_parts
            ]
        
        # Join back together
        clean_key = '.'.join(clean_parts) if clean_parts else key
        
        # Store with simplified name
        simplified[clean_key] = value
    
    return simplified


class ParamNameSimplifier:
    """
    Configurable parameter name simplifier with regex pattern support.
    """
    
    default_rules = [
        (r'\.init_args\.', '.'),  # Remove init_args segments
        (r'^(model|data|trainer)\.', ''),  # Remove top-level categories
        (r'_hparams', ''),  # Remove _hparams suffixes
        (r'transformer_hparams', 'transformer'),
        (r'adapter_hparams', 'adapter'),
        (r'image_processor_hparams', 'image'),
    ]
    
    def __init__(self, rules=None):
        """
        Initialize simplifier with custom or default rules.
        
        Args:
            rules: List of (pattern, replacement) tuples for regex substitution
        """
        self.rules = rules or self.default_rules
    
    def simplify(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simplify parameter names using configured rules.
        
        Args:
            params: Dictionary with full config paths as keys
            
        Returns:
            Dictionary with simplified keys
        """
        simplified = {}
        
        for key, value in params.items():
            simplified_key = key
            
            # Apply each rule in order
            for pattern, replacement in self.rules:
                simplified_key = re.sub(pattern, replacement, simplified_key)
            
            # Clean up any double dots or leading/trailing dots
            simplified_key = re.sub(r'\.+', '.', simplified_key).strip('.')
            
            simplified[simplified_key or key] = value
        
        return simplified