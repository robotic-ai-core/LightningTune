"""Utilities for torch.compile management in HPO context."""

from typing import Dict, Any, Optional
import torch
import logging

logger = logging.getLogger(__name__)


def get_compile_settings_for_mode(mode: str) -> Optional[Dict[str, Any]]:
    """
    Get torch.compile settings for a given mode.
    
    Args:
        mode: Compilation mode - "off", "safe", or "aggressive"
        
    Returns:
        Dictionary of compile settings or None for "off" mode
    """
    if mode == "off":
        return {"enabled": False}
    elif mode == "safe":
        # Safe settings for HPO - default PyTorch compilation
        return {
            "enabled": True,
            "backend": "inductor",
            # Use default settings without aggressive optimizations
        }
    elif mode == "aggressive":
        # More aggressive settings for maximum performance
        return {
            "enabled": True,
            "backend": "inductor",
            "mode": "max-autotune",
            "options": {
                "triton.cudagraphs": True,
                "max_autotune": True,
            }
        }
    else:
        raise ValueError(f"Unknown compile mode: {mode}. Use 'off', 'safe', or 'aggressive'")


def reset_torch_compile_state():
    """
    Reset torch.compile state between trials to prevent interference.
    
    This is useful for ensuring clean state between HPO trials,
    especially when models modify global torch.compile settings.
    """
    try:
        # Reset torch._dynamo if available
        if hasattr(torch, '_dynamo'):
            torch._dynamo.reset()
            logger.debug("Reset torch._dynamo state")
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared CUDA cache")
            
    except Exception as e:
        logger.warning(f"Failed to reset torch compile state: {e}")


def configure_compile_for_hpo(compile_mode: str = "safe") -> Dict[str, Any]:
    """
    Configure torch.compile settings optimized for HPO.
    
    Args:
        compile_mode: Mode to use - "off", "safe", or "aggressive"
        
    Returns:
        Configuration dict to pass to model
    """
    settings = get_compile_settings_for_mode(compile_mode)
    
    if settings and settings.get("enabled", False):
        # Reset state before starting to ensure clean environment
        reset_torch_compile_state()
        
        logger.info(f"🔧 Configured torch.compile in '{compile_mode}' mode")
    else:
        logger.info("⚡ Torch compilation disabled")
    
    return settings