"""
Simple memory cleanup utilities for HPO trials.

This module provides lightweight cleanup functions to prevent memory
accumulation during long-running HPO sessions.
"""

import gc
import logging

import torch

logger = logging.getLogger(__name__)


def cleanup_trial_resources():
    """
    Perform cleanup of trial resources to prevent memory accumulation.
    
    This function should be called after each trial completes to ensure
    proper memory management.
    """
    # Clear PyTorch CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    # Force Python garbage collection
    collected = gc.collect()
    if collected > 0:
        logger.debug(f"Garbage collector freed {collected} objects")
    
    # Clear any matplotlib figures if they exist
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except ImportError:
        pass
    
    # Log memory usage if psutil is available
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        logger.debug(
            f"Memory after cleanup: RSS={mem_info.rss/1024/1024:.1f}MB, "
            f"VMS={mem_info.vms/1024/1024:.1f}MB"
        )
    except ImportError:
        pass