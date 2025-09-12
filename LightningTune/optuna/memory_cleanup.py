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
    # Force Python garbage collection FIRST to release references
    collected = gc.collect()
    if collected > 0:
        logger.debug(f"Garbage collector freed {collected} objects")
    
    # Clear PyTorch CUDA cache if available
    if torch.cuda.is_available():
        # More aggressive GPU cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Reset peak memory stats to prevent accumulation
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        # Additional empty_cache after synchronize
        torch.cuda.empty_cache()
        
    # Clear any matplotlib figures if they exist
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except ImportError:
        pass
    
    # Additional garbage collection pass
    gc.collect()
    
    # Log memory usage if psutil is available
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        logger.debug(
            f"Memory after cleanup: RSS={mem_info.rss/1024/1024:.1f}MB, "
            f"VMS={mem_info.vms/1024/1024:.1f}MB"
        )
        
        # Also log GPU memory if available
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            logger.debug(
                f"GPU memory after cleanup: Allocated={allocated:.1f}MB, "
                f"Reserved={reserved:.1f}MB"
            )
    except ImportError:
        pass