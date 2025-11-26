"""
Simple memory cleanup utilities for HPO trials.

This module provides lightweight cleanup functions to prevent memory
accumulation during long-running HPO sessions.
"""

import gc
import logging

import torch

logger = logging.getLogger(__name__)


def cleanup_dataloader_workers(trainer=None, datamodule=None):
    """
    Explicitly clean up PyTorch DataLoader workers.

    DataLoader workers (QueueFeederThread, _pin_memory_loop threads) can
    accumulate across trials if not properly terminated. This function
    ensures they are cleaned up.

    This function delegates to the canonical cleanup implementation in
    LightningReflow.utils.cleanup_utils.

    Args:
        trainer: Lightning Trainer instance (optional)
        datamodule: Lightning DataModule instance (optional)
    """
    # Import canonical cleanup from LightningReflow
    try:
        from lightning_reflow.utils import cleanup_dataloader_workers as canonical_cleanup
        canonical_cleanup(trainer=trainer, datamodule=datamodule, verbose=False)
    except ImportError:
        # Fallback if LightningReflow not available (should not happen in practice)
        logger.warning(
            "Could not import canonical cleanup from LightningReflow. "
            "Using local fallback implementation."
        )
        # Minimal fallback implementation
        if datamodule is not None:
            try:
                datamodule.teardown('fit')
            except Exception as e:
                logger.warning(f"datamodule.teardown() failed: {e}")

        if trainer is not None:
            try:
                if hasattr(trainer, 'train_dataloader') and trainer.train_dataloader is not None:
                    if hasattr(trainer.train_dataloader, '_iterator'):
                        del trainer.train_dataloader._iterator
            except Exception as e:
                logger.warning(f"Trainer DataLoader cleanup failed: {e}")


def cleanup_trial_resources(trainer=None, datamodule=None):
    """
    Perform cleanup of trial resources to prevent memory accumulation.

    This function is lightweight since process restart (with --restart-on-save)
    handles complete memory cleanup. It only cleans up DataLoader workers which
    can accumulate threads if not explicitly terminated.

    Args:
        trainer: Lightning Trainer instance (optional, for DataLoader cleanup)
        datamodule: Lightning DataModule instance (optional, for DataLoader cleanup)
    """
    logger.info("[MemoryCleanup] Cleaning up DataLoader workers")

    # CRITICAL: Clean up DataLoader workers to prevent thread accumulation
    # This is necessary even with process restart because threads can accumulate
    # within a single process lifetime before restart
    cleanup_dataloader_workers(trainer, datamodule)

    # Basic garbage collection
    collected = gc.collect()
    if collected > 0:
        logger.debug(f"[MemoryCleanup] Garbage collector freed {collected} objects")

    # Basic CUDA cache clear (lightweight, non-aggressive)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def aggressive_cleanup():
    """
    Perform aggressive memory cleanup after a trial.

    This function should be called after all trial references have been
    deleted to ensure memory is fully released. It performs multiple
    garbage collection passes and clears CUDA caches.
    """
    # Multiple GC passes to handle reference cycles
    for _ in range(3):
        gc.collect()

    # Clear CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Reset torch.compile caches if available
    if hasattr(torch, '_dynamo'):
        try:
            if hasattr(torch._dynamo, 'reset'):
                torch._dynamo.reset()
        except Exception:
            pass

    # Clear any inductor caches
    if hasattr(torch, '_inductor'):
        try:
            if hasattr(torch._inductor, 'codecache'):
                # Don't clear the entire cache, just trim it
                pass
        except Exception:
            pass

    # Final GC pass
    gc.collect()