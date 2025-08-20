"""
Early stopping callback based on number of training steps.

This callback is useful for HPO where you want to limit trial duration
without affecting the learning rate scheduler's total_steps calculation.
"""

import logging
from typing import Optional

import lightning as L
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


class EarlyStoppingSteps(Callback):
    """
    Stop training after a specified number of training steps.
    
    Unlike setting trainer.max_steps, this doesn't affect the learning rate
    scheduler's total_steps calculation, preserving the intended LR schedule.
    
    This is particularly useful for HPO where you want shorter trials but
    need to maintain the same LR schedule profile as full training.
    
    Args:
        stopping_steps: Number of training steps after which to stop
        verbose: Whether to log when stopping
    """
    
    def __init__(
        self,
        stopping_steps: int,
        verbose: bool = True
    ):
        """
        Initialize the callback.
        
        Args:
            stopping_steps: Stop training after this many steps
            verbose: Whether to log when stopping
        """
        super().__init__()
        self.stopping_steps = stopping_steps
        self.verbose = verbose
        self._should_stop = False
    
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Check if we should stop training."""
        current_step = trainer.global_step
        
        if current_step >= self.stopping_steps:
            if self.verbose and not self._should_stop:
                logger.info(
                    f"🛑 EarlyStoppingSteps: Stopping at step {current_step} "
                    f"(target: {self.stopping_steps})"
                )
            self._should_stop = True
            trainer.should_stop = True
    
    def on_validation_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule
    ) -> None:
        """Also check at validation end to ensure we stop."""
        if trainer.global_step >= self.stopping_steps:
            if self.verbose and not self._should_stop:
                logger.info(
                    f"🛑 EarlyStoppingSteps: Stopping at step {trainer.global_step} "
                    f"(target: {self.stopping_steps})"
                )
            self._should_stop = True
            trainer.should_stop = True