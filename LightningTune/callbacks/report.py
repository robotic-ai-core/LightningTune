"""
Lightning callback for reporting metrics to BOHB.

This callback integrates with Lightning training to report metrics
back to Ray Tune for BOHB optimization.
"""

import logging
from typing import Dict, Any, Optional, List
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from pathlib import Path

try:
    from ray.air import session
except ImportError:
    session = None
    
logger = logging.getLogger(__name__)


class BOHBReportCallback(Callback):
    """
    Lightning callback for reporting metrics to BOHB.
    
    This callback automatically reports training and validation metrics
    to Ray Tune at appropriate intervals for BOHB decision making.
    """
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        report_on_epoch: bool = True,
        report_on_train_end: bool = False,
        checkpoint_on_epoch: bool = True,
        primary_metric: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the callback.
        
        Args:
            metrics: List of metric names to report (None = report all)
            report_on_epoch: Report metrics at epoch end
            report_on_train_end: Report metrics at training end
            checkpoint_on_epoch: Save checkpoint at epoch end
            primary_metric: Primary metric for BOHB (defaults to first in metrics)
            verbose: Enable verbose logging
        """
        super().__init__()
        
        self.metrics = metrics
        self.report_on_epoch = report_on_epoch
        self.report_on_train_end = report_on_train_end
        self.checkpoint_on_epoch = checkpoint_on_epoch
        self.primary_metric = primary_metric
        self.verbose = verbose
        
        # Track training iteration for BOHB
        self._iteration = 0
    
    def _gather_metrics(self, trainer: pl.Trainer) -> Dict[str, Any]:
        """Gather metrics from trainer."""
        metrics = {}
        
        # Get all callback metrics
        callback_metrics = trainer.callback_metrics
        
        if self.metrics is None:
            # Report all metrics
            for key, value in callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics[key] = float(value.cpu().item())
                else:
                    metrics[key] = value
        else:
            # Report only specified metrics
            for metric_name in self.metrics:
                if metric_name in callback_metrics:
                    value = callback_metrics[metric_name]
                    if isinstance(value, torch.Tensor):
                        metrics[metric_name] = float(value.cpu().item())
                    else:
                        metrics[metric_name] = value
        
        # Add epoch and iteration information
        metrics["epoch"] = trainer.current_epoch
        metrics["training_iteration"] = self._iteration
        metrics["global_step"] = trainer.global_step
        
        return metrics
    
    def _create_checkpoint(self, trainer: pl.Trainer) -> Optional[Dict[str, str]]:
        """Create checkpoint for Ray Tune."""
        if not self.checkpoint_on_epoch:
            return None
        
        try:
            # Get trial directory from Ray session
            if session:
                checkpoint_dir = Path(session.get_trial_dir()) / "checkpoints"
            else:
                # Fallback to trainer's default directory
                checkpoint_dir = Path(trainer.default_root_dir) / "checkpoints"
            
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"epoch_{trainer.current_epoch}.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            
            return {"checkpoint_path": str(checkpoint_path)}
            
        except Exception as e:
            logger.warning(f"Failed to create checkpoint: {e}")
            return None
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Report metrics at the end of each training epoch."""
        # Increment iteration counter
        self._iteration += 1
        
        # Gather metrics
        metrics = self._gather_metrics(trainer)
        
        if self.verbose:
            logger.info(f"Reporting metrics to BOHB: {metrics}")
        
        # Create checkpoint if needed
        checkpoint = self._create_checkpoint(trainer)
        
        # Report to Ray Tune
        try:
            if checkpoint:
                session.report(metrics, checkpoint=checkpoint)
            else:
                session.report(metrics)
        except Exception as e:
            if self.verbose:
                logger.warning(f"Failed to report to Ray Tune: {e}")
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Report validation metrics."""
        if trainer.sanity_checking:
            return
        
            return
        
        # Only report if we haven't already reported this epoch
        # (avoid double reporting if validation happens at epoch end)
        if hasattr(self, '_last_reported_epoch'):
            if self._last_reported_epoch == trainer.current_epoch:
                return
        
        # Gather metrics
        metrics = self._gather_metrics(trainer)
        
        # Report primary metric for BOHB scheduling decisions
        if self.primary_metric and self.primary_metric in metrics:
            primary_value = metrics[self.primary_metric]
            
            if self.verbose:
                logger.info(f"Reporting primary metric to BOHB: {self.primary_metric}={primary_value}")
            
            try:
                # Report just the primary metric for quick BOHB decisions
                session.report({self.primary_metric: primary_value})
                self._last_reported_epoch = trainer.current_epoch
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Failed to report validation metric: {e}")
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Report final metrics at training end."""
        # Gather final metrics
        metrics = self._gather_metrics(trainer)
        metrics["training_completed"] = True
        
        if self.verbose:
            logger.info(f"Reporting final metrics to BOHB: {metrics}")
        
        # Final checkpoint
        checkpoint = self._create_checkpoint(trainer)
        
        # Report to Ray Tune
        try:
            if checkpoint:
                session.report(metrics, checkpoint=checkpoint)
            else:
                session.report(metrics)
        except Exception as e:
            if self.verbose:
                logger.warning(f"Failed to report final metrics: {e}")
    
    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: Exception):
        """Handle exceptions by reporting failure to BOHB."""
        logger.error(f"Training failed with exception: {exception}")
        
        # Report failure to Ray Tune
        try:
            session.report(
                {
                    "training_failed": True,
                    "exception": str(exception),
                    "epoch": trainer.current_epoch,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to report exception to Ray Tune: {e}")


class AdaptiveBOHBCallback(BOHBReportCallback):
    """
    Advanced callback with adaptive reporting frequency.
    
    This callback adjusts reporting frequency based on training progress
    to reduce overhead while maintaining good BOHB scheduling.
    """
    
    def __init__(
        self,
        warmup_epochs: int = 5,
        report_frequency: int = 1,
        report_frequency_after_warmup: int = 5,
        **kwargs
    ):
        """
        Initialize adaptive callback.
        
        Args:
            warmup_epochs: Number of initial epochs with frequent reporting
            report_frequency: Reporting frequency during warmup
            report_frequency_after_warmup: Reporting frequency after warmup
            **kwargs: Additional arguments for BOHBReportCallback
        """
        super().__init__(**kwargs)
        
        self.warmup_epochs = warmup_epochs
        self.report_frequency = report_frequency
        self.report_frequency_after_warmup = report_frequency_after_warmup
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Adaptively report based on epoch."""
        current_epoch = trainer.current_epoch
        
        # Determine if we should report this epoch
        if current_epoch < self.warmup_epochs:
            # During warmup, report frequently
            should_report = (current_epoch % self.report_frequency) == 0
        else:
            # After warmup, report less frequently
            should_report = (
                (current_epoch - self.warmup_epochs) % self.report_frequency_after_warmup
            ) == 0
        
        # Always report the last epoch
        if current_epoch == trainer.max_epochs - 1:
            should_report = True
        
        if should_report:
            super().on_train_epoch_end(trainer, pl_module)