"""
NaN/Inf detection callback for Optuna trials.

This callback automatically terminates trials that produce NaN or Inf values,
preventing wasted compute on diverged models.
"""

import math
import logging
from typing import Optional, Any

import optuna
import lightning as L
from lightning.pytorch.callbacks import Callback
import torch

logger = logging.getLogger(__name__)


def _report_then_prune(
    trial: optuna.Trial,
    value: float,
    index: int,
    monitor: str,
    index_name: str,
) -> None:
    """Report the metric value to Optuna, mark reason, and prune.

    This ensures Optuna records the intermediate value (even if NaN/Inf)
    before the trial is pruned, improving diagnostics and dashboards.
    """
    try:
        trial.report(value, index)
    except Exception:
        # Reporting failure should not block pruning
        pass
    try:
        trial.set_user_attr('failed_reason', 'nan_or_inf_loss')
    except Exception:
        pass
    raise optuna.TrialPruned(f"{monitor} is NaN/Inf at {index_name} {index}")


class NaNDetectionCallback(Callback):
    """
    Callback that terminates trials when NaN or Inf values are detected.
    
    This prevents wasted compute on diverged models and helps Optuna
    focus on promising hyperparameter regions.
    """
    
    def __init__(
        self,
        trial: optuna.Trial,
        monitor: str = "val_loss",
        check_train_loss: bool = True,
        check_every_n_steps: int = 100,  # Check every 100 steps (we check all loss keys, so less frequent is fine)
        verbose: bool = True,
    ):
        """
        Initialize NaN detection callback.
        
        Args:
            trial: Optuna trial object
            monitor: Metric to monitor for NaN/Inf
            check_train_loss: Also check training loss for NaN/Inf
            check_every_n_steps: Check frequency during training (default: 100, checks all loss keys)
            verbose: Whether to log when NaN/Inf is detected
        """
        self.trial = trial
        self.monitor = monitor
        self.check_train_loss = check_train_loss
        self.check_every_n_steps = check_every_n_steps
        self.verbose = verbose
        self.step_count = 0
    
    def _check_value(self, value: float, metric_name: str) -> bool:
        """
        Check if value is NaN or Inf.
        
        Returns:
            True if value is invalid (NaN or Inf), False otherwise
        """
        if math.isnan(value) or math.isinf(value):
            if self.verbose:
                state = "NaN" if math.isnan(value) else "Inf"
                logger.warning(
                    f"🚨 Trial {self.trial.number}: {metric_name} is {state}! "
                    f"Terminating trial to save compute."
                )
            return True
        return False
    
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Check training loss for NaN/Inf during training."""
        if not self.check_train_loss:
            return
        
        self.step_count += 1
        if self.step_count % self.check_every_n_steps != 0:
            return
        
        # Collect all loss values to check
        losses_to_check = {}
        
        # Method 1: Check outputs - look for ANY key containing "loss"
        if outputs is not None:
            if isinstance(outputs, dict):
                # Check all keys containing "loss" (case-insensitive)
                for key, value in outputs.items():
                    if 'loss' in key.lower():
                        losses_to_check[f"outputs.{key}"] = value
            elif isinstance(outputs, torch.Tensor):
                losses_to_check["outputs"] = outputs
        
        # Method 2: Check trainer's callback metrics for any loss-related metrics
        if hasattr(trainer, 'callback_metrics'):
            for key, value in trainer.callback_metrics.items():
                # Check if key contains "loss" and is training-related
                if 'loss' in key.lower() and ('train' in key.lower() or '/' not in key):
                    losses_to_check[f"metrics.{key}"] = value
        
        # Method 3: Check logged metrics as fallback
        if not losses_to_check and hasattr(trainer, 'logged_metrics'):
            for key, value in trainer.logged_metrics.items():
                if 'loss' in key.lower() and ('train' in key.lower() or '/' not in key):
                    losses_to_check[f"logged.{key}"] = value
        
        # Check all collected loss values
        for loss_name, loss in losses_to_check.items():
            if loss is not None:
                if isinstance(loss, torch.Tensor):
                    loss_value = loss.item()
                else:
                    try:
                        loss_value = float(loss)
                    except (TypeError, ValueError):
                        continue  # Skip non-numeric values
                
                if self._check_value(loss_value, f"{loss_name}"):
                    # Report then prune so Optuna records the intermediate value
                    _report_then_prune(
                        self.trial,
                        loss_value,
                        trainer.global_step,
                        loss_name,
                        "step",
                    )
        
        # Debug logging if no losses found (only log once)
        if not losses_to_check and self.verbose and self.step_count == 100:
            logger.debug(f"NaN detector: No loss values found in outputs or metrics at step {trainer.global_step}")
    
    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Check validation metrics for NaN/Inf."""
        # Check monitored metric
        if self.monitor in trainer.callback_metrics:
            value = trainer.callback_metrics[self.monitor].item()
            
            if self._check_value(value, f"{self.monitor}"):
                _report_then_prune(
                    self.trial,
                    value,
                    trainer.current_epoch,
                    self.monitor,
                    "epoch",
                )
        
        # Also check all validation metrics for completeness
        for key, value in trainer.callback_metrics.items():
            if 'val' in key and isinstance(value, torch.Tensor):
                metric_value = value.item()
                if math.isnan(metric_value) or math.isinf(metric_value):
                    if self.verbose:
                        state = "NaN" if math.isnan(metric_value) else "Inf"
                        logger.info(f"   Note: {key} is also {state}")


from .callbacks import OptunaPruningCallback


class EnhancedOptunaPruningCallback(OptunaPruningCallback):
    """
    Enhanced Optuna pruning callback that combines regular pruning with NaN detection.
    
    This callback:
    1. Reports metrics to Optuna for pruning decisions
    2. Automatically terminates trials with NaN/Inf values
    """
    
    def __init__(
        self,
        trial: optuna.Trial,
        monitor: str = "val_loss",
        report_every_n_epochs: int = 1,
        report_every_n_steps: Optional[int] = None,
        check_nan: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize enhanced pruning callback.
        
        Args:
            trial: Optuna trial object
            monitor: Metric to monitor
            report_every_n_epochs: Report metric every N epochs
            report_every_n_steps: Report metric every N steps (overrides epoch reporting)
            check_nan: Whether to check for NaN/Inf values
            verbose: Whether to log NaN/Inf detection
        """
        super().__init__(trial, monitor, report_every_n_epochs, report_every_n_steps)
        self.check_nan = check_nan
        self.verbose = verbose
    
    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Check for NaN/Inf before regular pruning logic."""
        # Skip if using step-based reporting
        if self.report_every_n_steps is not None:
            return
        
        # Check if we should report this epoch
        epoch = trainer.current_epoch
        if epoch % self.report_every_n_epochs != 0:
            return
        
        # Get metric value
        if self.monitor in trainer.callback_metrics:
            value = trainer.callback_metrics[self.monitor].item()
            
            # Check for NaN/Inf first
            if self.check_nan and (math.isnan(value) or math.isinf(value)):
                if self.verbose:
                    state = "NaN" if math.isnan(value) else "Inf"
                    logger.warning(
                        f"🚨 Trial {self.trial.number}: {self.monitor} is {state} at epoch {epoch}! "
                        f"Terminating trial."
                    )
                _report_then_prune(
                    self.trial,
                    value,
                    epoch,
                    self.monitor,
                    "epoch",
                )
            
            # Regular pruning logic
            self.trial.report(value, epoch)
            
            if self.trial.should_prune():
                logger.info(f"Trial {self.trial.number} pruned at epoch {epoch}")
                raise optuna.TrialPruned()