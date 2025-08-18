"""
PyTorch Lightning callbacks for Optuna integration.

This module provides callbacks that integrate Optuna's trial pruning
and checkpointing with PyTorch Lightning training.
"""

import os
from pathlib import Path
from typing import Optional, Any, Dict
import logging

import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch

logger = logging.getLogger(__name__)


class OptunaPruningCallback(Callback):
    """
    PyTorch Lightning callback for Optuna trial pruning.
    
    This callback enables early stopping of unpromising trials
    based on intermediate metric values, significantly speeding
    up hyperparameter optimization.
    """
    
    def __init__(
        self,
        trial: optuna.Trial,
        monitor: str = "val_loss",
        report_every_n_epochs: int = 1,
        report_every_n_steps: Optional[int] = None,
    ):
        """
        Initialize pruning callback.
        
        Args:
            trial: Optuna trial object
            monitor: Metric to monitor for pruning decisions
            report_every_n_epochs: Report metric every N epochs
            report_every_n_steps: Report metric every N steps (overrides epoch reporting)
        """
        self.trial = trial
        self.monitor = monitor
        self.report_every_n_epochs = report_every_n_epochs
        self.report_every_n_steps = report_every_n_steps
        self.current_step = 0
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check if trial should be pruned at validation end."""
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
            
            # Report to Optuna
            self.trial.report(value, epoch)
            
            # Check if trial should be pruned
            if self.trial.should_prune():
                logger.info(f"Trial {self.trial.number} pruned at epoch {epoch}")
                raise optuna.TrialPruned()
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Check if trial should be pruned at batch end (for step-based reporting)."""
        if self.report_every_n_steps is None:
            return
        
        self.current_step += 1
        
        # Check if we should report this step
        if self.current_step % self.report_every_n_steps != 0:
            return
        
        # Get metric value (might be from training metrics)
        if self.monitor in trainer.callback_metrics:
            value = trainer.callback_metrics[self.monitor].item()
            
            # Report to Optuna
            self.trial.report(value, self.current_step)
            
            # Check if trial should be pruned
            if self.trial.should_prune():
                logger.info(f"Trial {self.trial.number} pruned at step {self.current_step}")
                raise optuna.TrialPruned()


class OptunaCheckpointCallback(Callback):
    """
    Enhanced checkpointing callback for Optuna trials.
    
    Saves checkpoints with trial information and manages
    checkpoint storage for easy retrieval of best models.
    """
    
    def __init__(
        self,
        trial: optuna.Trial,
        checkpoint_dir: Optional[Path] = None,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        save_on_train_epoch_end: bool = False,
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            trial: Optuna trial object
            checkpoint_dir: Directory for saving checkpoints
            monitor: Metric to monitor for best model
            mode: "min" or "max"
            save_top_k: Number of best models to keep
            save_on_train_epoch_end: Save at end of training epochs
        """
        self.trial = trial
        self.checkpoint_dir = Path(checkpoint_dir or f"./checkpoints/trial_{trial.number}")
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_on_train_epoch_end = save_on_train_epoch_end
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = float('inf') if mode == "min" else float('-inf')
        self.best_checkpoint_path = None
        self.checkpoint_paths = []
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == "min":
            return current < best
        return current > best
    
    def _save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        epoch: int,
        step: int,
        metric_value: Optional[float] = None,
    ) -> Path:
        """Save a checkpoint."""
        # Create checkpoint data
        checkpoint = {
            'epoch': epoch,
            'global_step': step,
            'model_state_dict': pl_module.state_dict(),
            'optimizer_state_dict': trainer.optimizers[0].state_dict() if trainer.optimizers else None,
            'trial_number': self.trial.number,
            'trial_params': self.trial.params,
            'metric_value': metric_value,
            'metric_name': self.monitor,
        }
        
        # Add scheduler state if present
        if trainer.lr_scheduler_configs:
            checkpoint['scheduler_state_dict'] = trainer.lr_scheduler_configs[0].scheduler.state_dict()
        
        # Create filename
        if metric_value is not None:
            filename = f"epoch={epoch}_step={step}_{self.monitor}={metric_value:.4f}.ckpt"
        else:
            filename = f"epoch={epoch}_step={step}.ckpt"
        
        filepath = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        logger.debug(f"Saved checkpoint: {filepath}")
        
        return filepath
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save checkpoint if metric improved."""
        if self.monitor not in trainer.callback_metrics:
            return
        
        current_metric = trainer.callback_metrics[self.monitor].item()
        
        # Check if we should save
        if self._is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            
            # Save checkpoint
            checkpoint_path = self._save_checkpoint(
                trainer,
                pl_module,
                trainer.current_epoch,
                trainer.global_step,
                current_metric
            )
            
            self.best_checkpoint_path = checkpoint_path
            self.checkpoint_paths.append(checkpoint_path)
            
            # Store best checkpoint path in trial
            self.trial.set_user_attr('best_checkpoint', str(checkpoint_path))
            self.trial.set_user_attr(f'best_{self.monitor}', current_metric)
            
            # Manage checkpoint count
            if len(self.checkpoint_paths) > self.save_top_k:
                # Remove oldest checkpoint
                oldest = self.checkpoint_paths.pop(0)
                if oldest != self.best_checkpoint_path and oldest.exists():
                    oldest.unlink()
                    logger.debug(f"Removed old checkpoint: {oldest}")
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Optionally save checkpoint at end of training epoch."""
        if not self.save_on_train_epoch_end:
            return
        
        self._save_checkpoint(
            trainer,
            pl_module,
            trainer.current_epoch,
            trainer.global_step,
        )
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log final checkpoint information."""
        if self.best_checkpoint_path:
            logger.info(f"Best checkpoint for trial {self.trial.number}: {self.best_checkpoint_path}")
            logger.info(f"Best {self.monitor}: {self.best_metric}")


class OptunaProgressCallback(Callback):
    """
    Callback for tracking and displaying optimization progress.
    
    Provides real-time feedback about the optimization process
    including current best values and parameter importance.
    """
    
    def __init__(
        self,
        study: optuna.Study,
        trial: optuna.Trial,
        print_every_n_epochs: int = 10,
    ):
        """
        Initialize progress callback.
        
        Args:
            study: Optuna study object
            trial: Current trial
            print_every_n_epochs: Print progress every N epochs
        """
        self.study = study
        self.trial = trial
        self.print_every_n_epochs = print_every_n_epochs
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Print progress at end of training epoch."""
        epoch = trainer.current_epoch
        
        if epoch % self.print_every_n_epochs != 0:
            return
        
        # Get current best from study
        try:
            best_value = self.study.best_value
            best_trial = self.study.best_trial.number
        except:
            best_value = None
            best_trial = None
        
        print(f"\n[Trial {self.trial.number}, Epoch {epoch}]")
        print(f"  Current trial params: {self.trial.params}")
        
        if best_value is not None:
            print(f"  Study best: {best_value:.6f} (trial {best_trial})")
        
        # Print current metrics
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.6f}")


class OptunaEarlyStoppingCallback(Callback):
    """
    Early stopping callback that integrates with Optuna.
    
    Stops training when metric stops improving and reports
    final value to Optuna.
    """
    
    def __init__(
        self,
        trial: optuna.Trial,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0001,
    ):
        """
        Initialize early stopping callback.
        
        Args:
            trial: Optuna trial object
            monitor: Metric to monitor
            patience: Number of epochs with no improvement to wait
            mode: "min" or "max"
            min_delta: Minimum change to qualify as improvement
        """
        self.trial = trial
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_metric = float('inf') if mode == "min" else float('-inf')
        self.patience_counter = 0
        self.stopped_epoch = 0
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == "min":
            return current < (best - self.min_delta)
        return current > (best + self.min_delta)
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check for early stopping at validation end."""
        if self.monitor not in trainer.callback_metrics:
            return
        
        current = trainer.callback_metrics[self.monitor].item()
        
        if self._is_better(current, self.best_metric):
            self.best_metric = current
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True
                logger.info(f"Early stopping triggered at epoch {self.stopped_epoch}")
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Report final metric to Optuna."""
        if self.stopped_epoch > 0:
            logger.info(f"Training stopped early at epoch {self.stopped_epoch}")
            logger.info(f"Best {self.monitor}: {self.best_metric}")