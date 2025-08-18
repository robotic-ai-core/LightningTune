"""
TunePauseCallback - Pause callback for Ray Tune trials.

This callback checks for pause signals from TuneReflowCLI and pauses
trials at validation boundaries, similar to LightningReflow's PauseCallback.
"""

import time
import json
import logging
from pathlib import Path
from typing import Optional, Any, Dict

import lightning.pytorch as pl
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback

try:
    from ray.air import session
except ImportError:
    session = None

logger = logging.getLogger(__name__)


class TunePauseCallback(Callback):
    """
    Callback that enables pausing Ray Tune trials at validation boundaries.
    
    This callback works in conjunction with TuneReflowCLI to provide
    interactive pause/resume functionality for Ray Tune optimization.
    
    Features:
    - Checks for pause signals from main process
    - Pauses at validation boundaries for clean state
    - Saves checkpoint before pausing
    - Reports pause status back to Ray Tune
    
    Example:
        ```python
        # Automatically added by TuneReflowCLI
        callback = TunePauseCallback(
            pause_signal_file="/tmp/tune_pause_experiment.signal"
        )
        ```
    """
    
    def __init__(
        self,
        pause_signal_file: Optional[Path] = None,
        check_interval: float = 1.0,
        cli_instance: Optional[Any] = None,
        verbose: bool = False,
    ):
        """
        Initialize TunePauseCallback.
        
        Args:
            pause_signal_file: Path to signal file for pause requests
            check_interval: How often to check for pause signal (seconds)
            cli_instance: Reference to TuneReflowCLI instance
            verbose: Enable verbose logging
        """
        super().__init__()
        
        # Pause signal file
        if pause_signal_file is None:
            self.pause_signal_file = Path("/tmp/tune_pause.signal")
        else:
            self.pause_signal_file = Path(pause_signal_file)
        
        self.check_interval = check_interval
        self.cli_instance = cli_instance
        self.verbose = verbose
        
        # State
        self._pause_requested = False
        self._last_check_time = 0
        self._pause_executed = False
        
        # Trial identification
        self._trial_id = None
        if session:
            try:
                self._trial_id = session.get_trial_id()
            except:
                self._trial_id = "unknown"
        else:
            self._trial_id = "unknown"
    
    def _check_pause_signal(self) -> bool:
        """Check if pause has been requested."""
        # Rate limit checks
        current_time = time.time()
        if current_time - self._last_check_time < self.check_interval:
            return self._pause_requested
        
        self._last_check_time = current_time
        
        # Check signal file
        if self.pause_signal_file.exists():
            try:
                signal_data = json.loads(self.pause_signal_file.read_text())
                if signal_data.get("pause_requested", False):
                    self._pause_requested = True
                    if self.verbose:
                        logger.info(f"Trial {self._trial_id}: Pause signal detected")
                    return True
            except Exception as e:
                logger.debug(f"Error reading pause signal: {e}")
        
        return False
    
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        """Check for pause at validation boundary."""
        # Skip if already paused or not requested
        if self._pause_executed:
            return
        
        # Check for pause signal
        if not self._check_pause_signal():
            return
        
        # Execute pause at validation boundary
        self._execute_pause(trainer, pl_module)
    
    def _execute_pause(self, trainer: Trainer, pl_module: LightningModule):
        """Execute pause at validation boundary."""
        if self._pause_executed:
            return
        
        logger.info(f"Trial {self._trial_id}: Executing pause at epoch {trainer.current_epoch}")
        
        try:
            # Save checkpoint
            checkpoint_path = self._save_checkpoint(trainer)
            
            # Report pause status to Ray Tune
            if session:
                metrics = {
                    "paused": True,
                    "pause_epoch": trainer.current_epoch,
                    "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                }
                
                # Include current validation metrics
                for key, value in trainer.callback_metrics.items():
                    if "val" in key:
                        try:
                            metrics[key] = float(value)
                        except:
                            pass
                
                # Report to Ray Tune
                session.report(metrics)
            
            # Update CLI state if available
            if self.cli_instance:
                self.cli_instance.state.trials_paused += 1
            
            self._pause_executed = True
            
            # Signal trainer to stop
            trainer.should_stop = True
            
            logger.info(f"Trial {self._trial_id}: Paused successfully at epoch {trainer.current_epoch}")
            
        except Exception as e:
            logger.error(f"Trial {self._trial_id}: Failed to execute pause: {e}")
    
    def _save_checkpoint(self, trainer: Trainer) -> Optional[Path]:
        """Save checkpoint for resume."""
        try:
            # Use trainer's checkpoint directory
            checkpoint_dir = Path(trainer.default_root_dir) / "pause_checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save checkpoint with pause marker
            checkpoint_path = checkpoint_dir / f"pause_epoch_{trainer.current_epoch}.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            
            # Also save pause metadata
            metadata_path = checkpoint_dir / "pause_metadata.json"
            metadata = {
                "trial_id": self._trial_id,
                "pause_epoch": trainer.current_epoch,
                "pause_time": time.time(),
                "checkpoint_path": str(checkpoint_path),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2))
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        """Clean up on training end."""
        # Report final status if paused
        if self._pause_executed and session:
            try:
                session.report({
                    "training_completed": False,  # Paused, not completed
                    "paused_at_epoch": trainer.current_epoch,
                })
            except:
                pass
    
    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException):
        """Handle exceptions during training."""
        # If pause was requested, treat as graceful pause
        if self._pause_requested and not self._pause_executed:
            logger.info(f"Trial {self._trial_id}: Converting exception to pause")
            self._execute_pause(trainer, pl_module)


class TuneResumeCallback(Callback):
    """
    Callback to handle resuming from TunePauseCallback checkpoints.
    
    This callback checks for pause metadata and resumes training
    from the appropriate checkpoint.
    """
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize TuneResumeCallback.
        
        Args:
            checkpoint_dir: Directory containing pause checkpoints
        """
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self._resumed = False
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """Check for and load pause checkpoint."""
        if self._resumed:
            return
        
        # Look for pause metadata
        if self.checkpoint_dir is None:
            checkpoint_dir = Path(trainer.default_root_dir) / "pause_checkpoints"
        else:
            checkpoint_dir = Path(self.checkpoint_dir)
        
        metadata_path = checkpoint_dir / "pause_metadata.json"
        
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
                checkpoint_path = Path(metadata["checkpoint_path"])
                
                if checkpoint_path.exists():
                    logger.info(f"Resuming from pause checkpoint: {checkpoint_path}")
                    
                    # Load checkpoint
                    checkpoint = trainer.checkpoint_callback.load_checkpoint(
                        checkpoint_path,
                        trainer=trainer,
                        pl_module=pl_module
                    )
                    
                    self._resumed = True
                    logger.info(f"Successfully resumed from epoch {metadata['pause_epoch']}")
                    
            except Exception as e:
                logger.warning(f"Could not resume from pause checkpoint: {e}")