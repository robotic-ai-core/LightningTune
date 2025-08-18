"""
LightningReflow integration for hyperparameter optimization.

This module provides HPO-aware callbacks and trainable wrappers that integrate
LightningReflow's pause/resume capabilities with Ray Tune and other HPO frameworks.
"""

from typing import Optional, Callable, Dict, Any, Type, Union, List
from pathlib import Path
import logging

# Import from LightningReflow - we assume it's available
try:
    from lightning_reflow import LightningReflow
    from lightning_reflow.callbacks.pause import PauseCallback, PauseAction
    REFLOW_AVAILABLE = True
except ImportError:
    REFLOW_AVAILABLE = False
    # Mock classes for testing when LightningReflow isn't installed
    from lightning.pytorch.callbacks import Callback
    class PauseCallback(Callback):
        def __init__(self, checkpoint_dir="pause_checkpoints", **kwargs):
            super().__init__()
            from pathlib import Path
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.last_checkpoint_path = None
            self._state_machine = None
        def get_last_checkpoint(self):
            return self.last_checkpoint_path
    class PauseAction:
        TOGGLE_PAUSE = "toggle_pause"
    class LightningReflow:
        def __init__(self, **kwargs):
            pass
        def fit(self, **kwargs):
            return {}

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


class ReflowHPOCallback(PauseCallback):
    """
    PauseCallback extension with programmatic control for HPO.
    
    This callback adds the ability to trigger pauses based on an external
    function, making it suitable for HPO frameworks that need to control
    when checkpointing happens.
    
    Args:
        should_pause: Optional function that returns True when training should pause
        checkpoint_dir: Directory for pause checkpoints (default: "pause_checkpoints")
        **kwargs: Additional arguments passed to PauseCallback
    
    Example:
        >>> import ray.tune as tune
        >>> callback = ReflowHPOCallback(
        ...     should_pause=lambda: tune.should_checkpoint()
        ... )
    """
    
    def __init__(
        self,
        should_pause: Optional[Callable[[], bool]] = None,
        checkpoint_dir: str = "pause_checkpoints",
        **kwargs
    ):
        super().__init__(checkpoint_dir=checkpoint_dir, **kwargs)
        self.should_pause_fn = should_pause
        self._pause_triggered = False
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Check for programmatic pause at validation boundary."""
        # Call parent's validation end logic first if available
        if hasattr(super(), 'on_validation_end'):
            super().on_validation_end(trainer, pl_module)
        
        # Check if we should pause programmatically
        if self.should_pause_fn and self.should_pause_fn():
            if not self._pause_triggered:
                logger.info("HPO: Triggering programmatic pause at validation boundary")
                if hasattr(self, '_state_machine') and self._state_machine is not None:
                    self._state_machine.transition(PauseAction.TOGGLE_PAUSE)
                self._pause_triggered = True
                # The pause will be executed by parent class logic
    
    def reset_pause_trigger(self):
        """Reset the pause trigger for next use."""
        self._pause_triggered = False


class ReflowTrainable:
    """
    LightningReflow wrapper for HPO frameworks like Ray Tune.
    
    This class wraps LightningReflow to make it compatible with HPO frameworks
    by providing a callable interface and handling checkpoint/resume logic.
    
    Args:
        model_class: PyTorch Lightning module class
        datamodule_class: Optional PyTorch Lightning datamodule class
        base_config: Base configuration dictionary
        should_pause: Optional function that returns True when training should pause
        reflow_callbacks: Additional callbacks to pass to LightningReflow
        **reflow_kwargs: Additional arguments passed to LightningReflow
    
    Example:
        >>> # Create trainable for Ray Tune
        >>> trainable = ReflowTrainable(
        ...     model_class=MyModel,
        ...     datamodule_class=MyDataModule,
        ...     base_config={"trainer": {"max_epochs": 50}},
        ...     should_pause=lambda: tune.should_checkpoint()
        ... )
        >>> 
        >>> # Use in Ray Tune
        >>> tune.run(trainable, config={"lr": tune.loguniform(1e-4, 1e-1)})
    """
    
    def __init__(
        self,
        model_class: Type[pl.LightningModule],
        datamodule_class: Optional[Type[pl.LightningDataModule]] = None,
        base_config: Optional[Dict[str, Any]] = None,
        should_pause: Optional[Callable[[], bool]] = None,
        reflow_callbacks: Optional[List[Callback]] = None,
        **reflow_kwargs
    ):
        if not REFLOW_AVAILABLE:
            logger.warning(
                "LightningReflow is not installed. "
                "Using mock implementation for testing."
            )
        
        self.model_class = model_class
        self.datamodule_class = datamodule_class
        self.base_config = base_config or {}
        self.should_pause = should_pause
        self.reflow_callbacks = reflow_callbacks or []
        self.reflow_kwargs = reflow_kwargs
        
        # Track last checkpoint for convenience
        self.last_checkpoint_path = None
    
    def __call__(
        self,
        config: Dict[str, Any],
        checkpoint_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Train with the given config, making this callable for HPO frameworks.
        
        Args:
            config: Hyperparameter configuration from HPO framework
            checkpoint_dir: Optional checkpoint directory to resume from
        
        Returns:
            Dictionary with training results including:
                - checkpoint_path: Path to saved checkpoint if paused
                - metrics: Training metrics
                - val_loss: Validation loss (for HPO metric)
        """
        # Merge base config with trial config
        full_config = {**self.base_config, **config}
        
        # Create HPO-aware pause callback
        pause_callback = ReflowHPOCallback(should_pause=self.should_pause)
        
        # Combine callbacks
        all_callbacks = self.reflow_callbacks + [pause_callback]
        
        # Create LightningReflow instance
        reflow = LightningReflow(
            model_class=self.model_class,
            datamodule_class=self.datamodule_class,
            config_overrides=full_config,
            callbacks=all_callbacks,
            **self.reflow_kwargs
        )
        
        # Convert checkpoint_dir to string if needed
        ckpt_path = str(checkpoint_dir) if checkpoint_dir else None
        
        # Train
        result = reflow.fit(ckpt_path=ckpt_path)
        
        # Get checkpoint path if paused
        checkpoint_path = pause_callback.get_last_checkpoint()
        if checkpoint_path:
            self.last_checkpoint_path = str(checkpoint_path)
            logger.info(f"HPO: Training paused, checkpoint saved at {checkpoint_path}")
        
        # Extract metrics
        metrics = {}
        if hasattr(reflow, 'trainer'):
            # Try to get metrics from callback_metrics
            if hasattr(reflow.trainer, '_callback_metrics'):
                # Access private attribute for mock compatibility
                try:
                    raw_metrics = reflow.trainer._callback_metrics
                    # Convert tensors to floats
                    metrics = {}
                    for k, v in raw_metrics.items():
                        if hasattr(v, 'item'):  # It's a tensor
                            metrics[k] = v.item()
                        else:
                            metrics[k] = v
                except:
                    pass
            elif hasattr(reflow.trainer, 'callback_metrics'):
                # Try public property
                try:
                    raw_metrics = reflow.trainer.callback_metrics
                    # Convert tensors to floats
                    metrics = {}
                    for k, v in raw_metrics.items():
                        if hasattr(v, 'item'):  # It's a tensor
                            metrics[k] = v.item()
                        else:
                            metrics[k] = v
                except:
                    pass
            # Fallback: try to get from logged metrics
            if not metrics and hasattr(reflow.trainer, 'logged_metrics'):
                try:
                    metrics = dict(reflow.trainer.logged_metrics)
                except:
                    pass
        
        # Return HPO-friendly result
        return {
            "checkpoint_path": self.last_checkpoint_path,
            "metrics": metrics,
            "val_loss": metrics.get("val_loss", float('inf')),
            "trainer_result": result
        }
    
    def get_last_checkpoint(self) -> Optional[str]:
        """Get the path to the last saved checkpoint."""
        return self.last_checkpoint_path


def create_reflow_trainable(
    model_class: Type[pl.LightningModule],
    datamodule_class: Optional[Type[pl.LightningDataModule]] = None,
    base_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ReflowTrainable:
    """
    Factory function to create a Ray Tune compatible trainable.
    
    This convenience function creates a ReflowTrainable with Ray Tune's
    checkpoint signal automatically configured.
    
    Args:
        model_class: PyTorch Lightning module class
        datamodule_class: Optional datamodule class
        base_config: Base configuration dictionary
        **kwargs: Additional arguments passed to ReflowTrainable
    
    Returns:
        ReflowTrainable configured for Ray Tune
    
    Example:
        >>> trainable = create_reflow_trainable(
        ...     model_class=MyModel,
        ...     base_config={"trainer": {"max_epochs": 50}}
        ... )
        >>> tune.run(trainable, config={"lr": tune.loguniform(1e-4, 1e-1)})
    """
    try:
        import ray.tune as tune
        should_pause = lambda: tune.should_checkpoint()
    except ImportError:
        logger.warning("Ray Tune not installed, should_pause will default to False")
        should_pause = lambda: False
    
    return ReflowTrainable(
        model_class=model_class,
        datamodule_class=datamodule_class,
        base_config=base_config,
        should_pause=should_pause,
        **kwargs
    )