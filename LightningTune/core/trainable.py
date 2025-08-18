"""
Lightning-compatible trainable for Ray Tune BOHB optimization.

This module provides a generic trainable that works with any Lightning module
and data module, with optional LightningReflow integration.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Type, Callable, Union
import yaml

import torch
import lightning.pytorch as pl
from lightning.pytorch import LightningModule, LightningDataModule
from lightning.pytorch.callbacks import Callback

from ray import tune
from ray.air import session

from .config import ConfigManager

logger = logging.getLogger(__name__)


class LightningBOHBTrainable:
    """
    Generic trainable class for BOHB optimization with PyTorch Lightning.
    
    This trainable is designed to work with any Lightning module and data module,
    providing a clean interface between BOHB and Lightning training.
    """
    
    def __init__(
        self,
        model_class: Optional[Type[LightningModule]] = None,
        datamodule_class: Optional[Type[LightningDataModule]] = None,
        model_factory: Optional[Callable[[Dict], LightningModule]] = None,
        datamodule_factory: Optional[Callable[[Dict], LightningDataModule]] = None,
        base_config_path: Optional[Union[str, Path]] = None,
        use_lightning_reflow: bool = False,
        lightning_reflow_kwargs: Optional[Dict[str, Any]] = None,
        callbacks: Optional[list[Callback]] = None,
        trainer_defaults: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the trainable.
        
        Args:
            model_class: Lightning module class (if using class instantiation)
            datamodule_class: Lightning data module class (if using class instantiation)
            model_factory: Factory function to create model (if using factory pattern)
            datamodule_factory: Factory function to create data module (if using factory pattern)
            base_config_path: Path to base configuration file
            use_lightning_reflow: Whether to use LightningReflow for training
            lightning_reflow_kwargs: Additional kwargs for LightningReflow
            callbacks: Additional callbacks to include in training
            trainer_defaults: Default trainer configuration
        """
        # Validate inputs
        if model_class is None and model_factory is None:
            raise ValueError("Either model_class or model_factory must be provided")
        if model_class is not None and model_factory is not None:
            raise ValueError("Cannot provide both model_class and model_factory")
            
        self.model_class = model_class
        self.datamodule_class = datamodule_class
        self.model_factory = model_factory
        self.datamodule_factory = datamodule_factory
        self.base_config_path = base_config_path
        self.use_lightning_reflow = use_lightning_reflow
        self.lightning_reflow_kwargs = lightning_reflow_kwargs or {}
        self.callbacks = callbacks or []
        self.trainer_defaults = trainer_defaults or {}
        
        # Configuration manager
        self.config_manager = ConfigManager(base_config_path)
        
        # Training components (initialized in setup)
        self.model = None
        self.datamodule = None
        self.trainer = None
        self.reflow = None
        
    def setup(self, config: Dict[str, Any]) -> None:
        """
        Setup training components with BOHB-provided configuration.
        
        Args:
            config: Configuration from BOHB
        """
        # Get trial directory
        trial_dir = Path(session.get_trial_dir())
        
        # Create trial configuration
        config_path = self.config_manager.create_trial_config(config, trial_dir)
        
        # Merge configurations
        merged_config = self.config_manager.merge_configs(
            self.config_manager.base_config,
            config
        )
        
        if self.use_lightning_reflow:
            self._setup_with_reflow(config_path, merged_config)
        else:
            self._setup_standard(merged_config, trial_dir)
    
    def _setup_with_reflow(self, config_path: Path, merged_config: Dict[str, Any]) -> None:
        """Setup using LightningReflow."""
        try:
            from lightning_reflow import LightningReflow
            from lightning_reflow.callbacks import PauseCallback
        except ImportError:
            raise ImportError(
                "LightningReflow not found. Install it or set use_lightning_reflow=False"
            )
        
        # Add BOHB callback to report metrics
        from ..callbacks.report import BOHBReportCallback
        
        callbacks = self.callbacks.copy()
        callbacks.append(BOHBReportCallback())
        callbacks.append(PauseCallback(
            checkpoint_dir=str(Path(session.get_trial_dir()) / "checkpoints"),
            enable_pause=True,
            show_pause_countdown=False
        ))
        
        # Initialize LightningReflow
        self.reflow = LightningReflow(
            model_class=self.model_class,
            datamodule_class=self.datamodule_class,
            config_files=str(config_path),
            callbacks=callbacks,
            trainer_defaults={
                **self.trainer_defaults,
                "default_root_dir": session.get_trial_dir(),
                "enable_progress_bar": False,  # Cleaner output for BOHB
            },
            **self.lightning_reflow_kwargs
        )
    
    def _setup_standard(self, config: Dict[str, Any], trial_dir: Path) -> None:
        """Setup using standard Lightning components."""
        # Create model
        if self.model_factory:
            self.model = self.model_factory(config)
        else:
            model_config = config.get("model", {})
            if isinstance(model_config, dict) and "init_args" in model_config:
                self.model = self.model_class(**model_config["init_args"])
            else:
                self.model = self.model_class(**model_config)
        
        # Create data module
        if self.datamodule_factory:
            self.datamodule = self.datamodule_factory(config)
        elif self.datamodule_class:
            data_config = config.get("data", {})
            if isinstance(data_config, dict) and "init_args" in data_config:
                self.datamodule = self.datamodule_class(**data_config["init_args"])
            else:
                self.datamodule = self.datamodule_class(**data_config)
        
        # Add BOHB callback
        from ..callbacks.report import BOHBReportCallback
        callbacks = self.callbacks.copy()
        callbacks.append(BOHBReportCallback())
        
        # Create trainer
        trainer_config = config.get("trainer", {})
        trainer_config = {
            **self.trainer_defaults,
            **trainer_config,
            "default_root_dir": str(trial_dir),
            "callbacks": callbacks,
            "enable_progress_bar": False,
        }
        
        self.trainer = pl.Trainer(**trainer_config)
    
    def train(self) -> Dict[str, Any]:
        """
        Execute one training iteration.
        
        Returns:
            Dictionary of metrics to report to BOHB
        """
        if self.use_lightning_reflow:
            return self._train_with_reflow()
        else:
            return self._train_standard()
    
    def _train_with_reflow(self) -> Dict[str, Any]:
        """Train using LightningReflow."""
        # Check for checkpoint to resume from
        checkpoint = session.get_checkpoint()
        
        if checkpoint and "checkpoint_path" in checkpoint:
            result = self.reflow.resume(checkpoint["checkpoint_path"])
        else:
            result = self.reflow.fit()
        
        # Extract metrics from result
        if isinstance(result, dict):
            return result
        else:
            # Return empty dict if no metrics available
            return {}
    
    def _train_standard(self) -> Dict[str, Any]:
        """Train using standard Lightning."""
        # Check for checkpoint to resume from
        checkpoint = session.get_checkpoint()
        ckpt_path = None
        
        if checkpoint and "checkpoint_path" in checkpoint:
            ckpt_path = checkpoint["checkpoint_path"]
        
        # Train
        self.trainer.fit(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=ckpt_path
        )
        
        # Return metrics from callback metrics
        metrics = {}
        for key, value in self.trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = float(value)
            else:
                metrics[key] = value
        
        return metrics
    
    def save_checkpoint(self) -> Dict[str, str]:
        """
        Save checkpoint for BOHB.
        
        Returns:
            Dictionary with checkpoint path
        """
        checkpoint_dir = Path(session.get_trial_dir()) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        if self.use_lightning_reflow:
            # LightningReflow handles checkpointing
            checkpoint_path = checkpoint_dir / "latest.ckpt"
            # Find the most recent checkpoint
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if checkpoints:
                checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        else:
            # Save using trainer
            checkpoint_path = checkpoint_dir / f"epoch_{self.trainer.current_epoch}.ckpt"
            self.trainer.save_checkpoint(checkpoint_path)
        
        return {"checkpoint_path": str(checkpoint_path)}
    
    def cleanup(self) -> None:
        """Cleanup resources after training."""
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear references
        self.model = None
        self.datamodule = None
        self.trainer = None
        self.reflow = None


class TrainableFactory:
    """
    Factory for creating trainables with different configurations.
    
    This allows for easy creation of trainables with preset configurations.
    """
    
    @staticmethod
    def create_simple_trainable(
        model_class: Type[LightningModule],
        datamodule_class: Type[LightningDataModule],
        **kwargs
    ) -> Callable:
        """
        Create a simple trainable function for Ray Tune.
        
        Args:
            model_class: Lightning module class
            datamodule_class: Lightning data module class
            **kwargs: Additional arguments for LightningBOHBTrainable
            
        Returns:
            Trainable function for Ray Tune
        """
        def trainable_fn(config: Dict[str, Any]) -> Dict[str, Any]:
            trainable = LightningBOHBTrainable(
                model_class=model_class,
                datamodule_class=datamodule_class,
                **kwargs
            )
            trainable.setup(config)
            metrics = trainable.train()
            checkpoint = trainable.save_checkpoint()
            
            # Report to Ray Tune
            session.report(metrics, checkpoint=checkpoint)
            
            trainable.cleanup()
            return metrics
        
        return trainable_fn
    
    @staticmethod
    def create_reflow_trainable(
        model_class: Type[LightningModule],
        datamodule_class: Type[LightningDataModule],
        base_config_path: Union[str, Path],
        **kwargs
    ) -> Callable:
        """
        Create a trainable using LightningReflow.
        
        Args:
            model_class: Lightning module class
            datamodule_class: Lightning data module class
            base_config_path: Path to base configuration
            **kwargs: Additional arguments for LightningBOHBTrainable
            
        Returns:
            Trainable function for Ray Tune
        """
        def trainable_fn(config: Dict[str, Any]) -> Dict[str, Any]:
            trainable = LightningBOHBTrainable(
                model_class=model_class,
                datamodule_class=datamodule_class,
                base_config_path=base_config_path,
                use_lightning_reflow=True,
                **kwargs
            )
            trainable.setup(config)
            metrics = trainable.train()
            checkpoint = trainable.save_checkpoint()
            
            # Report to Ray Tune
            session.report(metrics, checkpoint=checkpoint)
            
            trainable.cleanup()
            return metrics
        
        return trainable_fn
    
    @staticmethod
    def create_factory_trainable(
        model_factory: Callable[[Dict], LightningModule],
        datamodule_factory: Callable[[Dict], LightningDataModule],
        **kwargs
    ) -> Callable:
        """
        Create a trainable using factory functions.
        
        This is useful when model/data module creation requires complex logic.
        
        Args:
            model_factory: Function to create model from config
            datamodule_factory: Function to create data module from config
            **kwargs: Additional arguments for LightningBOHBTrainable
            
        Returns:
            Trainable function for Ray Tune
        """
        def trainable_fn(config: Dict[str, Any]) -> Dict[str, Any]:
            trainable = LightningBOHBTrainable(
                model_factory=model_factory,
                datamodule_factory=datamodule_factory,
                **kwargs
            )
            trainable.setup(config)
            metrics = trainable.train()
            checkpoint = trainable.save_checkpoint()
            
            # Report to Ray Tune
            session.report(metrics, checkpoint=checkpoint)
            
            trainable.cleanup()
            return metrics
        
        return trainable_fn