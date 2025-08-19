"""
Optuna-driven optimizer for PyTorch Lightning using direct dependency injection.

This module provides a clean optimizer that directly uses Optuna's samplers and pruners
without unnecessary abstraction layers.
"""

import os
import json
import yaml
import tempfile
import shutil
import atexit
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union, Type, List
import logging

import optuna
from optuna.samplers import BaseSampler, TPESampler
from optuna.pruners import BasePruner, MedianPruner, NopPruner
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from .search_space import OptunaSearchSpace
from .callbacks import OptunaPruningCallback

logger = logging.getLogger(__name__)


class OptunaDrivenOptimizer:
    """
    Simple, clean optimizer using Optuna with dependency injection.
    
    No unnecessary strategy abstraction - just pass in Optuna's samplers and pruners directly.
    """
    
    def __init__(
        self,
        base_config: Union[str, Path, Dict[str, Any]],
        search_space: OptunaSearchSpace,
        model_class: Type[LightningModule],
        datamodule_class: Optional[Type[pl.LightningDataModule]] = None,
        sampler: Optional[BaseSampler] = None,  # Direct Optuna sampler
        pruner: Optional[BasePruner] = None,    # Direct Optuna pruner
        config_overrides: Optional[Dict[str, Any]] = None,  # Fixed config overrides
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        direction: str = "minimize",
        n_trials: int = 100,
        timeout: Optional[float] = None,
        callbacks: Optional[List[Callback]] = None,
        experiment_dir: Optional[Path] = None,
        save_checkpoints: bool = True,
        metric: str = "val_loss",
        verbose: bool = True,
    ):
        """
        Initialize the optimizer with direct Optuna components.
        
        Args:
            base_config: Base configuration (path to YAML/JSON or dict)
            search_space: OptunaSearchSpace instance defining parameters to optimize
            model_class: PyTorch Lightning module class
            datamodule_class: Optional PyTorch Lightning datamodule class
            sampler: Optuna sampler (e.g., TPESampler, RandomSampler, CmaEsSampler)
                    If None, defaults to TPESampler()
            pruner: Optuna pruner (e.g., MedianPruner, HyperbandPruner, SuccessiveHalvingPruner)
                   If None, defaults to MedianPruner()
            config_overrides: Optional dict of config values to override (not optimized).
                             These are applied after base_config but before search_space suggestions.
                             Useful for reducing epochs/data during HPO.
            study_name: Name for the Optuna study
            storage: Storage URL for Optuna (e.g., "sqlite:///study.db")
            direction: Optimization direction ("minimize" or "maximize")
            n_trials: Number of trials to run
            timeout: Time limit for optimization in seconds
            callbacks: Additional Lightning callbacks
            experiment_dir: Directory for saving experiments. If None, uses a temporary
                           directory that will be cleaned up after optimization
            save_checkpoints: Whether to save model checkpoints
            metric: Metric to optimize
            verbose: Whether to print progress
            
        Example:
            >>> from optuna.samplers import TPESampler
            >>> from optuna.pruners import HyperbandPruner
            >>> 
            >>> optimizer = OptunaDrivenOptimizer(
            ...     base_config="config.yaml",
            ...     search_space=search_space,
            ...     model_class=MyModel,
            ...     sampler=TPESampler(n_startup_trials=10),
            ...     pruner=HyperbandPruner(min_resource=1, max_resource=100)
            ... )
            >>> study = optimizer.optimize()
        """
        self.base_config = self._load_config(base_config)
        self.search_space = search_space
        self.model_class = model_class
        self.datamodule_class = datamodule_class
        self.config_overrides = config_overrides or {}
        
        # Use provided sampler/pruner or defaults
        self.sampler = sampler if sampler is not None else TPESampler()
        self.pruner = pruner if pruner is not None else MedianPruner()
        
        self.study_name = study_name or "optuna_study"
        self.storage = storage
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.callbacks = callbacks or []
        self.save_checkpoints = save_checkpoints
        self.metric = metric
        self.verbose = verbose
        
        # Setup experiment directory
        self._temp_dir = None
        if experiment_dir is None:
            # Create temporary directory
            self._temp_dir = tempfile.mkdtemp(prefix=f"{study_name}_")
            self.experiment_dir = Path(self._temp_dir)
            if self.verbose:
                logger.info(f"📁 Using temporary directory: {self.experiment_dir}")
                logger.info("   Results will be cleaned up after optimization")
            # Register cleanup function
            atexit.register(self._cleanup_temp_dir)
        else:
            self.experiment_dir = Path(experiment_dir)
            if self.verbose:
                logger.info(f"📁 Using persistent directory: {self.experiment_dir}")
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize study
        self.study = None
        self.best_trial = None
        self.best_checkpoint = None
    
    def _cleanup_temp_dir(self):
        """Clean up temporary directory if it was created."""
        if self._temp_dir and Path(self._temp_dir).exists():
            try:
                shutil.rmtree(self._temp_dir)
                if self.verbose:
                    logger.info(f"🧹 Cleaned up temporary directory: {self._temp_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up temporary directory: {e}")
    
    def _load_config(self, config_source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration from file or dict."""
        if isinstance(config_source, dict):
            return config_source
        
        config_path = Path(config_source)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif config_path.suffix == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    def create_objective(self) -> Callable[[optuna.Trial], float]:
        """
        Create the objective function for Optuna.
        
        Returns:
            Objective function that takes a trial and returns a metric value
        """
        def objective(trial: optuna.Trial) -> float:
            # Start with base config
            config = self.base_config.copy()
            
            # Apply fixed config overrides first
            if self.config_overrides:
                config = self._merge_configs(config, self.config_overrides)
            
            # Then apply suggested hyperparameters from search space
            suggested_params = self.search_space.suggest_params(trial)
            config = self._merge_configs(config, suggested_params)
            
            # Create model and datamodule
            # Handle LightningCLI-style config with class_path and init_args
            model_config = config.get('model', {})
            if 'init_args' in model_config:
                model_args = model_config['init_args']
            else:
                model_args = model_config
            
            data_config = config.get('data', {})
            if 'init_args' in data_config:
                data_args = data_config['init_args']
            else:
                data_args = data_config
                
            model = self.model_class(**model_args)
            
            if self.datamodule_class:
                datamodule = self.datamodule_class(**data_args)
            else:
                datamodule = None
            
            # Setup callbacks
            callbacks = list(self.callbacks)
            
            # Add pruning callback if pruner is not NopPruner
            if not isinstance(self.pruner, NopPruner):
                pruning_callback = OptunaPruningCallback(trial, monitor=self.metric)
                callbacks.append(pruning_callback)
            
            # Create trainer config
            trainer_config = config.get('trainer', {})
            
            # Add checkpoint callback if requested
            if self.save_checkpoints:
                from pytorch_lightning.callbacks import ModelCheckpoint
                checkpoint_callback = ModelCheckpoint(
                    dirpath=self.experiment_dir / f"trial_{trial.number}",
                    filename="{epoch}-{val_loss:.2f}",
                    monitor=self.metric,
                    mode="min" if self.direction == "minimize" else "max",
                    save_top_k=1,
                )
                callbacks.append(checkpoint_callback)
                # Ensure checkpointing is enabled if we're adding a checkpoint callback
                trainer_config['enable_checkpointing'] = True
            
            # Remove any conflicting parameters
            trainer_config.pop('callbacks', None)
            trainer_config.pop('enable_progress_bar', None)
            
            trainer = Trainer(
                callbacks=callbacks,
                enable_progress_bar=self.verbose,
                **trainer_config
            )
            
            # Train model
            try:
                if datamodule:
                    trainer.fit(model, datamodule=datamodule)
                else:
                    trainer.fit(model)
                
                # Return the metric value
                if self.metric in trainer.callback_metrics:
                    return trainer.callback_metrics[self.metric].item()
                else:
                    logger.warning(f"Metric {self.metric} not found in callback_metrics")
                    return float('inf') if self.direction == "minimize" else float('-inf')
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                return float('inf') if self.direction == "minimize" else float('-inf')
        
        return objective
    
    def _merge_configs(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration updates into base config."""
        result = base.copy()
        
        for key, value in updates.items():
            if '.' in key:
                # Handle nested keys like "model.learning_rate"
                parts = key.split('.')
                current = result
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                result[key] = value
        
        return result
    
    def optimize(self) -> optuna.Study:
        """
        Run the optimization.
        
        Returns:
            The Optuna study object with results
        """
        # Create or load study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=self.sampler,
            pruner=self.pruner,
            direction=self.direction,
            load_if_exists=True
        )
        
        # Create objective
        objective = self.create_objective()
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=self.verbose
        )
        
        # Store best trial
        self.best_trial = self.study.best_trial
        
        if self.verbose:
            print(f"\nBest trial: {self.best_trial.number}")
            print(f"Best value: {self.best_trial.value}")
            print(f"Best params: {self.best_trial.params}")
        
        # Save results
        results_file = self.experiment_dir / "best_params.json"
        with open(results_file, 'w') as f:
            json.dump({
                "trial_number": self.best_trial.number,
                "value": self.best_trial.value,
                "params": self.best_trial.params,
            }, f, indent=2)
        
        return self.study
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get the configuration of the best trial."""
        if not self.best_trial:
            raise ValueError("No optimization has been run yet")
        
        config = self.base_config.copy()
        return self._merge_configs(config, self.best_trial.params)
    
    def resume(self) -> optuna.Study:
        """Resume optimization from a previous run."""
        if not self.storage:
            raise ValueError("Cannot resume without storage. Set storage parameter.")
        
        return self.optimize()