"""
Simplified Optuna-driven optimizer for PyTorch Lightning using dependency injection.

This module provides a clean optimizer that directly uses Optuna's samplers and pruners
without unnecessary abstraction layers.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union, Type
import logging

import optuna
from optuna.samplers import BaseSampler, TPESampler
from optuna.pruners import BasePruner, MedianPruner
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from .search_space import OptunaSearchSpace
from .callbacks import OptunaPruningCallback
from ..utils.config_utils import apply_dotted_updates

logger = logging.getLogger(__name__)


class OptunaOptimizer:
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
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        direction: str = "minimize",
        n_trials: int = 100,
        timeout: Optional[float] = None,
        callbacks: Optional[list] = None,
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
            pruner: Optuna pruner (e.g., MedianPruner, HyperbandPruner, SuccessiveHalvingPruner)
            study_name: Name for the Optuna study
            storage: Storage URL for Optuna (e.g., "sqlite:///study.db")
            direction: Optimization direction ("minimize" or "maximize")
            n_trials: Number of trials to run
            timeout: Time limit for optimization in seconds
            callbacks: Additional Lightning callbacks
            experiment_dir: Directory for saving experiments
            save_checkpoints: Whether to save model checkpoints
            metric: Metric to optimize
            verbose: Whether to print progress
        """
        self.base_config = self._load_config(base_config)
        self.search_space = search_space
        self.model_class = model_class
        self.datamodule_class = datamodule_class
        
        # Use provided sampler/pruner or defaults
        self.sampler = sampler or TPESampler()
        self.pruner = pruner or MedianPruner()
        
        self.study_name = study_name or "optuna_study"
        self.storage = storage
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.callbacks = callbacks or []
        self.experiment_dir = Path(experiment_dir or "./optuna_experiments")
        self.save_checkpoints = save_checkpoints
        self.metric = metric
        self.verbose = verbose
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize study
        self.study = None
        self.best_trial = None
        self.best_checkpoint = None
    
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
            # Suggest hyperparameters using the search space
            config = self.base_config.copy()
            suggested_params = self.search_space.suggest(trial)
            
            # Merge suggested params into config
            config = apply_dotted_updates(config, suggested_params)
            
            # Create model and datamodule
            model = self.model_class(**config.get('model', {}))
            
            if self.datamodule_class:
                datamodule = self.datamodule_class(**config.get('data', {}))
            else:
                datamodule = None
            
            # Setup callbacks
            callbacks = list(self.callbacks)
            
            # Add pruning callback if pruner is not NopPruner
            if self.pruner and not isinstance(self.pruner, optuna.pruners.NopPruner):
                pruning_callback = OptunaPruningCallback(trial, monitor=self.metric)
                callbacks.append(pruning_callback)
            
            # Create trainer
            trainer_config = config.get('trainer', {})
            trainer = Trainer(
                callbacks=callbacks,
                enable_progress_bar=not self.verbose,
                **trainer_config
            )
            
            # Train model
            try:
                if datamodule:
                    trainer.fit(model, datamodule=datamodule)
                else:
                    trainer.fit(model)
                
                # Return the metric value
                return trainer.callback_metrics[self.metric].item()
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return float('inf') if self.direction == "minimize" else float('-inf')
        
        return objective
    
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
        
        return self.study
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get the configuration of the best trial."""
        if not self.best_trial:
            raise ValueError("No optimization has been run yet")
        
        config = self.base_config.copy()
        return apply_dotted_updates(config, self.best_trial.params)


# Usage example:
"""
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner, SuccessiveHalvingPruner, NopPruner

# Simple TPE optimization
optimizer = OptunaOptimizer(
    base_config="config.yaml",
    search_space=search_space,
    model_class=MyModel,
    sampler=TPESampler(n_startup_trials=10),
    pruner=MedianPruner(n_warmup_steps=5)
)

# TPE with Hyperband (what people call "BOHB-like")
optimizer = OptunaOptimizer(
    base_config="config.yaml",
    search_space=search_space,
    model_class=MyModel,
    sampler=TPESampler(),
    pruner=HyperbandPruner(min_resource=1, max_resource=100)
)

# Random search without pruning
optimizer = OptunaOptimizer(
    base_config="config.yaml",
    search_space=search_space,
    model_class=MyModel,
    sampler=RandomSampler(),
    pruner=NopPruner()
)

# CMA-ES for continuous optimization
optimizer = OptunaOptimizer(
    base_config="config.yaml",
    search_space=search_space,
    model_class=MyModel,
    sampler=CmaEsSampler(),
    pruner=MedianPruner()
)
"""