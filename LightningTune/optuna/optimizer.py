"""
Optuna-driven optimizer for PyTorch Lightning.

This module provides the main optimizer class that orchestrates
hyperparameter optimization using Optuna with PyTorch Lightning.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union, Type
from abc import ABC, abstractmethod
import logging

import optuna
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from .strategies import OptunaStrategy, TPEStrategy
from .search_space import OptunaSearchSpace
from .callbacks import OptunaPruningCallback

logger = logging.getLogger(__name__)


class OptunaDrivenOptimizer:
    """
    Config-driven optimizer using Optuna for hyperparameter optimization.
    
    This class provides a high-level interface for optimizing PyTorch Lightning
    models using Optuna, with support for various optimization strategies,
    automatic checkpointing, and experiment tracking.
    """
    
    def __init__(
        self,
        base_config: Union[str, Path, Dict[str, Any]],
        search_space: OptunaSearchSpace,
        model_class: Type[LightningModule],
        datamodule_class: Optional[Type[pl.LightningDataModule]] = None,
        strategy: Optional[OptunaStrategy] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        direction: str = "minimize",
        n_trials: int = 100,
        timeout: Optional[float] = None,
        callbacks: Optional[list] = None,
        experiment_dir: Optional[Path] = None,
        save_checkpoints: bool = True,
        metric: str = "val_loss",
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Initialize the Optuna-driven optimizer.
        
        Args:
            base_config: Base configuration (path to YAML/JSON or dict)
            search_space: OptunaSearchSpace instance defining parameters to optimize
            model_class: PyTorch Lightning module class
            datamodule_class: Optional PyTorch Lightning datamodule class
            strategy: Optimization strategy (default: TPEStrategy)
            study_name: Name for the Optuna study
            storage: Storage URL for Optuna (e.g., "sqlite:///study.db")
            direction: Optimization direction ("minimize" or "maximize")
            n_trials: Number of trials to run
            timeout: Time limit for optimization in seconds
            callbacks: Additional Lightning callbacks
            experiment_dir: Directory for saving experiments
            save_checkpoints: Whether to save model checkpoints
            metric: Metric to optimize
            mode: Optimization mode ("min" or "max")
            verbose: Whether to print progress
        """
        self.base_config = self._load_config(base_config)
        self.search_space = search_space
        self.model_class = model_class
        self.datamodule_class = datamodule_class
        self.strategy = strategy or TPEStrategy()
        self.study_name = study_name or f"optuna_study_{self.strategy.name}"
        self.storage = storage
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.callbacks = callbacks or []
        self.experiment_dir = Path(experiment_dir or "./optuna_experiments")
        self.save_checkpoints = save_checkpoints
        self.metric = metric
        self.mode = mode
        self.verbose = verbose
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize study
        self.study = None
        self.best_trial = None
        self.best_checkpoint = None
    
    def _load_config(self, config: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration from file or dict."""
        if isinstance(config, dict):
            return config
        
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif config_path.suffix == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    def _merge_configs(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration updates into base config."""
        result = base.copy()
        
        for key, value in updates.items():
            if '.' in key:
                # Handle nested keys like "model.init_args.learning_rate"
                parts = key.split('.')
                current = result
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                    result[key] = self._merge_configs(result[key], value)
                else:
                    result[key] = value
        
        return result
    
    def _create_objective(self) -> Callable[[optuna.Trial], float]:
        """Create the objective function for Optuna."""
        
        def objective(trial: optuna.Trial) -> float:
            # Get hyperparameters from search space
            params = self.search_space.suggest_params(trial)
            
            # Merge with base config
            config = self._merge_configs(self.base_config, params)
            
            # Save trial config
            trial_dir = self.experiment_dir / f"trial_{trial.number}"
            trial_dir.mkdir(exist_ok=True)
            
            with open(trial_dir / "config.yaml", 'w') as f:
                yaml.dump(config, f)
            
            # Initialize model and datamodule
            model = self.model_class(**config.get("model", {}))
            
            if self.datamodule_class:
                datamodule = self.datamodule_class(**config.get("data", {}))
            else:
                datamodule = None
            
            # Setup callbacks
            callbacks = self.callbacks.copy()
            
            # Add pruning callback
            if hasattr(self.strategy.create_pruner(), '__class__'):
                pruning_callback = OptunaPruningCallback(trial, monitor=self.metric)
                callbacks.append(pruning_callback)
            
            # Get trainer config
            trainer_config = config.get("trainer", {})
            enable_checkpointing = trainer_config.get("enable_checkpointing", True)
            
            # Add checkpoint callback if requested and not disabled in trainer config
            if self.save_checkpoints and enable_checkpointing:
                from pytorch_lightning.callbacks import ModelCheckpoint
                checkpoint_callback = ModelCheckpoint(
                    dirpath=trial_dir / "checkpoints",
                    filename=f"{{epoch}}-{{step}}-{{{self.metric}:.4f}}",
                    monitor=self.metric,
                    mode=self.mode,
                    save_top_k=1,
                )
                callbacks.append(checkpoint_callback)
            
            # Create trainer
            # Override progress bar setting if not explicitly set in config
            if "enable_progress_bar" not in trainer_config:
                trainer_config["enable_progress_bar"] = self.verbose
            trainer = Trainer(
                callbacks=callbacks,
                default_root_dir=trial_dir,
                **trainer_config
            )
            
            # Train model
            try:
                trainer.fit(model, datamodule=datamodule)
                
                # Get metric value
                if self.metric in trainer.callback_metrics:
                    metric_value = trainer.callback_metrics[self.metric].item()
                else:
                    logger.warning(f"Metric {self.metric} not found in callback metrics")
                    metric_value = float('inf') if self.mode == "min" else float('-inf')
                
                # Save best checkpoint path
                if self.save_checkpoints and enable_checkpointing:
                    for callback in callbacks:
                        if hasattr(callback, 'best_model_path'):
                            trial.set_user_attr('checkpoint_path', callback.best_model_path)
                            break
                
                return metric_value
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                return float('inf') if self.mode == "min" else float('-inf')
        
        return objective
    
    def run(self) -> optuna.Study:
        """
        Run the optimization.
        
        Returns:
            The Optuna study object with results
        """
        # Create or load study
        self.study = self.strategy.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            load_if_exists=True
        )
        
        # Create objective
        objective = self._create_objective()
        
        # Run optimization
        if self.verbose:
            print(f"Starting Optuna optimization: {self.study_name}")
            print(f"Strategy: {self.strategy.name}")
            print(f"Trials: {self.n_trials}")
            print(f"Metric: {self.metric} ({self.direction})")
            print("-" * 50)
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=self.verbose,
        )
        
        # Get best trial
        self.best_trial = self.study.best_trial
        
        if self.verbose:
            print("-" * 50)
            print(f"Best trial: {self.best_trial.number}")
            print(f"Best {self.metric}: {self.best_trial.value:.6f}")
            print("\nBest parameters:")
            for key, value in self.best_trial.params.items():
                print(f"  {key}: {value}")
        
        # Get best checkpoint path if available
        if 'checkpoint_path' in self.best_trial.user_attrs:
            self.best_checkpoint = self.best_trial.user_attrs['checkpoint_path']
            if self.verbose:
                print(f"\nBest checkpoint: {self.best_checkpoint}")
        
        return self.study
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get the configuration of the best trial."""
        if not self.best_trial:
            raise ValueError("No optimization has been run yet")
        
        return self._merge_configs(self.base_config, self.best_trial.params)
    
    def save_best_config(self, path: Optional[Path] = None) -> Path:
        """Save the best configuration to a file."""
        if not self.best_trial:
            raise ValueError("No optimization has been run yet")
        
        path = path or self.experiment_dir / "best_config.yaml"
        config = self.get_best_config()
        
        with open(path, 'w') as f:
            yaml.dump(config, f)
        
        return path
    
    def visualize(self) -> None:
        """Generate visualization plots for the optimization."""
        try:
            import optuna.visualization as vis
            
            # Create visualization directory
            viz_dir = self.experiment_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Generate plots
            plots = {
                "optimization_history": vis.plot_optimization_history(self.study),
                "param_importances": vis.plot_param_importances(self.study),
                "parallel_coordinate": vis.plot_parallel_coordinate(self.study),
                "slice": vis.plot_slice(self.study),
            }
            
            # Save plots
            for name, fig in plots.items():
                fig.write_html(viz_dir / f"{name}.html")
            
            if self.verbose:
                print(f"Visualizations saved to: {viz_dir}")
                
        except ImportError:
            logger.warning("Plotly not installed. Skipping visualizations.")
    
    def export_results(self, path: Optional[Path] = None) -> Path:
        """Export optimization results to CSV."""
        if not self.study:
            raise ValueError("No optimization has been run yet")
        
        path = path or self.experiment_dir / "results.csv"
        df = self.study.trials_dataframe()
        df.to_csv(path, index=False)
        
        if self.verbose:
            print(f"Results exported to: {path}")
        
        return path