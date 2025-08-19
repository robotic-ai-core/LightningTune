"""
Optuna-driven optimizer using LightningReflow for proper environment setup and compilation.

This module provides an optimizer that uses LightningReflow instead of vanilla Lightning,
ensuring HPO trials run with the same optimizations as standalone training.
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
import lightning as L
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

# Import LightningReflow for proper training orchestration
import sys
lightning_reflow_path = Path(__file__).parent.parent.parent.parent / "LightningReflow"
if lightning_reflow_path.exists():
    sys.path.insert(0, str(lightning_reflow_path))
from lightning_reflow import LightningReflow

from .search_space import OptunaSearchSpace
from .callbacks import OptunaPruningCallback
from ..utils.config_utils import apply_dotted_updates

logger = logging.getLogger(__name__)


class ReflowOptunaDrivenOptimizer:
    """
    Optuna optimizer using LightningReflow for consistent training environment.
    
    This optimizer ensures that HPO trials run with the same optimizations as standalone
    training, including:
    - PyTorch compilation (if configured)
    - Environment variable setup (CUDA configs, etc.)
    - Proper callback management
    - Consistent configuration handling
    """
    
    def __init__(
        self,
        base_config: Union[str, Path, Dict[str, Any]],
        search_space: Union[OptunaSearchSpace, Callable[[optuna.Trial], Dict[str, Any]]],
        model_class: Type,  # Type[LightningModule]
        datamodule_class: Optional[Type] = None,  # Type[LightningDataModule]
        sampler: Optional[BaseSampler] = None,
        pruner: Optional[BasePruner] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
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
        wandb_project: Optional[str] = None,
        upload_checkpoints: bool = False,
        use_reflow: bool = True,  # Option to fall back to vanilla Lightning
    ):
        """
        Initialize the optimizer with LightningReflow support.
        
        Args:
            base_config: Base configuration (path to YAML/JSON or dict)
            search_space: OptunaSearchSpace instance or callable function
            model_class: PyTorch Lightning module class
            datamodule_class: Optional PyTorch Lightning datamodule class
            sampler: Optuna sampler
            pruner: Optuna pruner
            config_overrides: Fixed config overrides (applied before search space)
            study_name: Name for the Optuna study
            storage: Storage URL for Optuna
            direction: Optimization direction
            n_trials: Number of trials to run
            timeout: Time limit for optimization
            callbacks: Additional Lightning callbacks
            experiment_dir: Directory for saving experiments
            save_checkpoints: Whether to save model checkpoints
            metric: Metric to optimize
            verbose: Whether to print progress
            wandb_project: Optional WandB project name
            upload_checkpoints: Whether to upload checkpoints to WandB
            use_reflow: Whether to use LightningReflow (True) or vanilla Lightning (False)
        """
        self.base_config = self._load_config(base_config)
        self.search_space = search_space
        self.model_class = model_class
        self.datamodule_class = datamodule_class
        self.config_overrides = config_overrides or {}
        self.use_reflow = use_reflow
        
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
        self.wandb_project = wandb_project
        self.upload_checkpoints = upload_checkpoints
        
        # Setup experiment directory
        self._temp_dir = None
        if experiment_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix=f"{study_name}_")
            self.experiment_dir = Path(self._temp_dir)
            if self.verbose:
                logger.info(f"📁 Using temporary directory: {self.experiment_dir}")
            atexit.register(self._cleanup_temp_dir)
        else:
            self.experiment_dir = Path(experiment_dir)
            if self.verbose:
                logger.info(f"📁 Using persistent directory: {self.experiment_dir}")
        
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
        Create the objective function using LightningReflow.
        
        Returns:
            Objective function that takes a trial and returns a metric value
        """
        def objective(trial: optuna.Trial) -> float:
            # Start with base config
            config = self.base_config.copy()
            
            # Apply fixed config overrides first
            if self.config_overrides:
                config = apply_dotted_updates(config, self.config_overrides)
            
            # Then apply suggested hyperparameters from search space
            if callable(self.search_space) and not hasattr(self.search_space, 'suggest_params'):
                suggested_params = self.search_space(trial)
            else:
                suggested_params = self.search_space.suggest_params(trial)
            config = apply_dotted_updates(config, suggested_params)
            
            # Setup callbacks
            callbacks = list(self.callbacks)
            
            # Add pruning callback if pruner is not NopPruner
            if not isinstance(self.pruner, NopPruner):
                pruning_callback = OptunaPruningCallback(trial, monitor=self.metric)
                callbacks.append(pruning_callback)
            
            # Add checkpoint callback if requested
            if self.save_checkpoints:
                from lightning.pytorch.callbacks import ModelCheckpoint
                checkpoint_callback = ModelCheckpoint(
                    dirpath=self.experiment_dir / f"trial_{trial.number}",
                    filename="{epoch}-{val_loss:.2f}",
                    monitor=self.metric,
                    mode="min" if self.direction == "minimize" else "max",
                    save_top_k=1,
                )
                callbacks.append(checkpoint_callback)
            
            # Configure trainer settings
            trainer_config = config.get('trainer', {})
            trainer_config.pop('callbacks', None)  # Remove any existing callbacks config
            trainer_config.pop('logger', None)  # Will be set by Reflow or below
            
            # For HPO, we can keep Lightning's default progress bar since we only pause at trial boundaries
            # Override Reflow's default behavior of disabling the progress bar
            if 'enable_progress_bar' not in trainer_config:
                trainer_config['enable_progress_bar'] = True  # Keep Lightning's progress bar for HPO
            
            # Setup WandB logger if requested
            if self.wandb_project:
                from lightning.pytorch.loggers import WandbLogger
                wandb_logger = WandbLogger(
                    project=self.wandb_project,
                    name=f"{self.study_name}_trial_{trial.number}",
                    config=config,
                    log_model=self.upload_checkpoints,
                )
                trainer_config['logger'] = wandb_logger
            
            if self.use_reflow:
                # Use LightningReflow for proper environment setup and compilation
                try:
                    # Extract model and data configs
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
                    
                    # Create LightningReflow instance
                    reflow = LightningReflow(
                        model_class=self.model_class,
                        datamodule_class=self.datamodule_class,
                        model_init_args=model_args,
                        datamodule_init_args=data_args,
                        trainer_defaults=trainer_config,
                        callbacks=callbacks,
                        seed_everything=config.get('seed_everything', None),
                        # Pass the full config for environment setup
                        config_overrides=config,
                        auto_configure_logging=False  # We handle logging ourselves
                    )
                    
                    # Run training
                    result = reflow.fit()
                    
                    # Get the metric value from trainer
                    if hasattr(reflow.trainer, 'callback_metrics') and self.metric in reflow.trainer.callback_metrics:
                        return reflow.trainer.callback_metrics[self.metric].item()
                    else:
                        logger.warning(f"Metric {self.metric} not found in callback_metrics")
                        return float('inf') if self.direction == "minimize" else float('-inf')
                    
                except optuna.TrialPruned:
                    raise
                except Exception as e:
                    logger.error(f"Trial {trial.number} failed with Reflow: {e}")
                    # Optionally fall back to vanilla Lightning
                    if self.verbose:
                        logger.info("Falling back to vanilla Lightning")
                    return self._run_vanilla_lightning(config, callbacks, trainer_config, trial)
            else:
                # Use vanilla Lightning (original implementation)
                return self._run_vanilla_lightning(config, callbacks, trainer_config, trial)
        
        return objective
    
    def _run_vanilla_lightning(self, config, callbacks, trainer_config, trial):
        """Run training with vanilla Lightning (fallback or when use_reflow=False)."""
        # Extract model and data configs
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
        
        # Create model and datamodule
        model = self.model_class(**model_args)
        
        # Manually trigger compilation if configured (since we're not using Reflow)
        if hasattr(model, '_apply_torch_compile'):
            model._apply_torch_compile()
        
        if self.datamodule_class:
            datamodule = self.datamodule_class(**data_args)
        else:
            datamodule = None
        
        # Create trainer
        trainer = Trainer(
            callbacks=callbacks,
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
        if self.verbose:
            print(f"\n🔬 Running {self.n_trials} trials with {'LightningReflow' if self.use_reflow else 'vanilla Lightning'}...")
            if self.use_reflow:
                print("   ✓ Environment variables will be set properly")
                print("   ✓ PyTorch compilation will be applied if configured")
                print("   ✓ Using Reflow's callback management")
        
        for i in range(self.n_trials):
            if self.verbose:
                print(f"\n📊 Trial {i+1}/{self.n_trials}")
            
            # Run single trial
            self.study.optimize(
                objective,
                n_trials=1,
                timeout=self.timeout if i == self.n_trials - 1 else None,
                show_progress_bar=False
            )
            
            # Report current best after each trial
            if self.verbose and self.study.best_trial:
                print(f"   Current best value: {self.study.best_value:.6f} (trial {self.study.best_trial.number})")
        
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
        return apply_dotted_updates(config, self.best_trial.params)
    
    def resume(self) -> optuna.Study:
        """Resume optimization from a previous run."""
        if not self.storage:
            raise ValueError("Cannot resume without storage. Set storage parameter.")
        
        return self.optimize()