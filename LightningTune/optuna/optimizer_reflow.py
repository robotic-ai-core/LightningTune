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
import torch
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
from ..utils.config_utils import apply_dotted_updates, load_config

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
    
    def _reset_torch_compile_state(self):
        """Reset torch compile state between trials to prevent interference."""
        try:
            import gc

            # Reset torch._dynamo state if available
            if hasattr(torch, '_dynamo'):
                # Clear dynamo cache
                if hasattr(torch._dynamo, 'reset'):
                    torch._dynamo.reset()

                # Reset config to defaults if modified
                if hasattr(torch._dynamo, 'config'):
                    # Common settings that might be modified
                    default_settings = {
                        'cache_size_limit': 64,
                        'recompile_limit': 8,
                    }
                    for key, default_value in default_settings.items():
                        if hasattr(torch._dynamo.config, key):
                            setattr(torch._dynamo.config, key, default_value)

            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Reset CUDA state more aggressively
                torch.cuda.ipc_collect()
                # Reset all CUDA RNG states
                torch.cuda.manual_seed_all(torch.initial_seed())

            # Force garbage collection
            gc.collect()

            if self.verbose:
                logger.debug("Reset torch compile state between trials")

        except Exception as e:
            logger.debug(f"Could not reset torch compile state: {e}")

    def _load_config(self, config_source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration from file or dict."""
        return load_config(config_source)

    def _extract_config_args(self, config: Dict[str, Any]) -> tuple:
        """
        Extract model and data args from config.

        Handles both flat configs and configs with 'init_args' structure.

        Returns:
            Tuple of (model_args, data_args)
        """
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

        return model_args, data_args

    def _extract_metric_value(self, trainer, reflow=None) -> float:
        """
        Extract the target metric value from trainer's callback_metrics.

        Args:
            trainer: Lightning Trainer instance
            reflow: Optional LightningReflow instance (if using Reflow path)

        Returns:
            Metric value as float, or default value if metric not found
        """
        default_value = float('inf') if self.direction == "minimize" else float('-inf')

        # Handle both Reflow and vanilla Lightning paths
        if reflow is not None:
            # Reflow path: check reflow.trainer.callback_metrics
            if hasattr(reflow, 'trainer') and hasattr(reflow.trainer, 'callback_metrics'):
                if self.metric in reflow.trainer.callback_metrics:
                    return reflow.trainer.callback_metrics[self.metric].item()
        else:
            # Vanilla path: check trainer.callback_metrics directly
            if hasattr(trainer, 'callback_metrics') and self.metric in trainer.callback_metrics:
                return trainer.callback_metrics[self.metric].item()

        logger.warning(f"Metric {self.metric} not found in callback_metrics")
        return default_value

    def _finalize_wandb(self, wandb_logger, status: str):
        """
        Finalize WandB logger with proper cleanup.

        Args:
            wandb_logger: WandB logger instance
            status: Trial status ('success', 'pruned', 'failed')
        """
        if wandb_logger:
            from ..utils.wandb_logger import finalize_wandb_logger
            finalize_wandb_logger(wandb_logger, status)

    def _cleanup_dataloader_workers(self, trial_number: int, trainer=None, datamodule=None, reflow=None, status: str = "completed"):
        """
        Clean up DataLoader workers to prevent thread accumulation.

        NOTE: This cleanup is now primarily handled by LightningReflow.fit() finally block
        and MemoryCleanupCallback. This method is kept for backward compatibility but
        will be removed in future versions.

        Args:
            trial_number: Trial number for logging
            trainer: Lightning Trainer instance
            datamodule: DataModule instance
            reflow: Optional LightningReflow instance
            status: Trial status for logging ('completed', 'pruned', 'failed')
        """
        # Skip if no trainer/datamodule/reflow provided
        if not any([trainer, datamodule, reflow]):
            return

        try:
            from .memory_cleanup import cleanup_trial_resources

            # Extract trainer and datamodule from reflow if provided
            if reflow:
                trainer = reflow.trainer if hasattr(reflow, 'trainer') else trainer
                datamodule = reflow.datamodule if hasattr(reflow, 'datamodule') else datamodule

            cleanup_trial_resources(trainer=trainer, datamodule=datamodule)
            logger.debug(f"Trial {trial_number} ({status}): cleanup_trial_resources completed")
        except Exception as cleanup_err:
            logger.warning(f"Trial {trial_number} ({status}): cleanup failed: {cleanup_err}")
    
    def create_objective(self) -> Callable[[optuna.Trial], float]:
        """
        Create the objective function using LightningReflow.
        
        Returns:
            Objective function that takes a trial and returns a metric value
        """
        def objective(trial: optuna.Trial) -> float:
            # Start with base config (defensive for None)
            config = (self.base_config or {}).copy()
            
            # Apply fixed config overrides first
            if self.config_overrides:
                config = apply_dotted_updates(config, self.config_overrides)

            # Then apply suggested hyperparameters from search space
            if callable(self.search_space) and not hasattr(self.search_space, 'suggest_params'):
                suggested_params = self.search_space(trial)
            else:
                suggested_params = self.search_space.suggest_params(trial)
            config = apply_dotted_updates(config, suggested_params)

            # CRITICAL FIX: Re-apply config_overrides to ensure they take precedence
            # over search space suggestions (important for test mode settings)
            if self.config_overrides:
                config = apply_dotted_updates(config, self.config_overrides)
            
            # Setup callbacks
            callbacks = list(self.callbacks)
            
            # Add pruning callback and NaN detection
            if not isinstance(self.pruner, NopPruner):
                # Import enhanced pruning callback (validation-end) without step-based reporting
                try:
                    from .nan_detection_callback import EnhancedOptunaPruningCallback
                    pruning_callback = EnhancedOptunaPruningCallback(
                        trial,
                        monitor=self.metric,
                        check_nan=True,
                        verbose=True,
                    )
                except ImportError:
                    pruning_callback = OptunaPruningCallback(trial, monitor=self.metric)
                callbacks.append(pruning_callback)
                # Always add train-step NaN detection alongside pruner
                try:
                    from .nan_detection_callback import NaNDetectionCallback
                    nan_callback = NaNDetectionCallback(
                        trial,
                        monitor=self.metric,
                        check_train_loss=True,
                        check_every_n_steps=10,
                        verbose=True
                    )
                    callbacks.append(nan_callback)
                except ImportError:
                    pass
            else:
                # NopPruner: still add NaN detection to kill NaN trials quickly
                try:
                    from .nan_detection_callback import NaNDetectionCallback
                    nan_callback = NaNDetectionCallback(
                        trial,
                        monitor=self.metric,
                        check_train_loss=True,
                        check_every_n_steps=10,
                        verbose=True
                    )
                    callbacks.append(nan_callback)
                except ImportError:
                    pass  # NaN detection not available
            
            # Configure trainer settings (must come BEFORE any use of trainer_config)
            trainer_config = config.get('trainer', {})

            # Extract and preserve callbacks from config (except problematic ones)
            config_callbacks = trainer_config.pop('callbacks', None)
            if config_callbacks:
                # Helper to instantiate from class_path + init_args config
                def instantiate_callback(cb_config):
                    import importlib
                    class_path = cb_config['class_path']
                    module_path, class_name = class_path.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    cls = getattr(module, class_name)
                    init_args = cb_config.get('init_args', {})
                    return cls(**init_args)

                for cb_config in config_callbacks:
                    if isinstance(cb_config, dict) and 'class_path' in cb_config:
                        try:
                            cb = instantiate_callback(cb_config)
                            # Filter out callbacks that conflict with HPO
                            # Keep: Visualizer, custom callbacks
                            # Skip: ModelCheckpoint (we add our own), ProgressBar, etc.
                            from lightning.pytorch.callbacks import ModelCheckpoint, ProgressBar, RichProgressBar, TQDMProgressBar
                            if not isinstance(cb, (ModelCheckpoint, ProgressBar, RichProgressBar, TQDMProgressBar)):
                                callbacks.append(cb)
                        except Exception as e:
                            # Log but don't fail if callback instantiation fails
                            import logging
                            logging.getLogger(__name__).warning(f"Failed to instantiate callback from config: {e}")
                    elif not isinstance(cb_config, dict):
                        # Already instantiated callback
                        callbacks.append(cb_config)

            trainer_config.pop('logger', None)  # Will be set by Reflow or below

            # Do not mutate max_steps here; prefer config-driven control

            # Prefer automatic device selection unless explicitly set
            if 'accelerator' not in trainer_config:
                trainer_config['accelerator'] = 'auto'
            if 'devices' not in trainer_config:
                trainer_config['devices'] = 'auto'

            # Keep Lightning's default progress bar enabled for visual feedback during training
            # (We only disable PauseCallback to prevent validation-boundary pauses)

            # Add checkpoint callback if requested and ensure checkpointing is enabled
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
                # Ensure Trainer is allowed to use checkpoint callbacks
                try:
                    # Respect explicit True and override only when explicitly disabled
                    if trainer_config.get('enable_checkpointing') is False:
                        trainer_config['enable_checkpointing'] = True
                except Exception:
                    # If trainer_config is not a plain dict-like, set defensively
                    trainer_config['enable_checkpointing'] = True

            # Setup WandB logger if requested
            # Add prune-on-exception to free resources on early failures
            try:
                from .callbacks import PruneOnExceptionCallback
                callbacks.append(PruneOnExceptionCallback(trial))
            except Exception:
                pass
            
            # Setup WandB logger if requested (centralized utility)
            wandb_logger = None
            if self.wandb_project:
                from ..utils.wandb_logger import create_wandb_logger
                wandb_logger = create_wandb_logger(
                    project=self.wandb_project,
                    study_name=self.study_name,
                    trial_number=trial.number,
                    suggested_params=suggested_params,
                    sampler_name=self.sampler.__class__.__name__,
                    pruner_name=self.pruner.__class__.__name__,
                    upload_checkpoints=self.upload_checkpoints,
                )
                trainer_config['logger'] = wandb_logger
            
            if self.use_reflow:
                # Use LightningReflow for proper environment setup and compilation
                try:
                    # Extract model and data configs using helper
                    model_args, data_args = self._extract_config_args(config)

                    # Create LightningReflow instance
                    # Note: logger is passed through trainer_defaults
                    reflow = LightningReflow(
                        model_class=self.model_class,
                        datamodule_class=self.datamodule_class,
                        model_init_args=model_args,
                        datamodule_init_args=data_args,
                        trainer_defaults=trainer_config,  # Includes logger
                        callbacks=callbacks,
                        seed_everything=config.get('seed_everything', None),
                        # Don't pass full config as overrides - it has non-primitive values
                        # Only pass the environment-related configs
                        config_overrides={
                            'environment': config.get('environment', {}),
                            'compile': config.get('compile', {})
                        },
                        auto_configure_logging=False,  # We handle logging ourselves
                        disable_pause_callback=True  # HPO manages pause at trial boundaries
                    )

                    # CRITICAL: Remove PauseCallback from callbacks if it was added
                    # Even with disable_pause_callback=True, the callback might still be added
                    # We keep FlowProgressBarCallback for visual feedback, only remove PauseCallback
                    try:
                        from lightning_reflow.callbacks.pause import PauseCallback
                        if hasattr(reflow, 'callbacks') and reflow.callbacks:
                            original_count = len(reflow.callbacks)
                            reflow.callbacks = [
                                cb for cb in reflow.callbacks
                                if not isinstance(cb, PauseCallback)
                            ]
                            if len(reflow.callbacks) < original_count:
                                logger.info(f"🚫 Removed {original_count - len(reflow.callbacks)} PauseCallback(s) for HPO")
                    except ImportError:
                        pass  # Callbacks not available

                    # Run training
                    result = reflow.fit()

                    # Extract metric value using helper
                    metric_value = self._extract_metric_value(trainer=None, reflow=reflow)

                    # IMPORTANT: Let callbacks finish logging before closing WandB
                    if wandb_logger:
                        try:
                            import wandb
                            if wandb.run is not None:
                                wandb.run.log_code()
                                wandb.run.summary.update({"final_metric": metric_value})
                        except Exception:
                            pass

                    # Clean up torch compile state between trials
                    self._reset_torch_compile_state()

                    # Finalize WandB logger using helper
                    self._finalize_wandb(wandb_logger, "success")

                    # NOTE: DataLoader cleanup is now handled automatically by:
                    # 1. LightningReflow.fit() finally block (always runs)
                    # 2. MemoryCleanupCallback on_fit_end hook (if enabled)
                    # This manual cleanup call is kept for backward compatibility
                    # but will be removed in future versions.
                    self._cleanup_dataloader_workers(
                        trial_number=trial.number,
                        reflow=reflow,
                        status="completed"
                    )

                    return metric_value
                    
                except optuna.TrialPruned:
                    # Clean up torch compile state
                    self._reset_torch_compile_state()
                    # Clean up WandB using helper
                    self._finalize_wandb(wandb_logger, "pruned")
                    # Clean up DataLoader workers using helper
                    self._cleanup_dataloader_workers(
                        trial_number=trial.number,
                        reflow=reflow,
                        status="pruned"
                    )
                    raise
                except Exception as e:
                    logger.error(f"Trial {trial.number} failed with Reflow: {e}")
                    # Clean up torch compile state
                    self._reset_torch_compile_state()
                    # Clean up WandB using helper
                    self._finalize_wandb(wandb_logger, "failed")
                    # Clean up DataLoader workers using helper
                    self._cleanup_dataloader_workers(
                        trial_number=trial.number,
                        reflow=reflow,
                        status="failed"
                    )
                    # Optionally fall back to vanilla Lightning
                    if self.verbose:
                        logger.info("Falling back to vanilla Lightning")
                    # Logger should already be in trainer_config
                    return self._run_vanilla_lightning(config, callbacks, trainer_config, trial, wandb_logger)
            else:
                # Use vanilla Lightning (original implementation)
                # Logger should already be in trainer_config from above
                return self._run_vanilla_lightning(config, callbacks, trainer_config, trial, wandb_logger)
        
        return objective
    
    def _run_vanilla_lightning(self, config, callbacks, trainer_config, trial, wandb_logger=None):
        """Run training with vanilla Lightning (fallback or when use_reflow=False)."""
        # Extract model and data configs using helper
        model_args, data_args = self._extract_config_args(config)

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
        # If a ModelCheckpoint is present in callbacks, ensure checkpointing isn't disabled
        try:
            from lightning.pytorch.callbacks import ModelCheckpoint as _ModelCheckpoint
            if any(isinstance(cb, _ModelCheckpoint) for cb in callbacks):
                if trainer_config.get('enable_checkpointing') is False:
                    trainer_config['enable_checkpointing'] = True
        except Exception:
            pass

        # Keep Lightning's default progress bar for visual feedback

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

            # Extract metric value using helper
            metric_value = self._extract_metric_value(trainer=trainer, reflow=None)

            # Finalize WandB logger using helper
            self._finalize_wandb(wandb_logger, "success")

            # NOTE: DataLoader cleanup is now handled automatically by:
            # 1. LightningReflow.fit() finally block (for Reflow path)
            # 2. MemoryCleanupCallback on_fit_end hook (if enabled)
            # This manual cleanup call is kept for backward compatibility
            # but will be removed in future versions.
            self._cleanup_dataloader_workers(
                trial_number=trial.number,
                trainer=trainer,
                datamodule=datamodule,
                status="completed"
            )

            return metric_value

        except optuna.TrialPruned:
            # Clean up WandB using helper
            self._finalize_wandb(wandb_logger, "pruned")
            # Clean up DataLoader workers using helper
            self._cleanup_dataloader_workers(
                trial_number=trial.number,
                trainer=trainer,
                datamodule=datamodule,
                status="pruned"
            )
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # Clean up WandB using helper
            self._finalize_wandb(wandb_logger, "failed")
            # Clean up DataLoader workers using helper
            self._cleanup_dataloader_workers(
                trial_number=trial.number,
                trainer=trainer,
                datamodule=datamodule,
                status="failed"
            )
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