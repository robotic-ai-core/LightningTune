"""
Optuna-driven optimizer using LightningReflow for HPO trials.

This module provides a clean optimizer that uses LightningReflow for training execution,
ensuring HPO trials run with the same optimizations as standalone training.
"""

import json
import tempfile
import shutil
import atexit
import copy
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union, Type, List
import logging

import optuna
from optuna.samplers import BaseSampler, TPESampler
from optuna.pruners import BasePruner, MedianPruner, NopPruner
import torch
from lightning.pytorch.callbacks import Callback

# Import LightningReflow for training orchestration
import sys
lightning_reflow_path = Path(__file__).parent.parent.parent.parent / "LightningReflow"
if lightning_reflow_path.exists():
    sys.path.insert(0, str(lightning_reflow_path))
from lightning_reflow import LightningReflow

from .search_space import OptunaSearchSpace
from .callbacks import OptunaPruningCallback
from ..utils.config_utils import apply_dotted_updates, load_config

logger = logging.getLogger(__name__)


class TrialCheckpointManager:
    """
    Manages trial checkpoints, keeping only the top-k best trials.

    This helps reduce disk usage during HPO by automatically cleaning up
    checkpoints from trials that are no longer in the top-k.
    """

    def __init__(self, top_k: int = 0, direction: str = "minimize"):
        """
        Initialize the checkpoint manager.

        Args:
            top_k: Number of best trial checkpoints to keep. 0 disables checkpointing.
            direction: Optimization direction ("minimize" or "maximize")
        """
        self.top_k = top_k
        self.direction = direction
        self.trial_checkpoints: Dict[int, tuple] = {}  # {trial_number: (value, path)}

    @property
    def checkpoints_enabled(self) -> bool:
        """Whether checkpointing is enabled (top_k > 0)."""
        return self.top_k > 0

    def register_trial(self, trial_number: int, value: float, checkpoint_path: Path) -> None:
        """
        Register a completed trial's checkpoint.

        Args:
            trial_number: The trial number
            value: The objective value achieved
            checkpoint_path: Path to the trial's checkpoint directory
        """
        if not self.checkpoints_enabled:
            return

        self.trial_checkpoints[trial_number] = (value, checkpoint_path)
        self._cleanup_if_needed()

    def _cleanup_if_needed(self) -> None:
        """Remove checkpoints from worst trials if we exceed top_k."""
        if len(self.trial_checkpoints) <= self.top_k:
            return

        # Sort by value - best first
        reverse = (self.direction == "maximize")
        sorted_trials = sorted(
            self.trial_checkpoints.items(),
            key=lambda x: x[1][0],
            reverse=reverse
        )

        # Keep only top_k, delete the rest
        trials_to_delete = sorted_trials[self.top_k:]

        for trial_number, (value, path) in trials_to_delete:
            try:
                if path.exists():
                    shutil.rmtree(path)
                    logger.debug(f"Cleaned up checkpoint for trial {trial_number} (value={value:.6f})")
            except Exception as e:
                logger.warning(f"Could not clean up checkpoint for trial {trial_number}: {e}")
            del self.trial_checkpoints[trial_number]

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get the path to the best trial's checkpoint."""
        if not self.trial_checkpoints:
            return None

        reverse = (self.direction == "maximize")
        best_trial = max(
            self.trial_checkpoints.items(),
            key=lambda x: x[1][0] if reverse else -x[1][0]
        )
        return best_trial[1][1]  # Return the path


class OptunaDrivenOptimizer:
    """
    Optuna optimizer using LightningReflow for consistent training environment.

    This optimizer ensures that HPO trials run with the same optimizations as standalone
    training, including:
    - PyTorch compilation (if configured)
    - Environment variable setup (CUDA configs, etc.)
    - Proper callback management
    - Consistent configuration handling
    - Nested class_path config instantiation
    - Resource cleanup between trials
    """

    def __init__(
        self,
        base_config: Union[str, Path, Dict[str, Any]],
        search_space: Union[OptunaSearchSpace, Callable[[optuna.Trial], Dict[str, Any]]],
        model_class: Type,
        datamodule_class: Optional[Type] = None,
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
        checkpoint_top_k: int = 0,
        metric: str = "val_loss",
        verbose: bool = True,
        wandb_project: Optional[str] = None,
        upload_checkpoints: bool = False,
    ):
        """
        Initialize the optimizer.

        Args:
            base_config: Base configuration (path to YAML/JSON or dict)
            search_space: OptunaSearchSpace instance or callable function
            model_class: PyTorch Lightning module class
            datamodule_class: Optional PyTorch Lightning datamodule class
            sampler: Optuna sampler (defaults to TPESampler)
            pruner: Optuna pruner (defaults to MedianPruner)
            config_overrides: Fixed config overrides (applied before and after search space)
            study_name: Name for the Optuna study
            storage: Storage URL for Optuna
            direction: Optimization direction ("minimize" or "maximize")
            n_trials: Number of trials to run
            timeout: Time limit for optimization
            callbacks: Additional Lightning callbacks
            experiment_dir: Directory for saving experiments
            save_checkpoints: Whether to save model checkpoints (deprecated, use checkpoint_top_k)
            checkpoint_top_k: Number of best trial checkpoints to keep. 0 disables checkpointing (default).
            metric: Metric to optimize
            verbose: Whether to print progress
            wandb_project: Optional WandB project name
            upload_checkpoints: Whether to upload checkpoints to WandB
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
        self.metric = metric
        self.verbose = verbose
        self.wandb_project = wandb_project
        self.upload_checkpoints = upload_checkpoints

        # Handle checkpoint settings
        # checkpoint_top_k=0 means no checkpoints (new default)
        # For backward compatibility, if save_checkpoints=True and checkpoint_top_k=0,
        # we keep the old behavior (save all). Otherwise, use checkpoint_top_k.
        if checkpoint_top_k > 0:
            self.checkpoint_top_k = checkpoint_top_k
        elif save_checkpoints:
            # Backward compatibility: save_checkpoints=True means save all (no limit)
            self.checkpoint_top_k = float('inf')
        else:
            self.checkpoint_top_k = 0

        # Initialize checkpoint manager
        self.checkpoint_manager = TrialCheckpointManager(
            top_k=checkpoint_top_k,
            direction=direction
        )

        # Setup experiment directory
        self._temp_dir = None
        if experiment_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix=f"{study_name}_")
            self.experiment_dir = Path(self._temp_dir)
            if self.verbose:
                logger.info(f"Using temporary directory: {self.experiment_dir}")
            atexit.register(self._cleanup_temp_dir)
        else:
            self.experiment_dir = Path(experiment_dir)
            if self.verbose:
                logger.info(f"Using persistent directory: {self.experiment_dir}")

        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize study tracking
        self.study = None
        self.best_trial = None
        self.best_checkpoint = None

    def _cleanup_temp_dir(self):
        """Clean up temporary directory if it was created."""
        if self._temp_dir and Path(self._temp_dir).exists():
            try:
                shutil.rmtree(self._temp_dir)
                if self.verbose:
                    logger.info(f"Cleaned up temporary directory: {self._temp_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up temporary directory: {e}")

    def _load_config(self, config_source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration from file or dict."""
        return load_config(config_source)

    def _reset_torch_compile_state(self):
        """Reset torch compile state between trials to prevent interference."""
        try:
            import gc

            # Reset torch._dynamo state if available
            if hasattr(torch, '_dynamo'):
                if hasattr(torch._dynamo, 'reset'):
                    torch._dynamo.reset()

                if hasattr(torch._dynamo, 'config'):
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
                torch.cuda.ipc_collect()
                torch.cuda.manual_seed_all(torch.initial_seed())

            # Force garbage collection
            gc.collect()

            if self.verbose:
                logger.debug("Reset torch compile state between trials")

        except Exception as e:
            logger.debug(f"Could not reset torch compile state: {e}")

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

    def _extract_metric_value(self, reflow: LightningReflow) -> float:
        """
        Extract the target metric value from trainer's callback_metrics.

        Args:
            reflow: LightningReflow instance

        Returns:
            Metric value as float, or default value if metric not found
        """
        default_value = float('inf') if self.direction == "minimize" else float('-inf')

        if hasattr(reflow, 'trainer') and hasattr(reflow.trainer, 'callback_metrics'):
            if self.metric in reflow.trainer.callback_metrics:
                return reflow.trainer.callback_metrics[self.metric].item()

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

    def _cleanup_dataloader_workers(self, trial_number: int, reflow: LightningReflow = None, status: str = "completed"):
        """
        Clean up DataLoader workers to prevent thread accumulation.

        Note: This cleanup is primarily handled by LightningReflow.fit() finally block.
        This method provides additional cleanup for edge cases.

        Args:
            trial_number: Trial number for logging
            reflow: LightningReflow instance
            status: Trial status for logging
        """
        if not reflow:
            return

        try:
            from .memory_cleanup import cleanup_trial_resources

            trainer = reflow.trainer if hasattr(reflow, 'trainer') else None
            datamodule = reflow.datamodule if hasattr(reflow, 'datamodule') else None

            cleanup_trial_resources(trainer=trainer, datamodule=datamodule)
            logger.debug(f"Trial {trial_number} ({status}): cleanup completed")
        except Exception as cleanup_err:
            logger.warning(f"Trial {trial_number} ({status}): cleanup failed: {cleanup_err}")

    def _prepare_callbacks(self, trial: optuna.Trial, config: Dict[str, Any]) -> List[Callback]:
        """
        Prepare callbacks for a trial including pruning, NaN detection, and checkpointing.

        Args:
            trial: Optuna trial
            config: Trial configuration

        Returns:
            List of callbacks
        """
        callbacks = list(self.callbacks)

        # Add pruning callback and NaN detection
        if not isinstance(self.pruner, NopPruner):
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

            # Add train-step NaN detection
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
            # NopPruner: still add NaN detection
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

        # Extract and instantiate callbacks from config
        trainer_config = config.get('trainer', {})
        config_callbacks = trainer_config.get('callbacks', [])
        if config_callbacks:
            for cb_config in config_callbacks:
                if isinstance(cb_config, dict) and 'class_path' in cb_config:
                    try:
                        cb = self._instantiate_callback(cb_config)
                        # Filter out callbacks that conflict with HPO
                        from lightning.pytorch.callbacks import ModelCheckpoint, ProgressBar, RichProgressBar, TQDMProgressBar
                        if not isinstance(cb, (ModelCheckpoint, ProgressBar, RichProgressBar, TQDMProgressBar)):
                            callbacks.append(cb)
                    except Exception as e:
                        logger.warning(f"Failed to instantiate callback from config: {e}")
                elif not isinstance(cb_config, dict):
                    # Already instantiated callback
                    callbacks.append(cb_config)

        # Add checkpoint callback if checkpointing is enabled
        if self.checkpoint_manager.checkpoints_enabled:
            from lightning.pytorch.callbacks import ModelCheckpoint
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.experiment_dir / f"trial_{trial.number}",
                filename="{epoch}-{val_loss:.2f}",
                monitor=self.metric,
                mode="min" if self.direction == "minimize" else "max",
                save_top_k=1,
            )
            callbacks.append(checkpoint_callback)

        # Add prune-on-exception callback
        try:
            from .callbacks import PruneOnExceptionCallback
            callbacks.append(PruneOnExceptionCallback(trial))
        except Exception:
            pass

        return callbacks

    def _instantiate_callback(self, cb_config: Dict[str, Any]) -> Callback:
        """Instantiate a callback from class_path + init_args config."""
        import importlib
        class_path = cb_config['class_path']
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        init_args = cb_config.get('init_args', {})
        return cls(**init_args)

    def _prepare_trainer_config(self, config: Dict[str, Any], wandb_logger) -> Dict[str, Any]:
        """
        Prepare trainer configuration for a trial.

        Args:
            config: Trial configuration
            wandb_logger: WandB logger instance (or None)

        Returns:
            Trainer configuration dict
        """
        trainer_config = config.get('trainer', {}).copy()

        # Remove fields that will be set separately
        trainer_config.pop('callbacks', None)
        trainer_config.pop('logger', None)

        # Set logger if provided
        if wandb_logger:
            trainer_config['logger'] = wandb_logger

        # Prefer automatic device selection unless explicitly set
        if 'accelerator' not in trainer_config:
            trainer_config['accelerator'] = 'auto'
        if 'devices' not in trainer_config:
            trainer_config['devices'] = 'auto'

        # Ensure checkpointing is enabled if we have checkpoint callbacks
        if self.save_checkpoints and trainer_config.get('enable_checkpointing') is False:
            trainer_config['enable_checkpointing'] = True

        return trainer_config

    def create_objective(self) -> Callable[[optuna.Trial], float]:
        """
        Create the objective function using LightningReflow.

        Returns:
            Objective function that takes a trial and returns a metric value
        """
        def objective(trial: optuna.Trial) -> float:
            reflow = None
            wandb_logger = None

            try:
                # Prepare config with search space suggestions
                config = copy.deepcopy(self.base_config or {})

                # Apply fixed config overrides first
                if self.config_overrides:
                    config = apply_dotted_updates(config, self.config_overrides)

                # Apply suggested hyperparameters from search space
                if callable(self.search_space) and not hasattr(self.search_space, 'suggest_params'):
                    # Function signature: search_space(trial, config) -> config
                    config = self.search_space(trial, config)
                    suggested_params = trial.params
                else:
                    # Object-based search space
                    suggested_params = self.search_space.suggest_params(trial)
                    config = apply_dotted_updates(config, suggested_params)

                # Re-apply config_overrides to ensure they take precedence
                if self.config_overrides:
                    config = apply_dotted_updates(config, self.config_overrides)

                # Setup WandB logger if requested
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

                # Prepare callbacks and trainer config
                callbacks = self._prepare_callbacks(trial, config)
                trainer_config = self._prepare_trainer_config(config, wandb_logger)

                # Extract model and data configs
                model_args, data_args = self._extract_config_args(config)

                # Create LightningReflow instance
                reflow = LightningReflow(
                    model_class=self.model_class,
                    datamodule_class=self.datamodule_class,
                    model_init_args=model_args,
                    datamodule_init_args=data_args,
                    trainer_defaults=trainer_config,
                    callbacks=callbacks,
                    seed_everything=config.get('seed_everything', None),
                    config_overrides={
                        'environment': config.get('environment', {}),
                        'compile': config.get('compile', {})
                    },
                    auto_configure_logging=False,
                    disable_pause_callback=True  # HPO manages pause at trial boundaries
                )

                # Remove PauseCallback if it was added despite disable_pause_callback
                try:
                    from lightning_reflow.callbacks.pause import PauseCallback
                    if hasattr(reflow, 'callbacks') and reflow.callbacks:
                        reflow.callbacks = [
                            cb for cb in reflow.callbacks
                            if not isinstance(cb, PauseCallback)
                        ]
                except ImportError:
                    pass

                # Run training
                reflow.fit()

                # Extract metric value
                metric_value = self._extract_metric_value(reflow)

                # Register checkpoint with manager (for top-k cleanup)
                if self.checkpoint_manager.checkpoints_enabled:
                    checkpoint_path = self.experiment_dir / f"trial_{trial.number}"
                    self.checkpoint_manager.register_trial(
                        trial_number=trial.number,
                        value=metric_value,
                        checkpoint_path=checkpoint_path
                    )

                # Log final metric to WandB
                if wandb_logger:
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.run.summary.update({"final_metric": metric_value})
                    except Exception:
                        pass

                # Cleanup
                self._reset_torch_compile_state()
                self._finalize_wandb(wandb_logger, "success")
                self._cleanup_dataloader_workers(trial.number, reflow, "completed")

                return metric_value

            except optuna.TrialPruned:
                self._reset_torch_compile_state()
                self._finalize_wandb(wandb_logger, "pruned")
                self._cleanup_dataloader_workers(trial.number, reflow, "pruned")
                raise

            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                self._reset_torch_compile_state()
                self._finalize_wandb(wandb_logger, "failed")
                self._cleanup_dataloader_workers(trial.number, reflow, "failed")
                # Return worst possible value instead of failing the entire study
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
        if self.verbose:
            print(f"\nRunning {self.n_trials} trials with LightningReflow...")

        for i in range(self.n_trials):
            if self.verbose:
                print(f"\nTrial {i+1}/{self.n_trials}")

            self.study.optimize(
                objective,
                n_trials=1,
                timeout=self.timeout if i == self.n_trials - 1 else None,
                show_progress_bar=False
            )

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


# Backward compatibility alias
ReflowOptunaDrivenOptimizer = OptunaDrivenOptimizer
