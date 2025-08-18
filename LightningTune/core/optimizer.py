"""
Unified config-driven optimizer for PyTorch Lightning with multiple strategy support.

This module provides a consolidated optimizer that supports various optimization
strategies (BOHB, Optuna, ASHA, Random Search, Grid Search, PBT, etc.) through 
a clean dependency injection pattern. It combines the best features from all 
previous optimizer versions into a single, maintainable implementation.

Key Features:
- Support for multiple optimization strategies via dependency injection
- Config-driven approach (YAML/dict/JSON support)
- Integration with LightningReflow for pause/resume capabilities
- Clean separation of concerns with strategy pattern
- Full Ray Tune integration
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import yaml
import json
import shutil
import pandas as pd

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.air.config import RunConfig, CheckpointConfig
from ray.air import session

try:
    from lightning_reflow import LightningReflow
    from lightning_reflow.callbacks import PauseCallback
except ImportError:
    raise ImportError(
        "LightningReflow is required for config-driven optimization. "
        "Install it with: pip install lightning-reflow"
    )

from .config import SearchSpace, ConfigManager
from .strategies import OptimizationStrategy, OptimizationConfig, StrategyFactory

logger = logging.getLogger(__name__)


class ConfigDrivenOptimizer:
    """
    Generic config-driven optimizer for Lightning pipelines.
    
    This optimizer uses the strategy pattern to support different optimization
    algorithms (BOHB, Optuna, Random Search, etc.) while maintaining the same
    interface and config-driven approach.
    
    Example:
        ```python
        # Use BOHB
        optimizer = ConfigDrivenOptimizer(
            base_config_path="config.yaml",
            search_space=search_space,
            strategy="bohb",
            strategy_config={"grace_period": 10}
        )
        
        # Use Optuna
        optimizer = ConfigDrivenOptimizer(
            base_config_path="config.yaml",
            search_space=search_space,
            strategy="optuna",
            strategy_config={"use_pruner": True}
        )
        
        # Use Random Search
        optimizer = ConfigDrivenOptimizer(
            base_config_path="config.yaml",
            search_space=search_space,
            strategy="random"
        )
        ```
    """
    
    def __init__(
        self,
        base_config_source: Union[str, Path, Dict[str, Any]],
        search_space: Union[SearchSpace, Dict[str, Any]],
        strategy: Union[str, OptimizationStrategy] = "bohb",
        optimization_config: Optional[OptimizationConfig] = None,
        strategy_config: Optional[Dict[str, Any]] = None,
        lightning_reflow_kwargs: Optional[Dict[str, Any]] = None,
        additional_callbacks: Optional[List] = None,
        base_config_path: Optional[Union[str, Path]] = None,  # For backward compatibility
    ):
        """
        Initialize the config-driven optimizer.
        
        Args:
            base_config_source: Can be:
                - Path/str to a YAML/JSON config file
                - Dict containing the configuration directly
            search_space: Search space definition (SearchSpace object or dict)
            strategy: Optimization strategy name or instance
            optimization_config: Common optimization configuration
            strategy_config: Strategy-specific configuration
            lightning_reflow_kwargs: Additional kwargs for LightningReflow
            additional_callbacks: Extra callbacks to add to training
            base_config_path: (Deprecated) Use base_config_source instead
        """
        # Handle backward compatibility
        if base_config_path is not None:
            import warnings
            warnings.warn(
                "base_config_path is deprecated, use base_config_source instead",
                DeprecationWarning,
                stacklevel=2
            )
            base_config_source = base_config_path
        
        self.base_config_source = base_config_source
        
        # Store the path if it's a path, otherwise None
        if isinstance(base_config_source, dict):
            self.base_config_path = None
            self.base_config_dict = base_config_source
        else:
            self.base_config_path = Path(base_config_source)
            self.base_config_dict = None
            if not self.base_config_path.exists():
                raise FileNotFoundError(f"Base config not found: {base_config_source}")
        
        # Handle search space
        if isinstance(search_space, SearchSpace):
            self.search_space = search_space
            self.search_space_dict = search_space.get_search_space()
            metric_config = search_space.get_metric_config()
            default_metric = metric_config.get("metric", "val_loss")
            default_mode = metric_config.get("mode", "min")
        else:
            self.search_space = None
            self.search_space_dict = search_space
            default_metric = "val_loss"
            default_mode = "min"
        
        # Create optimization config
        if optimization_config is None:
            optimization_config = OptimizationConfig(
                metric=default_metric,
                mode=default_mode
            )
        elif self.search_space:
            # Override metric config from search space if provided
            metric_config = self.search_space.get_metric_config()
            optimization_config.metric = metric_config.get("metric", optimization_config.metric)
            optimization_config.mode = metric_config.get("mode", optimization_config.mode)
        
        self.optimization_config = optimization_config
        
        # Create or use strategy
        if isinstance(strategy, str):
            self.strategy = StrategyFactory.create(
                strategy,
                self.optimization_config,
                **(strategy_config or {})
            )
        elif isinstance(strategy, OptimizationStrategy):
            self.strategy = strategy
        else:
            raise ValueError(f"Invalid strategy type: {type(strategy)}")
        
        # Validate search space for the strategy
        if not self.strategy.validate_search_space(self.search_space_dict):
            logger.warning(f"Search space may not be optimal for {self.strategy.get_strategy_name()}")
        
        self.lightning_reflow_kwargs = lightning_reflow_kwargs or {}
        self.additional_callbacks = additional_callbacks or []
        
        # Config manager for merging
        self.config_manager = ConfigManager(base_config_source)
        
        # Setup directories
        self._setup_directories()
        
        # Results storage
        self.results = None
        
        logger.info(f"Initialized ConfigDrivenOptimizer with {self.strategy.get_strategy_name()} strategy")
    
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        # Ensure experiment_dir is a Path object
        if not isinstance(self.optimization_config.experiment_dir, Path):
            self.optimization_config.experiment_dir = Path(self.optimization_config.experiment_dir)
        self.optimization_config.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        if self.optimization_config.verbose > 0:
            log_dir = self.optimization_config.experiment_dir / "logs"
            log_dir.mkdir(exist_ok=True)
            
            # Add file handler
            file_handler = logging.FileHandler(
                log_dir / f"{self.optimization_config.experiment_name}.log"
            )
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(file_handler)
    
    def _create_trainable(self) -> callable:
        """
        Create a trainable function for Ray Tune.
        
        Returns:
            Trainable function that takes config and returns metrics
        """
        def trainable_fn(config: Dict[str, Any]) -> Dict[str, Any]:
            """Trainable function for a single trial."""
            
            # Ensure LightningTune can be imported in Ray workers
            import sys
            import os
            module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if module_path not in sys.path:
                sys.path.insert(0, module_path)
            
            # Also add tests directory if we're in a test environment
            tests_path = os.path.join(module_path, "tests")
            if os.path.exists(tests_path) and tests_path not in sys.path:
                sys.path.insert(0, tests_path)
            
            # Get trial directory
            trial_dir = Path(session.get_trial_dir())
            
            # Validate config if using SearchSpace
            if self.search_space and hasattr(self.search_space, 'validate_config'):
                if not self.search_space.validate_config(config):
                    logger.warning(f"Invalid config: {config}")
                    return {
                        self.optimization_config.metric: float('inf') 
                        if self.optimization_config.mode == 'min' else float('-inf')
                    }
            
            # Transform config if using SearchSpace
            if self.search_space and hasattr(self.search_space, 'transform_config'):
                config = self.search_space.transform_config(config)
            
            # Create trial-specific config by merging
            merged_config = self.config_manager.merge_configs(
                self.config_manager.base_config,
                config
            )
            
            # Save merged config
            trial_config_path = trial_dir / "config.yaml"
            with open(trial_config_path, 'w') as f:
                yaml.dump(merged_config, f, default_flow_style=False)
            
            # Log trial info
            logger.info(f"Trial {trial_dir.name} using {self.strategy.get_strategy_name()}")
            logger.debug(f"Config overrides: {config}")
            
            # Create callbacks
            from ..callbacks.report import BOHBReportCallback
            
            callbacks = self.additional_callbacks.copy()
            
            # Add reporting callback
            callbacks.append(BOHBReportCallback(
                metrics=None,  # Report all metrics
                primary_metric=self.optimization_config.metric,
                report_on_epoch=True,
                checkpoint_on_epoch=True,
                verbose=self.optimization_config.verbose > 1
            ))
            
            # Add pause callback for checkpointing
            callbacks.append(PauseCallback(
                checkpoint_dir=str(trial_dir / "checkpoints"),
                enable_pause=False,
                show_pause_countdown=False
            ))
            
            # Initialize LightningReflow
            reflow = LightningReflow(
                config_files=str(trial_config_path),
                callbacks=callbacks,
                trainer_defaults={
                    "default_root_dir": str(trial_dir),
                    "enable_progress_bar": False,
                    "enable_checkpointing": True,
                },
                auto_configure_logging=False,
                **self.lightning_reflow_kwargs
            )
            
            # Check for checkpoint to resume
            checkpoint = session.get_checkpoint()
            
            try:
                if checkpoint and "checkpoint_path" in checkpoint:
                    logger.info(f"Resuming from checkpoint: {checkpoint['checkpoint_path']}")
                    result = reflow.resume(checkpoint["checkpoint_path"])
                else:
                    result = reflow.fit()
                
                # Extract metrics
                if isinstance(result, dict):
                    return result
                else:
                    # Try to get metrics from trainer
                    if hasattr(reflow, 'trainer') and reflow.trainer:
                        metrics = {}
                        for key, value in reflow.trainer.callback_metrics.items():
                            if hasattr(value, 'item'):
                                metrics[key] = float(value.item())
                            else:
                                metrics[key] = float(value)
                        return metrics
                    return {}
                    
            except Exception as e:
                logger.error(f"Trial failed with error: {e}")
                return {
                    self.optimization_config.metric: float('inf') 
                    if self.optimization_config.mode == 'min' else float('-inf')
                }
            
            finally:
                # Cleanup
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return trainable_fn
    
    def create_reporter(self) -> CLIReporter:
        """Create CLI reporter for console output."""
        # Determine metrics to display
        metric_columns = [self.optimization_config.metric]
        
        # Add common metrics
        additional_metrics = ["val_loss", "train_loss", "epoch", "training_iteration"]
        for metric in additional_metrics:
            if metric not in metric_columns:
                metric_columns.append(metric)
        
        # Parameters to display (top 5)
        param_columns = list(self.search_space_dict.keys())[:5]
        
        return CLIReporter(
            metric_columns=metric_columns,
            parameter_columns=param_columns,
            max_progress_rows=20,
            max_report_frequency=30,
        )
    
    def run(
        self,
        resume: bool = False,
        time_budget_hrs: Optional[float] = None,
    ) -> tune.ResultGrid:
        """
        Run optimization using the configured strategy.
        
        Args:
            resume: Whether to resume from previous experiment
            time_budget_hrs: Optional time budget override
            
        Returns:
            Ray Tune ResultGrid with optimization results
        """
        # Initialize Ray if needed
        if not ray.is_initialized():
            total_cpus = int(
                self.optimization_config.resources_per_trial.get("cpu", 1) * 
                self.optimization_config.max_concurrent_trials
            )
            total_gpus = int(
                self.optimization_config.resources_per_trial.get("gpu", 0) * 
                self.optimization_config.max_concurrent_trials
            )
            
            init_kwargs = {"num_cpus": total_cpus}
            if total_gpus > 0:
                init_kwargs["num_gpus"] = total_gpus
                
            ray.init(**init_kwargs)
            logger.info(f"Initialized Ray with {total_cpus} CPUs and {total_gpus} GPUs")
        
        # Get strategy components
        search_algorithm = self.strategy.get_search_algorithm()
        scheduler = self.strategy.get_scheduler()
        num_samples = self.strategy.get_num_samples()
        
        # Configure run
        run_config = RunConfig(
            name=self.optimization_config.experiment_name,
            storage_path=str(self.optimization_config.experiment_dir),
            checkpoint_config=CheckpointConfig(
                # Function trainables handle checkpointing internally
                num_to_keep=2,
            ),
            verbose=self.optimization_config.verbose,
        )
        
        # Create reporter
        reporter = self.create_reporter() if self.optimization_config.verbose > 0 else None
        
        # Time budget
        time_budget = time_budget_hrs or self.optimization_config.time_budget_hrs
        time_budget_s = time_budget * 3600 if time_budget else None
        
        # Log optimization details
        logger.info("="*60)
        logger.info(f"Starting optimization with {self.strategy.get_strategy_name()}")
        if self.base_config_path:
            logger.info(f"Base config: {self.base_config_path}")
        else:
            logger.info(f"Base config: <dict with {len(self.base_config_dict)} keys>")
        logger.info(f"Search space: {list(self.search_space_dict.keys())}")
        logger.info(f"Optimizing {self.optimization_config.metric} ({self.optimization_config.mode})")
        logger.info(f"Max concurrent trials: {self.optimization_config.max_concurrent_trials}")
        logger.info("="*60)
        
        # Create tuner with strategy-specific config
        tune_config_kwargs = {
            "search_alg": search_algorithm,
            "scheduler": scheduler,
            "num_samples": num_samples,
            "max_concurrent_trials": self.optimization_config.max_concurrent_trials,
            "time_budget_s": time_budget_s,
            "reuse_actors": False,
        }
        
        # Add strategy-specific kwargs
        tune_config_kwargs.update(self.strategy.get_tune_config_kwargs())
        
        tuner = tune.Tuner(
            trainable=self._create_trainable(),
            param_space=self.search_space_dict,
            tune_config=tune.TuneConfig(**tune_config_kwargs),
            run_config=run_config,
        )
        
        # Resume or start fresh
        if resume:
            resume_path = self.optimization_config.experiment_dir / self.optimization_config.experiment_name
            logger.info(f"Resuming from: {resume_path}")
            tuner = tuner.restore(
                str(resume_path),
                trainable=self._create_trainable(),
            )
        
        # Run optimization
        self.results = tuner.fit()
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _save_results(self) -> None:
        """Save optimization results."""
        if not self.results:
            return
        
        results_dir = (
            self.optimization_config.experiment_dir / 
            self.optimization_config.experiment_name / 
            "results"
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Get best result
        best_result = self.results.get_best_result(
            metric=self.optimization_config.metric,
            mode=self.optimization_config.mode
        )
        
        # Save best configuration overrides
        best_overrides_path = results_dir / "best_overrides.yaml"
        with open(best_overrides_path, 'w') as f:
            yaml.dump(best_result.config, f, default_flow_style=False)
        
        # Create complete best config
        best_complete_config = self.config_manager.merge_configs(
            self.config_manager.base_config,
            best_result.config
        )
        best_config_path = results_dir / "best_complete_config.yaml"
        with open(best_config_path, 'w') as f:
            yaml.dump(best_complete_config, f, default_flow_style=False)
        
        # Save metrics
        best_metrics_path = results_dir / "best_metrics.json"
        with open(best_metrics_path, 'w') as f:
            json.dump(best_result.metrics, f, indent=2)
        
        # Save summary including strategy info
        summary = {
            "experiment_name": self.optimization_config.experiment_name,
            "strategy": self.strategy.get_strategy_name(),
            "base_config": str(self.base_config_path) if self.base_config_path else "<dict>",
            "best_metric_value": best_result.metrics.get(self.optimization_config.metric),
            "best_overrides": best_result.config,
            "total_trials": len(self.results),
        }
        
        summary_path = results_dir / "summary.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        # Save results dataframe
        try:
            df = self.results.get_dataframe()
            df.to_csv(results_dir / "all_trials.csv", index=False)
        except Exception as e:
            logger.warning(f"Could not save results dataframe: {e}")
        
        logger.info(f"Results saved to: {results_dir}")
        logger.info(f"Best {self.optimization_config.metric}: {best_result.metrics.get(self.optimization_config.metric):.4f}")
        logger.info(f"Strategy used: {self.strategy.get_strategy_name()}")
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get the best configuration overrides found."""
        if not self.results:
            raise ValueError("No results available. Run optimization first.")
        
        best_result = self.results.get_best_result(
            metric=self.optimization_config.metric,
            mode=self.optimization_config.mode
        )
        return best_result.config
    
    def get_best_complete_config(self) -> Dict[str, Any]:
        """Get the complete best configuration (base + overrides)."""
        best_overrides = self.get_best_config()
        return self.config_manager.merge_configs(
            self.config_manager.base_config,
            best_overrides
        )
    
    def create_production_config(
        self,
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Create production-ready configuration file from best trial.
        
        Args:
            output_path: Where to save the config
            
        Returns:
            Path to production configuration file
        """
        production_config = self.get_best_complete_config()
        
        # Determine output path
        if output_path is None:
            output_path = (
                self.optimization_config.experiment_dir / 
                self.optimization_config.experiment_name / 
                "results" / 
                "production_config.yaml"
            )
        else:
            output_path = Path(output_path)
        
        # Save configuration
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(production_config, f, default_flow_style=False)
        
        logger.info(f"Production configuration saved to: {output_path}")
        
        # Copy best checkpoint if available
        best_result = self.results.get_best_result(
            metric=self.optimization_config.metric,
            mode=self.optimization_config.mode
        )
        if best_result.checkpoint:
            checkpoint_src = Path(best_result.checkpoint.path)
            if checkpoint_src.exists():
                checkpoint_dst = output_path.parent / "best_checkpoint.ckpt"
                shutil.copy2(checkpoint_src, checkpoint_dst)
                logger.info(f"Best checkpoint copied to: {checkpoint_dst}")
        
        return output_path
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze optimization results.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.results:
            raise ValueError("No results available. Run optimization first.")
        
        # Get dataframe
        df = self.results.get_dataframe()
        
        # Best result
        best_result = self.results.get_best_result(
            metric=self.optimization_config.metric,
            mode=self.optimization_config.mode
        )
        
        # Analysis
        analysis = {
            "strategy": self.strategy.get_strategy_name(),
            "total_trials": len(df),
            "completed_trials": len(
                df[df["training_iteration"] == self.optimization_config.max_epochs]
            ) if "training_iteration" in df else 0,
            "best_metric_value": best_result.metrics.get(self.optimization_config.metric),
            "best_overrides": best_result.config,
        }
        
        # Metric statistics
        if self.optimization_config.metric in df:
            analysis["metric_statistics"] = {
                "mean": df[self.optimization_config.metric].mean(),
                "std": df[self.optimization_config.metric].std(),
                "min": df[self.optimization_config.metric].min(),
                "max": df[self.optimization_config.metric].max(),
                "median": df[self.optimization_config.metric].median(),
            }
        
        # Time statistics
        if "time_total_s" in df.columns:
            analysis["time_statistics"] = {
                "total_hours": df["time_total_s"].sum() / 3600,
                "mean_minutes": df["time_total_s"].mean() / 60,
                "max_minutes": df["time_total_s"].max() / 60,
            }
        
        # Parameter importance (correlation with metric)
        if self.optimization_config.metric in df:
            param_importance = {}
            for col in df.columns:
                if col.startswith("config/"):
                    try:
                        if df[col].dtype in ['float64', 'int64']:
                            corr = df[col].corr(df[self.optimization_config.metric])
                            if not pd.isna(corr):
                                param_name = col.replace("config/", "")
                                param_importance[param_name] = abs(corr)
                    except:
                        pass
            
            analysis["parameter_importance"] = dict(
                sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
            )
        
        return analysis