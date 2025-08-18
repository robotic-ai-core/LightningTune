"""
Improved strategy pattern with cleaner dependency injection.

This module provides optimization strategies that can be instantiated
with their configuration and injected directly into the optimizer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging

# Optional imports with fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

import ray
from ray import tune
from ray.tune.schedulers import (
    FIFOScheduler,
    ASHAScheduler, 
    HyperBandScheduler,
    HyperBandForBOHB,
    PopulationBasedTraining,
    MedianStoppingRule,
)
from ray.tune.search import (
    BasicVariantGenerator,
    SearchAlgorithm,
    Repeater,
)
from ray.tune.search.bohb import TuneBOHB

try:
    from ray.tune.search.optuna import OptunaSearch
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    OptunaSearch = None

try:
    from ray.tune.search.hyperopt import HyperOptSearch
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    HyperOptSearch = None

try:
    from ray.tune.search.ax import AxSearch
    AX_AVAILABLE = True
except ImportError:
    AX_AVAILABLE = False
    AxSearch = None

try:
    from ray.tune.search.bayesopt import BayesOptSearch
    BAYESOPT_AVAILABLE = True
except ImportError:
    BAYESOPT_AVAILABLE = False
    BayesOptSearch = None

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Common configuration for all optimization strategies."""
    
    # Optimization settings
    max_epochs: int = 100
    num_samples: int = -1  # -1 means algorithm decides
    max_concurrent_trials: int = 4
    
    # Resources
    resources_per_trial: Dict[str, Any] = field(default_factory=lambda: {
        "cpu": 4,
        "gpu": 1.0
    })
    
    # Metric
    metric: str = "val_loss"
    mode: str = "min"
    
    # Experiment
    experiment_name: str = "optimization"
    experiment_dir: Path = field(default_factory=lambda: Path("./experiments"))
    
    # Logging
    verbose: int = 1
    seed: int = 42
    
    # Time budget
    time_budget_hrs: Optional[float] = None


class OptimizationStrategy(ABC):
    """
    Abstract base class for optimization strategies.
    
    Strategies are self-contained and configured at initialization,
    making them perfect for dependency injection.
    """
    
    @abstractmethod
    def get_search_algorithm(self) -> Optional[SearchAlgorithm]:
        """Return the search algorithm for this strategy."""
        pass
    
    @abstractmethod
    def get_scheduler(self) -> Optional[Any]:
        """Return the scheduler for this strategy."""
        pass
    
    @abstractmethod
    def get_num_samples(self) -> int:
        """Return the number of samples for this strategy."""
        pass
    
    def get_tune_config_kwargs(self) -> Dict[str, Any]:
        """Return additional kwargs for TuneConfig."""
        return {}
    
    def get_strategy_name(self) -> str:
        """Return the name of this strategy."""
        return self.__class__.__name__.replace("Strategy", "")
    
    def validate_search_space(self, search_space: Dict[str, Any]) -> bool:
        """Validate that the search space is compatible with this strategy."""
        return True
    
    def set_optimization_config(self, config: OptimizationConfig) -> None:
        """Set the optimization config (called by optimizer if needed)."""
        pass
    
    def describe(self) -> str:
        """Return a human-readable description of strategy configuration."""
        return f"{self.get_strategy_name()} strategy"
    
    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of strategy configuration for logging."""
        return {"name": self.get_strategy_name()}
    
    def analyze(self, df) -> Dict[str, Any]:
        """Perform strategy-specific analysis on results dataframe."""
        if not PANDAS_AVAILABLE or df is None:
            return {}
        return {}


class BOHBStrategy(OptimizationStrategy):
    """
    BOHB (Bayesian Optimization and HyperBand) strategy.
    
    Best for: Expensive evaluations with smooth parameter landscapes.
    Combines Bayesian optimization with aggressive early stopping.
    
    Example:
        ```python
        strategy = BOHBStrategy(
            max_t=100,
            reduction_factor=3,
            metric="val_loss",
            mode="min",
        )
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source="config.yaml",
            search_space=search_space,
            strategy=strategy,  # Inject directly
        )
        ```
    """
    
    def __init__(
        self,
        max_t: int = 100,
        reduction_factor: int = 3,
        metric: str = "val_loss",
        mode: str = "min",
        seed: int = 42,
    ):
        """
        Initialize BOHB strategy.
        
        Note: HyperBandForBOHB does not support grace_period.
        Use ASHAStrategy if you need explicit grace_period control.
        
        Args:
            max_t: Maximum epochs per trial
            reduction_factor: Factor by which to reduce resources
            metric: Metric to optimize
            mode: "min" or "max"
            seed: Random seed for reproducibility
        """
        self.max_t = max_t
        self.reduction_factor = reduction_factor
        self.metric = metric
        self.mode = mode
        self.seed = seed
    
    def get_search_algorithm(self):
        return TuneBOHB(
            metric=self.metric,
            mode=self.mode,
            seed=self.seed,
        )
    
    def get_scheduler(self):
        return HyperBandForBOHB(
            time_attr="training_iteration",
            metric=self.metric,
            mode=self.mode,
            max_t=self.max_t,
            reduction_factor=self.reduction_factor,
        )
    
    def get_num_samples(self) -> int:
        # BOHB determines samples automatically
        return -1
    
    def describe(self) -> str:
        return (
            f"BOHB(max_t={self.max_t}, "
            f"reduction={self.reduction_factor})"
        )
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "name": "BOHB",
            "reduction_factor": self.reduction_factor,
            "max_t": self.max_t,
        }
    
    def set_optimization_config(self, config: OptimizationConfig) -> None:
        """Update metric/mode from optimization config if not set."""
        if hasattr(config, 'metric'):
            self.metric = config.metric
        if hasattr(config, 'mode'):
            self.mode = config.mode
        if hasattr(config, 'max_epochs'):
            self.max_t = config.max_epochs


class OptunaStrategy(OptimizationStrategy):
    """
    Optuna strategy with Tree-structured Parzen Estimator (TPE).
    
    Best for: Good balance of exploration/exploitation, handles 
    categorical parameters well, supports pruning.
    
    Example:
        ```python
        strategy = OptunaStrategy(
            n_startup_trials=20,
            use_pruner=True,
            pruner_type="median",
            num_samples=100,
        )
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source="config.yaml",
            search_space=search_space,
            strategy=strategy,
        )
        ```
    """
    
    def __init__(
        self,
        n_startup_trials: int = 10,
        use_pruner: bool = True,
        pruner_type: str = "median",  # "median", "percentile", "hyperband"
        num_samples: int = 100,
        metric: str = "val_loss",
        mode: str = "min",
        seed: int = 42,
    ):
        self.n_startup_trials = n_startup_trials
        self.use_pruner = use_pruner
        self.pruner_type = pruner_type
        self.num_samples = num_samples
        self.metric = metric
        self.mode = mode
        self.seed = seed
    
    def get_search_algorithm(self) -> SearchAlgorithm:
        # Import optuna here to make it optional
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna not installed. Install with: pip install optuna")
        
        # Create Optuna sampler
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.n_startup_trials,
            seed=self.seed,
        )
        
        return OptunaSearch(
            sampler=sampler,
            metric=self.metric,
            mode=self.mode,
        )
    
    def get_scheduler(self):
        if not self.use_pruner:
            return None
        
        if self.pruner_type == "median":
            return MedianStoppingRule(
                time_attr="training_iteration",
                metric=self.metric,
                mode=self.mode,
                grace_period=5,
            )
        elif self.pruner_type == "hyperband":
            return ASHAScheduler(
                time_attr="training_iteration",
                metric=self.metric,
                mode=self.mode,
                max_t=100,  # Will be updated from config
                grace_period=10,
                reduction_factor=3,
            )
        else:
            return None
    
    def get_num_samples(self) -> int:
        return self.num_samples
    
    def describe(self) -> str:
        pruner_str = f", pruner={self.pruner_type}" if self.use_pruner else ""
        return f"Optuna(n_startup={self.n_startup_trials}{pruner_str})"
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "name": "Optuna",
            "n_startup_trials": self.n_startup_trials,
            "use_pruner": self.use_pruner,
            "pruner_type": self.pruner_type if self.use_pruner else None,
            "num_samples": self.num_samples,
        }
    
    def set_optimization_config(self, config: OptimizationConfig) -> None:
        """Update from optimization config."""
        if hasattr(config, 'metric'):
            self.metric = config.metric
        if hasattr(config, 'mode'):
            self.mode = config.mode


class RandomSearchStrategy(OptimizationStrategy):
    """
    Random search strategy.
    
    Best for: Initial exploration, baseline comparison, 
    when you have lots of parallel resources.
    
    Example:
        ```python
        strategy = RandomSearchStrategy(
            num_samples=50,
            use_early_stopping=True,
        )
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source="config.yaml",
            search_space=search_space,
            strategy=strategy,
        )
        ```
    """
    
    def __init__(
        self,
        num_samples: int = 50,
        use_early_stopping: bool = False,
        grace_period: int = 10,
        reduction_factor: int = 2,
        metric: str = "val_loss",
        mode: str = "min",
    ):
        self.num_samples = num_samples
        self.use_early_stopping = use_early_stopping
        self.grace_period = grace_period
        self.reduction_factor = reduction_factor
        self.metric = metric
        self.mode = mode
    
    def get_search_algorithm(self) -> Optional[SearchAlgorithm]:
        # Random search uses None (default Ray Tune behavior)
        return None
    
    def get_scheduler(self):
        if self.use_early_stopping:
            return ASHAScheduler(
                time_attr="training_iteration",
                metric=self.metric,
                mode=self.mode,
                max_t=100,  # Will be updated from config
                grace_period=self.grace_period,
                reduction_factor=self.reduction_factor,
            )
        return None
    
    def get_num_samples(self) -> int:
        return self.num_samples
    
    def describe(self) -> str:
        early_stop = " with early stopping" if self.use_early_stopping else ""
        return f"RandomSearch(n={self.num_samples}{early_stop})"
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "name": "RandomSearch",
            "num_samples": self.num_samples,
            "use_early_stopping": self.use_early_stopping,
        }
    
    def set_optimization_config(self, config: OptimizationConfig) -> None:
        """Update from optimization config."""
        if hasattr(config, 'metric'):
            self.metric = config.metric
        if hasattr(config, 'mode'):
            self.mode = config.mode


class PBTStrategy(OptimizationStrategy):
    """
    Population Based Training strategy.
    
    Best for: Long training runs where hyperparameters should adapt
    during training (e.g., learning rate schedules).
    
    Example:
        ```python
        strategy = PBTStrategy(
            perturbation_interval=10,
            population_size=8,
            hyperparam_mutations={
                "learning_rate": tune.loguniform(1e-5, 1e-2),
                "weight_decay": tune.uniform(0.0, 0.1),
            }
        )
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source="config.yaml",
            search_space=search_space,
            strategy=strategy,
        )
        ```
    """
    
    def __init__(
        self,
        perturbation_interval: int = 10,
        population_size: int = 8,
        hyperparam_mutations: Optional[Dict[str, Any]] = None,
        metric: str = "val_loss",
        mode: str = "min",
    ):
        self.perturbation_interval = perturbation_interval
        self.population_size = population_size
        self.hyperparam_mutations = hyperparam_mutations or {}
        self.metric = metric
        self.mode = mode
    
    def get_search_algorithm(self) -> Optional[SearchAlgorithm]:
        # PBT doesn't use a separate search algorithm
        return None
    
    def get_scheduler(self):
        return PopulationBasedTraining(
            time_attr="training_iteration",
            metric=self.metric,
            mode=self.mode,
            perturbation_interval=self.perturbation_interval,
            hyperparam_mutations=self.hyperparam_mutations,
            quantile_fraction=0.25,
            resample_probability=0.25,
        )
    
    def get_num_samples(self) -> int:
        return self.population_size
    
    def describe(self) -> str:
        return (
            f"PBT(population={self.population_size}, "
            f"interval={self.perturbation_interval})"
        )
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "name": "PBT",
            "population_size": self.population_size,
            "perturbation_interval": self.perturbation_interval,
            "num_mutations": len(self.hyperparam_mutations),
        }
    
    def set_optimization_config(self, config: OptimizationConfig) -> None:
        """Update from optimization config."""
        if hasattr(config, 'metric'):
            self.metric = config.metric
        if hasattr(config, 'mode'):
            self.mode = config.mode


class GridSearchStrategy(OptimizationStrategy):
    """
    Grid search strategy for exhaustive parameter search.
    
    Best for: Small search spaces where you want to try all combinations.
    
    Example:
        ```python
        strategy = GridSearchStrategy()
        
        # Grid search requires discrete values in search space
        search_space = {
            "learning_rate": [0.001, 0.01, 0.1],
            "batch_size": [16, 32, 64],
        }
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source="config.yaml",
            search_space=search_space,
            strategy=strategy,
        )
        ```
    """
    
    def __init__(self, metric: str = "val_loss", mode: str = "min"):
        self.metric = metric
        self.mode = mode
    
    def get_search_algorithm(self) -> Optional[SearchAlgorithm]:
        # Grid search uses basic variant generator
        return BasicVariantGenerator()
    
    def get_scheduler(self):
        return None
    
    def get_num_samples(self) -> int:
        # Grid search will try all combinations
        return -1
    
    def validate_search_space(self, search_space: Dict[str, Any]) -> bool:
        """Validate that search space contains only discrete values."""
        for key, value in search_space.items():
            if hasattr(value, '__name__'):
                # It's a function (like tune.uniform), not valid for grid search
                logger.warning(
                    f"Grid search requires discrete values. "
                    f"Parameter '{key}' uses continuous distribution."
                )
                return False
        return True
    
    def describe(self) -> str:
        return "GridSearch(exhaustive)"
    
    def get_summary(self) -> Dict[str, Any]:
        return {"name": "GridSearch"}
    
    def set_optimization_config(self, config: OptimizationConfig) -> None:
        """Update from optimization config."""
        if hasattr(config, 'metric'):
            self.metric = config.metric
        if hasattr(config, 'mode'):
            self.mode = config.mode


class BayesianOptimizationStrategy(OptimizationStrategy):
    """
    Pure Bayesian Optimization using Gaussian Processes.
    
    Best for: Small search spaces with expensive evaluations,
    when you want uncertainty estimates.
    
    Example:
        ```python
        strategy = BayesianOptimizationStrategy(
            n_initial_points=10,
            acquisition_function="ucb",  # or "ei", "poi"
        )
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source="config.yaml",
            search_space=search_space,
            strategy=strategy,
        )
        ```
    """
    
    def __init__(
        self,
        n_initial_points: int = 10,
        acquisition_function: str = "ucb",
        num_samples: int = 50,
        metric: str = "val_loss",
        mode: str = "min",
        seed: int = 42,
    ):
        self.n_initial_points = n_initial_points
        self.acquisition_function = acquisition_function
        self.num_samples = num_samples
        self.metric = metric
        self.mode = mode
        self.seed = seed
    
    def get_search_algorithm(self) -> SearchAlgorithm:
        try:
            # BayesOpt requires sklearn
            import sklearn  # noqa
        except ImportError:
            raise ImportError(
                "BayesOpt requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )
        
        return BayesOptSearch(
            metric=self.metric,
            mode=self.mode,
            random_state=self.seed,
        )
    
    def get_scheduler(self):
        return None
    
    def get_num_samples(self) -> int:
        return self.num_samples
    
    def describe(self) -> str:
        return (
            f"BayesOpt(n_init={self.n_initial_points}, "
            f"acq={self.acquisition_function})"
        )
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "name": "BayesianOptimization",
            "n_initial_points": self.n_initial_points,
            "acquisition_function": self.acquisition_function,
            "num_samples": self.num_samples,
        }
    
    def set_optimization_config(self, config: OptimizationConfig) -> None:
        """Update from optimization config."""
        if hasattr(config, 'metric'):
            self.metric = config.metric
        if hasattr(config, 'mode'):
            self.mode = config.mode


class HyperOptStrategy(OptimizationStrategy):
    """
    HyperOpt strategy with Tree-structured Parzen Estimator.
    
    Alternative to Optuna with different TPE implementation.
    
    Example:
        ```python
        strategy = HyperOptStrategy(
            n_initial_points=20,
            num_samples=100,
        )
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source="config.yaml",
            search_space=search_space,
            strategy=strategy,
        )
        ```
    """
    
    def __init__(
        self,
        n_initial_points: int = 20,
        num_samples: int = 100,
        metric: str = "val_loss",
        mode: str = "min",
        seed: int = 42,
    ):
        self.n_initial_points = n_initial_points
        self.num_samples = num_samples
        self.metric = metric
        self.mode = mode
        self.seed = seed
    
    def get_search_algorithm(self) -> SearchAlgorithm:
        try:
            import hyperopt  # noqa
        except ImportError:
            raise ImportError(
                "HyperOpt not installed. Install with: pip install hyperopt"
            )
        
        return HyperOptSearch(
            metric=self.metric,
            mode=self.mode,
            n_initial_points=self.n_initial_points,
            random_state_seed=self.seed,
        )
    
    def get_scheduler(self):
        return None
    
    def get_num_samples(self) -> int:
        return self.num_samples
    
    def describe(self) -> str:
        return f"HyperOpt(n_init={self.n_initial_points})"
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "name": "HyperOpt",
            "n_initial_points": self.n_initial_points,
            "num_samples": self.num_samples,
        }
    
    def set_optimization_config(self, config: OptimizationConfig) -> None:
        """Update from optimization config."""
        if hasattr(config, 'metric'):
            self.metric = config.metric
        if hasattr(config, 'mode'):
            self.mode = config.mode


class ASHAStrategy(OptimizationStrategy):
    """
    ASHA (Asynchronous Successive Halving) strategy.
    
    Similar to BOHB but with native grace_period support and asynchronous evaluation.
    Best for: When you need explicit control over grace_period and want faster
    convergence than standard HyperBand.
    
    Example:
        ```python
        strategy = ASHAStrategy(
            grace_period=10,  # Native support for grace_period!
            max_t=100,
            reduction_factor=3,
        )
        ```
    """
    
    def __init__(
        self,
        grace_period: int = 10,
        max_t: int = 100,
        reduction_factor: float = 3,
        brackets: int = 1,
        metric: str = "val_loss",
        mode: str = "min",
    ):
        """
        Initialize ASHA strategy with native grace_period support.
        
        Args:
            grace_period: Minimum epochs before any pruning decision
            max_t: Maximum epochs per trial
            reduction_factor: Factor by which to reduce resources
            brackets: Number of brackets (different starting points)
            metric: Metric to optimize
            mode: "min" or "max"
        """
        self.grace_period = grace_period
        self.max_t = max_t
        self.reduction_factor = reduction_factor
        self.brackets = brackets
        self.metric = metric
        self.mode = mode
    
    def get_search_algorithm(self) -> Optional[SearchAlgorithm]:
        # ASHA works with any search algorithm or None for random search
        return None
    
    def get_scheduler(self):
        from ray.tune.schedulers import AsyncHyperBandScheduler
        return AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric=self.metric,
            mode=self.mode,
            max_t=self.max_t,
            grace_period=self.grace_period,
            reduction_factor=self.reduction_factor,
            brackets=self.brackets,
        )
    
    def get_num_samples(self) -> int:
        # Return -1 to let the optimizer decide
        return -1
    
    def get_strategy_name(self) -> str:
        return "ASHA"
    
    def describe(self) -> str:
        return (
            f"ASHA(grace={self.grace_period}, "
            f"max_t={self.max_t}, "
            f"reduction={self.reduction_factor}, "
            f"brackets={self.brackets})"
        )
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "name": "ASHA",
            "grace_period": self.grace_period,
            "max_t": self.max_t,
            "reduction_factor": self.reduction_factor,
            "brackets": self.brackets,
            "metric": self.metric,
            "mode": self.mode,
        }
    
    def validate_search_space(self, search_space: Dict[str, Any]) -> bool:
        # ASHA works with any search space
        return True


class StrategyFactory:
    """Factory for creating optimization strategies from string names."""
    
    @staticmethod
    def create(
        strategy_name: str,
        optimization_config: OptimizationConfig = None,
        **kwargs
    ) -> OptimizationStrategy:
        """Create a strategy by name.
        
        Args:
            strategy_name: Name of strategy ('bohb', 'random', 'optuna', etc.)
            optimization_config: Optional optimization config to apply
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            OptimizationStrategy instance
        """
        strategies = {
            'bohb': BOHBStrategy,
            'random': RandomSearchStrategy,
            'optuna': OptunaStrategy,
            'grid': GridSearchStrategy,
            'hyperopt': HyperOptStrategy,
            'bayesopt': BayesianOptimizationStrategy,
            'asha': ASHAStrategy,
        }
        
        strategy_name = strategy_name.lower()
        if strategy_name not in strategies:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available: {list(strategies.keys())}"
            )
        
        # Create strategy with kwargs
        strategy = strategies[strategy_name](**kwargs)
        
        # Apply optimization config if provided
        if optimization_config:
            strategy.set_optimization_config(optimization_config)
        
        return strategy
