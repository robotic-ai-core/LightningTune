"""
Optuna optimization strategies for LightningTune.

This module provides various optimization strategies using Optuna's
samplers and pruners, offering modern alternatives to Ray Tune strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import optuna
from optuna.samplers import (
    TPESampler,
    RandomSampler,
    GridSampler,
    CmaEsSampler,
)
from optuna.pruners import (
    SuccessiveHalvingPruner,
    HyperbandPruner,
    MedianPruner,
    NopPruner,
)


class OptunaStrategy(ABC):
    """Abstract base class for Optuna optimization strategies."""
    
    @abstractmethod
    def create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create the Optuna sampler for this strategy."""
        pass
    
    @abstractmethod
    def create_pruner(self) -> optuna.pruners.BasePruner:
        """Create the Optuna pruner for this strategy."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this strategy."""
        pass
    
    def create_study(
        self, 
        study_name: str,
        storage: Optional[str] = None,
        direction: str = "minimize",
        load_if_exists: bool = True
    ) -> optuna.Study:
        """Create an Optuna study with this strategy's sampler and pruner."""
        return optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=self.create_sampler(),
            pruner=self.create_pruner(),
            direction=direction,
            load_if_exists=load_if_exists
        )


class BOHBStrategy(OptunaStrategy):
    """
    BOHB (Bayesian Optimization with HyperBand) strategy.
    
    Uses TPE sampler with Hyperband pruning for efficient optimization.
    This is the Optuna equivalent of Ray Tune's BOHB.
    """
    
    def __init__(
        self,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        min_resource: int = 1,
        max_resource: int = 100,
        reduction_factor: int = 3,
    ):
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor
    
    def create_sampler(self) -> optuna.samplers.BaseSampler:
        return TPESampler(
            n_startup_trials=self.n_startup_trials,
            n_ei_candidates=self.n_ei_candidates,
        )
    
    def create_pruner(self) -> optuna.pruners.BasePruner:
        return HyperbandPruner(
            min_resource=self.min_resource,
            max_resource=self.max_resource,
            reduction_factor=self.reduction_factor,
        )
    
    @property
    def name(self) -> str:
        return "BOHB"


class TPEStrategy(OptunaStrategy):
    """
    Tree-structured Parzen Estimator (TPE) strategy.
    
    Optuna's default and most popular optimization algorithm.
    Efficient for most hyperparameter optimization tasks.
    """
    
    def __init__(
        self,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        use_pruner: bool = True,
        n_warmup_steps: int = 5,
    ):
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.use_pruner = use_pruner
        self.n_warmup_steps = n_warmup_steps
    
    def create_sampler(self) -> optuna.samplers.BaseSampler:
        return TPESampler(
            n_startup_trials=self.n_startup_trials,
            n_ei_candidates=self.n_ei_candidates,
        )
    
    def create_pruner(self) -> optuna.pruners.BasePruner:
        if self.use_pruner:
            return MedianPruner(n_warmup_steps=self.n_warmup_steps)
        return NopPruner()
    
    @property
    def name(self) -> str:
        return "TPE"


class RandomStrategy(OptunaStrategy):
    """
    Random search strategy.
    
    Simple baseline strategy that samples uniformly at random.
    Good for initial exploration and baseline comparisons.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
    
    def create_sampler(self) -> optuna.samplers.BaseSampler:
        return RandomSampler(seed=self.seed)
    
    def create_pruner(self) -> optuna.pruners.BasePruner:
        return NopPruner()  # No pruning for random search
    
    @property
    def name(self) -> str:
        return "Random"


class GridStrategy(OptunaStrategy):
    """
    Grid search strategy.
    
    Exhaustive search over a discrete set of hyperparameter values.
    Best for small search spaces or when you need complete coverage.
    """
    
    def __init__(self, search_space: Dict[str, Any]):
        """
        Initialize grid search with explicit search space.
        
        Args:
            search_space: Dictionary mapping parameter names to lists of values
        """
        self.search_space = search_space
    
    def create_sampler(self) -> optuna.samplers.BaseSampler:
        return GridSampler(self.search_space)
    
    def create_pruner(self) -> optuna.pruners.BasePruner:
        return NopPruner()  # No pruning for grid search
    
    @property
    def name(self) -> str:
        return "Grid"


class ASHAStrategy(OptunaStrategy):
    """
    ASHA (Asynchronous Successive Halving) strategy.
    
    Efficient early stopping strategy that promotes promising trials.
    Similar to BOHB but with simpler successive halving.
    """
    
    def __init__(
        self,
        min_resource: int = 1,
        max_resource: int = 100,
        reduction_factor: int = 3,
        use_tpe: bool = True,
    ):
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor
        self.use_tpe = use_tpe
    
    def create_sampler(self) -> optuna.samplers.BaseSampler:
        if self.use_tpe:
            return TPESampler()
        return RandomSampler()
    
    def create_pruner(self) -> optuna.pruners.BasePruner:
        return SuccessiveHalvingPruner(
            min_resource=self.min_resource,
            max_resource=self.max_resource,
            reduction_factor=self.reduction_factor,
        )
    
    @property
    def name(self) -> str:
        return "ASHA"


class CMAESStrategy(OptunaStrategy):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    
    Powerful derivative-free optimization algorithm.
    Best for continuous parameters and smooth objectives.
    """
    
    def __init__(
        self,
        sigma0: float = 1.0,
        seed: Optional[int] = None,
        n_warmup_steps: int = 5,
    ):
        self.sigma0 = sigma0
        self.seed = seed
        self.n_warmup_steps = n_warmup_steps
    
    def create_sampler(self) -> optuna.samplers.BaseSampler:
        return CmaEsSampler(sigma0=self.sigma0, seed=self.seed)
    
    def create_pruner(self) -> optuna.pruners.BasePruner:
        return MedianPruner(n_warmup_steps=self.n_warmup_steps)
    
    @property
    def name(self) -> str:
        return "CMA-ES"