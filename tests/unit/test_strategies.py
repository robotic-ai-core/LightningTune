"""
Unit tests for optimization strategies.
"""

import pytest
import pickle
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from LightningTune.core.strategies import (
    BOHBStrategy,
    OptunaStrategy,
    RandomSearchStrategy,
    PBTStrategy,
    GridSearchStrategy,
    OptimizationStrategy,
)


class TestBOHBStrategy:
    """Test BOHB strategy."""
    
    def test_init(self):
        """Test initialization."""
        strategy = BOHBStrategy(
            max_t=100,
            reduction_factor=3,
            metric="val_loss",
            mode="min"
        )
        
        assert strategy.max_t == 100
        assert strategy.reduction_factor == 3
        assert strategy.metric == "val_loss"
        assert strategy.mode == "min"
    
    def test_get_strategy_name(self):
        """Test strategy name."""
        strategy = BOHBStrategy()
        assert strategy.get_strategy_name() == "BOHB"
    
    def test_pickle_unpickle(self):
        """Test that strategy can be pickled and unpickled."""
        strategy = BOHBStrategy(
            max_t=50,
            reduction_factor=2,
            metric="accuracy",
            mode="max"
        )
        
        # Pickle
        pickled = pickle.dumps(strategy)
        
        # Unpickle
        restored = pickle.loads(pickled)
        
        assert restored.max_t == 50
        assert restored.reduction_factor == 2
        assert restored.metric == "accuracy"
        assert restored.mode == "max"
        assert restored.get_strategy_name() == "BOHB"
    
    def test_get_scheduler(self):
        """Test scheduler creation."""
        strategy = BOHBStrategy(max_t=100)
        
        # Get scheduler - should work since Ray is installed
        scheduler = strategy.get_scheduler()
        assert scheduler is not None
        from ray.tune.schedulers import HyperBandForBOHB
        assert isinstance(scheduler, HyperBandForBOHB)
    
    def test_get_search_algorithm(self):
        """Test search algorithm creation."""
        strategy = BOHBStrategy()
        
        # Try to get search algorithm - may fail if hpbandster not installed
        try:
            search_alg = strategy.get_search_algorithm()
            assert search_alg is not None
        except (ImportError, AssertionError) as e:
            # Skip if hpbandster not installed
            if "HpBandSter" in str(e) or "hpbandster" in str(e):
                pytest.skip("HpBandSter not installed")
            else:
                raise


class TestOptunaStrategy:
    """Test Optuna strategy."""
    
    def test_init(self):
        """Test initialization."""
        strategy = OptunaStrategy(
            n_startup_trials=10,
            use_pruner=True,
            pruner_type="median",
            metric="val_loss",
            mode="min"
        )
        
        assert strategy.n_startup_trials == 10
        assert strategy.use_pruner == True
        assert strategy.pruner_type == "median"
        assert strategy.metric == "val_loss"
        assert strategy.mode == "min"
    
    def test_get_strategy_name(self):
        """Test strategy name."""
        strategy = OptunaStrategy()
        assert strategy.get_strategy_name() == "Optuna"
    
    def test_pickle_unpickle(self):
        """Test pickling."""
        strategy = OptunaStrategy(
            n_startup_trials=20,
            pruner_type="hyperband"
        )
        
        pickled = pickle.dumps(strategy)
        restored = pickle.loads(pickled)
        
        assert restored.n_startup_trials == 20
        assert restored.pruner_type == "hyperband"


class TestRandomSearchStrategy:
    """Test Random Search strategy."""
    
    def test_init(self):
        """Test initialization."""
        strategy = RandomSearchStrategy(
            num_samples=100,
            metric="accuracy",
            mode="max"
        )
        
        assert strategy.num_samples == 100
        assert strategy.metric == "accuracy"
        assert strategy.mode == "max"
    
    def test_get_strategy_name(self):
        """Test strategy name."""
        strategy = RandomSearchStrategy()
        assert strategy.get_strategy_name() == "RandomSearch"
    
    def test_pickle_unpickle(self):
        """Test pickling."""
        strategy = RandomSearchStrategy(num_samples=50)
        
        pickled = pickle.dumps(strategy)
        restored = pickle.loads(pickled)
        
        assert restored.num_samples == 50


class TestPBTStrategy:
    """Test PBT strategy."""
    
    def test_init(self):
        """Test initialization."""
        strategy = PBTStrategy(
            perturbation_interval=4,
            hyperparam_mutations={"lr": [0.001, 0.01]},
            metric="val_loss",
            mode="min"
        )
        
        assert strategy.perturbation_interval == 4
        assert strategy.hyperparam_mutations == {"lr": [0.001, 0.01]}
        assert strategy.metric == "val_loss"
        assert strategy.mode == "min"
    
    def test_get_strategy_name(self):
        """Test strategy name."""
        strategy = PBTStrategy()
        assert strategy.get_strategy_name() == "PBT"
    
    def test_pickle_unpickle(self):
        """Test pickling."""
        strategy = PBTStrategy(
            perturbation_interval=5,
            hyperparam_mutations={"batch_size": [16, 32, 64]}
        )
        
        pickled = pickle.dumps(strategy)
        restored = pickle.loads(pickled)
        
        assert restored.perturbation_interval == 5
        assert restored.hyperparam_mutations == {"batch_size": [16, 32, 64]}


class TestGridSearchStrategy:
    """Test Grid Search strategy."""
    
    def test_init(self):
        """Test initialization."""
        strategy = GridSearchStrategy(
            metric="val_acc",
            mode="max"
        )
        
        assert strategy.metric == "val_acc"
        assert strategy.mode == "max"
    
    def test_get_strategy_name(self):
        """Test strategy name."""
        strategy = GridSearchStrategy()
        assert strategy.get_strategy_name() == "GridSearch"
    
    def test_pickle_unpickle(self):
        """Test pickling."""
        strategy = GridSearchStrategy(metric="loss", mode="min")
        
        pickled = pickle.dumps(strategy)
        restored = pickle.loads(pickled)
        
        assert restored.metric == "loss"
        assert restored.mode == "min"


class TestDependencyInjection:
    """Test dependency injection pattern."""
    
    def test_strategies_are_self_contained(self):
        """Test that strategies contain all their configuration."""
        strategies = [
            BOHBStrategy(max_t=100, reduction_factor=2),
            OptunaStrategy(n_startup_trials=10),
            RandomSearchStrategy(num_samples=20),
            PBTStrategy(perturbation_interval=3),
            GridSearchStrategy(),
        ]
        
        for strategy in strategies:
            # Check that each strategy has all needed attributes
            assert hasattr(strategy, 'metric')
            assert hasattr(strategy, 'mode')
            assert hasattr(strategy, 'get_strategy_name')
            assert callable(strategy.get_strategy_name)
            
            # Check that strategy can be pickled (needed for pause/resume)
            pickled = pickle.dumps(strategy)
            restored = pickle.loads(pickled)
            
            assert restored.get_strategy_name() == strategy.get_strategy_name()
            assert restored.metric == strategy.metric
            assert restored.mode == strategy.mode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])