"""
Integration tests for the optimizer with various strategies.
"""

import pytest
import tempfile
import yaml
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from LightningTune import ConfigDrivenOptimizer, SearchSpace
from LightningTune.core.strategies import (
    BOHBStrategy,
    OptunaStrategy,
    RandomSearchStrategy,
    GridSearchStrategy,
    ASHAStrategy,
    OptimizationConfig,
)


class TestSearchSpace(SearchSpace):
    """Test search space for integration tests."""
    
    def get_search_space(self):
        # Use simple discrete choices for faster testing
        return {
            "model.init_args.learning_rate": [0.001, 0.01],
            "model.init_args.hidden_dim": [16, 32],
            "data.init_args.batch_size": [16, 32],
        }
    
    def get_metric_config(self):
        return {"metric": "val_loss", "mode": "min"}


class TestOptimizerIntegration:
    """Integration tests for ConfigDrivenOptimizer."""
    
    @pytest.fixture
    def base_config(self):
        """Create base config for testing."""
        config = {
            "model": {
                "class_path": "tests.fixtures.dummy_model.DummyModel",
                "init_args": {
                    "input_dim": 10,
                    "hidden_dim": 32,
                    "output_dim": 2,
                    "learning_rate": 0.001,
                    "dropout": 0.1,
                }
            },
            "data": {
                "class_path": "tests.fixtures.dummy_model.DummyDataModule",
                "init_args": {
                    "batch_size": 32,
                    "num_samples": 100,
                    "input_dim": 10,
                    "num_classes": 2,
                }
            },
            "trainer": {
                "max_epochs": 2,
                "accelerator": "cpu",
                "devices": 1,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            }
        }
        return config
    
    @pytest.fixture
    def config_file(self, base_config):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(base_config, f)
            config_path = Path(f.name)
        
        yield config_path
        
        # Cleanup
        if config_path.exists():
            config_path.unlink()
    
    def test_optimizer_init_with_strategy(self, config_file):
        """Test optimizer initialization with dependency injection."""
        strategy = BOHBStrategy(max_t=2, reduction_factor=2)
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source=str(config_file),
            search_space=TestSearchSpace(),
            strategy=strategy,
            optimization_config=OptimizationConfig(
                max_epochs=2,
                max_concurrent_trials=1,
                experiment_name="test",
            ),
        )
        
        assert str(optimizer.base_config_path) == str(config_file)
        assert isinstance(optimizer.search_space, TestSearchSpace)
        assert isinstance(optimizer.strategy, BOHBStrategy)
        assert optimizer.strategy.max_t == 2
    
    def test_config_merging(self, config_file, base_config):
        """Test that configs are merged correctly."""
        optimizer = ConfigDrivenOptimizer(
            base_config_source=str(config_file),
            search_space=TestSearchSpace(),
            strategy=RandomSearchStrategy(num_samples=1),
        )
        
        # Test merging
        overrides = {
            "model.init_args.learning_rate": 0.01,
            "data.init_args.batch_size": 16,
        }
        
        merged = optimizer.config_manager.merge_configs(base_config, overrides)
        
        # Check overrides applied
        assert merged["model"]["init_args"]["learning_rate"] == 0.01
        assert merged["data"]["init_args"]["batch_size"] == 16
        
        # Check base values preserved
        assert merged["model"]["init_args"]["hidden_dim"] == 32
        assert merged["model"]["init_args"]["dropout"] == 0.1
    
    def test_optimizer_pickle(self, config_file):
        """Test that optimizer can be pickled."""
        strategy = OptunaStrategy(n_startup_trials=5)
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source=str(config_file),
            search_space=TestSearchSpace(),
            strategy=strategy,
            optimization_config=OptimizationConfig(max_epochs=2),
        )
        
        # Pickle
        pickled = pickle.dumps(optimizer)
        
        # Unpickle
        restored = pickle.loads(pickled)
        
        assert str(restored.base_config_path) == str(config_file)
        assert isinstance(restored.strategy, OptunaStrategy)
        assert restored.strategy.n_startup_trials == 5
    
    @patch('LightningTune.core.optimizer.ray')
    @patch('LightningTune.core.optimizer.tune')
    def test_run_with_bohb_strategy(self, mock_tune, mock_ray, config_file):
        """Test running optimization with BOHB strategy."""
        mock_ray.available = MagicMock(return_value=True)
        mock_ray.is_initialized = MagicMock(return_value=False)
        mock_ray.init = MagicMock()
        
        # Mock tuner with proper result structure
        mock_tuner = MagicMock()
        mock_results = MagicMock()
        mock_best_result = MagicMock()
        mock_best_result.config = {"model.init_args.learning_rate": 0.01}
        mock_best_result.metrics = {"val_loss": 0.5}
        mock_results.get_best_result.return_value = mock_best_result
        mock_results.__len__ = MagicMock(return_value=2)
        mock_tuner.fit.return_value = mock_results
        mock_tune.Tuner.return_value = mock_tuner
        
        strategy = BOHBStrategy(max_t=2, reduction_factor=2)
        # Mock the search algorithm to avoid HpBandSter dependency
        strategy.get_search_algorithm = MagicMock(return_value=None)
        strategy.get_scheduler = MagicMock(return_value=None)
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source=str(config_file),
            search_space=TestSearchSpace(),
            strategy=strategy,
            optimization_config=OptimizationConfig(
                max_epochs=2,
                max_concurrent_trials=1,
            ),
        )
        
        # Run
        results = optimizer.run()
        
        # Verify Ray initialized
        mock_ray.init.assert_called_once()
        
        # Verify Tuner created and run
        mock_tune.Tuner.assert_called_once()
        mock_tuner.fit.assert_called_once()
        
        assert results == mock_results
    
    @patch('LightningTune.core.optimizer.ray')
    @patch('LightningTune.core.optimizer.tune')
    def test_run_with_random_strategy(self, mock_tune, mock_ray, config_file):
        """Test running optimization with Random Search strategy."""
        mock_ray.available = MagicMock(return_value=True)
        mock_ray.is_initialized = MagicMock(return_value=False)
        mock_ray.init = MagicMock()
        
        # Mock tuner with proper result structure
        mock_tuner = MagicMock()
        mock_results = MagicMock()
        mock_best_result = MagicMock()
        mock_best_result.config = {"model.init_args.learning_rate": 0.01}
        mock_best_result.metrics = {"val_loss": 0.5}
        mock_results.get_best_result.return_value = mock_best_result
        mock_results.__len__ = MagicMock(return_value=2)
        mock_tuner.fit.return_value = mock_results
        mock_tune.Tuner.return_value = mock_tuner
        
        strategy = RandomSearchStrategy(num_samples=2)
        # Mock the strategy methods to avoid dependencies
        strategy.get_search_algorithm = MagicMock(return_value=None)
        strategy.get_scheduler = MagicMock(return_value=None)
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source=str(config_file),
            search_space=TestSearchSpace(),
            strategy=strategy,
        )
        
        # Run
        results = optimizer.run()
        
        # Verify
        mock_tune.Tuner.assert_called_once()
        assert results == mock_results
    
    def test_callbacks_injection(self, config_file):
        """Test that callbacks can be injected."""
        from LightningTune.callbacks.tune_pause_callback import TunePauseCallback
        
        strategy = RandomSearchStrategy()
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source=str(config_file),
            search_space=TestSearchSpace(),
            strategy=strategy,
        )
        
        # Add callback
        pause_callback = TunePauseCallback()
        optimizer.additional_callbacks.append(pause_callback)
        
        assert len(optimizer.additional_callbacks) == 1
        assert isinstance(optimizer.additional_callbacks[0], TunePauseCallback)
    
    def test_different_strategies(self, config_file):
        """Test that different strategies can be used interchangeably."""
        strategies = [
            BOHBStrategy(max_t=10, reduction_factor=2),
            OptunaStrategy(n_startup_trials=2),
            RandomSearchStrategy(num_samples=2),
            GridSearchStrategy(),
        ]
        
        for strategy in strategies:
            optimizer = ConfigDrivenOptimizer(
                base_config_source=str(config_file),
                search_space=TestSearchSpace(),
                strategy=strategy,
            )
            
            assert optimizer.strategy == strategy
            assert optimizer.strategy.get_strategy_name() in [
                "BOHB", "Optuna", "RandomSearch", "GridSearch"
            ]


class TestSearchSpaceIntegration:
    """Test search space integration."""
    
    def test_search_space_with_ray_tune(self):
        """Test that search space works with Ray Tune distributions."""
        try:
            from ray import tune
            
            class RayTuneSearchSpace(SearchSpace):
                def get_search_space(self):
                    return {
                        "lr": tune.loguniform(1e-5, 1e-2),
                        "batch_size": tune.choice([16, 32, 64]),
                        "hidden_dim": tune.randint(32, 256),
                    }
                
                def get_metric_config(self):
                    return {"metric": "val_loss", "mode": "min"}
            
            search_space = RayTuneSearchSpace()
            space = search_space.get_search_space()
            
            # Verify Ray Tune distributions
            assert hasattr(space["lr"], "sample")
            assert hasattr(space["batch_size"], "sample")
            assert hasattr(space["hidden_dim"], "sample")
            
        except ImportError:
            pytest.skip("Ray not installed")
    
    def test_search_space_validation(self):
        """Test search space with validation."""
        
        class ValidatedSearchSpace(SearchSpace):
            def get_search_space(self):
                return {
                    "lr": [0.001, 0.01, 0.1],
                    "optimizer": ["adam", "sgd"],
                }
            
            def get_metric_config(self):
                return {"metric": "accuracy", "mode": "max"}
            
            def validate_config(self, config):
                # SGD needs higher learning rate
                if config.get("optimizer") == "sgd" and config.get("lr", 0) < 0.01:
                    return False
                return True
        
        search_space = ValidatedSearchSpace()
        
        # Test validation
        assert search_space.validate_config({"optimizer": "adam", "lr": 0.001}) == True
        assert search_space.validate_config({"optimizer": "sgd", "lr": 0.001}) == False
        assert search_space.validate_config({"optimizer": "sgd", "lr": 0.01}) == True


class TestStrategyIntegration:
    """Test strategy integration with optimizer."""
    
    @pytest.fixture
    def simple_config_file(self):
        """Create simple config file."""
        config = {
            "model": {"class_path": "test.Model", "init_args": {"lr": 0.001}},
            "data": {"class_path": "test.Data", "init_args": {"batch_size": 32}},
            "trainer": {"max_epochs": 1},
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = Path(f.name)
        
        yield config_path
        
        if config_path.exists():
            config_path.unlink()
    
    def test_strategy_parameters_used(self, simple_config_file):
        """Test that strategy parameters are actually used."""
        # Test with different strategies
        strategies_and_params = [
            (BOHBStrategy(max_t=50, reduction_factor=2), 
             {"max_t": 50, "reduction_factor": 2}),
            (OptunaStrategy(n_startup_trials=10),
             {"n_startup_trials": 10}),
            (RandomSearchStrategy(num_samples=20),
             {"num_samples": 20}),
        ]
        
        for strategy, expected_params in strategies_and_params:
            optimizer = ConfigDrivenOptimizer(
                base_config_source=str(simple_config_file),
                search_space=TestSearchSpace(),
                strategy=strategy,
            )
            
            # Verify strategy parameters
            for param, value in expected_params.items():
                assert getattr(optimizer.strategy, param) == value
    
    def test_strategy_metric_config(self, simple_config_file):
        """Test that strategy metric configuration is used."""
        strategy = BOHBStrategy(max_t=100, metric="custom_metric", mode="max")
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source=str(simple_config_file),
            search_space=TestSearchSpace(),
            strategy=strategy,
        )
        
        assert optimizer.strategy.metric == "custom_metric"
        assert optimizer.strategy.mode == "max"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])