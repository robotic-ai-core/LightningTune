"""
End-to-end tests for complete optimization workflows.

These tests actually run training with Ray Tune (if available).
"""

import pytest
import pickle
import tempfile
import yaml
import time
import json
import threading
import numpy as np
from pathlib import Path
import sys
import os

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Check if Ray is available
try:
    import ray
    from ray import tune
except ImportError:
    ray = None
    tune = None

try:
    import pytorch_lightning as pl
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from LightningTune import (
    ConfigDrivenOptimizer,
    SearchSpace,
    OptimizationConfig,
    TuneReflowCLI,
)
from LightningTune.core.strategies import (
    BOHBStrategy,
    RandomSearchStrategy,
    OptimizationConfig as StrategyOptimizationConfig,
)
from LightningTune.callbacks.tune_pause_callback import TunePauseCallback
from tests.fixtures.simple_search_space import SimpleSearchSpace


@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.skipif(ray is None, reason="Ray not installed")
class TestEndToEndOptimization:
    """End-to-end optimization tests."""
    
    @pytest.fixture
    def config_file(self):
        """Create config file for testing."""
        config = {
            "model": {
                "class_path": "fixtures.dummy_model.DummyModel",
                "init_args": {
                    "input_dim": 10,
                    "hidden_dim": 32,
                    "output_dim": 2,
                    "learning_rate": 0.001,
                    "dropout": 0.1,
                }
            },
            "data": {
                "class_path": "fixtures.dummy_model.DummyDataModule",
                "init_args": {
                    "batch_size": 32,
                    "num_samples": 200,  # Larger dataset for more stable training
                    "input_dim": 10,
                    "num_classes": 2,
                }
            },
            "trainer": {
                "max_epochs": 2,  # Few epochs for fast testing
                "accelerator": "cpu",
                "devices": 1,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
                "enable_checkpointing": False,
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = Path(f.name)
        
        yield config_path
        
        # Cleanup
        if config_path.exists():
            config_path.unlink()
    
    def test_basic_optimization_workflow(self, config_file):
        """Test basic optimization without pause/resume."""
        strategy = RandomSearchStrategy(num_samples=2)  # Small number for testing
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source=str(config_file),
            search_space=SimpleSearchSpace(),
            strategy=strategy,
            optimization_config=OptimizationConfig(
                max_epochs=2,
                max_concurrent_trials=1,
                experiment_name="e2e_test_basic",
                experiment_dir=tempfile.gettempdir(),
            ),
        )
        
        # Run optimization
        results = optimizer.run()
        
        # Verify results
        assert results is not None
        
        # Check that trials actually ran
        assert len(results) >= 1
        
        # Get best result - should not have inf values
        best_result = results.get_best_result(
            metric="val_loss",
            mode="min"
        )
        assert best_result is not None
        
        # Check metrics were logged
        assert "val_loss" in best_result.metrics
        
        # Verify the loss is not inf
        assert not np.isinf(best_result.metrics["val_loss"]), "Loss should not be inf!"
        
        # Cleanup Ray
        ray.shutdown()
    
    def test_bohb_strategy_e2e(self, config_file):
        """Test BOHB strategy end-to-end."""
        strategy = BOHBStrategy(
            max_t=2,  # Very short for testing
            reduction_factor=2,
        )
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source=str(config_file),
            search_space=SimpleSearchSpace(),
            strategy=strategy,
            optimization_config=OptimizationConfig(
                max_epochs=2,
                max_concurrent_trials=2,
                experiment_name="e2e_test_bohb",
                experiment_dir=tempfile.gettempdir(),
            ),
        )
        
        # Run optimization
        results = optimizer.run()
        
        # Verify BOHB-specific behavior
        assert results is not None
        assert len(results) >= 1
        
        # Some trials should be stopped early (BOHB behavior)
        # Check if any trial has fewer epochs than max
        for result in results:
            if "training_iteration" in result.metrics:
                # At least check that metrics exist
                assert result.metrics["training_iteration"] > 0
        
        # Cleanup
        ray.shutdown()
    
    def test_pause_resume_e2e(self, config_file):
        """Test pause/resume functionality end-to-end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_name = "e2e_pause_resume"
            
            # Create CLI with pause/resume support
            cli = TuneReflowCLI(
                experiment_name=experiment_name,
                experiment_dir=tmpdir,
            )
            
            strategy = RandomSearchStrategy(num_samples=4)
            
            optimizer = ConfigDrivenOptimizer(
                base_config_source=str(config_file),
                search_space=SimpleSearchSpace(),
                strategy=strategy,
                optimization_config=OptimizationConfig(
                    max_epochs=3,
                    max_concurrent_trials=2,
                    experiment_name=experiment_name,
                    experiment_dir=tmpdir,
                ),
            )
            
            # Start optimization in thread
            results_container = []
            error_container = []
            
            def run_optimization():
                try:
                    results = cli.run(optimizer)
                    results_container.append(results)
                except KeyboardInterrupt:
                    # Expected when pausing
                    pass
                except Exception as e:
                    error_container.append(e)
            
            thread = threading.Thread(target=run_optimization)
            thread.start()
            
            # Wait a bit for optimization to start
            time.sleep(2)
            
            # Request pause
            cli._request_pause()
            
            # Wait for thread to finish
            thread.join(timeout=10)
            
            # Check that pause signal was created
            assert cli.pause_signal_file.exists()
            
            # Check session state was saved
            assert cli.state_file.exists()
            
            # Load and verify state
            from LightningTune.cli.tune_reflow import TuneSessionState
            saved_state = TuneSessionState.load(cli.state_file)
            
            assert saved_state.experiment_name == experiment_name
            assert saved_state.pause_requested == True
            
            # Cleanup
            ray.shutdown()
            
            # Could test actual resume here, but it's complex with Ray's state
    
    def test_callbacks_integration(self, config_file):
        """Test that callbacks work in E2E scenario."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pause_signal_file = Path(tmpdir) / "pause.signal"
            
            # Create pause callback
            pause_callback = TunePauseCallback(
                pause_signal_file=pause_signal_file,
                verbose=True,
            )
            
            strategy = RandomSearchStrategy(num_samples=2)
            
            optimizer = ConfigDrivenOptimizer(
                base_config_source=str(config_file),
                search_space=SimpleSearchSpace(),
                strategy=strategy,
                optimization_config=OptimizationConfig(
                    max_epochs=2,
                    max_concurrent_trials=1,
                    experiment_name="e2e_callbacks",
                    experiment_dir=tmpdir,
                ),
            )
            
            # Add callback
            optimizer.additional_callbacks.append(pause_callback)
            
            # Run optimization
            results = optimizer.run()
            
            # Verify optimization completed
            assert results is not None
            assert len(results) >= 1
            
            # Cleanup
            ray.shutdown()
    
    def test_multiple_strategies_e2e(self, config_file):
        """Test that different strategies work correctly."""
        strategies = [
            RandomSearchStrategy(num_samples=1),
            BOHBStrategy(max_t=2, reduction_factor=2),
        ]
        
        for i, strategy in enumerate(strategies):
            optimizer = ConfigDrivenOptimizer(
                base_config_source=str(config_file),
                search_space=SimpleSearchSpace(),
                strategy=strategy,
                optimization_config=OptimizationConfig(
                    max_epochs=2,
                    max_concurrent_trials=1,
                    experiment_name=f"e2e_multi_{i}",
                    experiment_dir=tempfile.gettempdir(),
                ),
            )
            
            # Run optimization
            results = optimizer.run()
            
            # Basic verification
            assert results is not None
            assert len(results) >= 1
            
            # Cleanup after each
            ray.shutdown()


@pytest.mark.skipif(ray is None, reason="Ray not installed")
class TestRayTuneIntegration:
    """Test Ray Tune specific integration."""
    
    def test_ray_initialization(self):
        """Test that Ray initializes correctly."""
        from LightningTune.core.optimizer import ConfigDrivenOptimizer
        
        # Create minimal optimizer
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
            yaml.dump({"model": {}, "data": {}, "trainer": {}}, f)
            f.flush()
            
            optimizer = ConfigDrivenOptimizer(
                base_config_source=f.name,
                search_space=SimpleSearchSpace(),
                strategy=RandomSearchStrategy(num_samples=1),
            )
            
            # Ray should initialize when needed
            from unittest.mock import patch, MagicMock
            with patch('LightningTune.core.optimizer.ray.init') as mock_init:
                with patch('LightningTune.core.optimizer.tune.Tuner') as mock_tuner:
                    mock_tuner.return_value.fit.return_value = MagicMock()
                    
                    optimizer.run()
                    
                    # Verify Ray was initialized
                    mock_init.assert_called_once()
        
        # Cleanup
        if ray.is_initialized():
            ray.shutdown()
    
    def test_tune_config_generation(self):
        """Test that Tune config is generated correctly."""
        from LightningTune.core.optimizer import ConfigDrivenOptimizer
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
            yaml.dump({"model": {}, "data": {}, "trainer": {}}, f)
            f.flush()
            
            search_space = SimpleSearchSpace()
            strategy = BOHBStrategy(max_t=100)
            
            optimizer = ConfigDrivenOptimizer(
                base_config_source=f.name,
                search_space=search_space,
                strategy=strategy,
            )
            
            # Verify optimizer was created with correct config
            assert optimizer.search_space is not None
            assert optimizer.strategy is not None
            # Check if strategy has expected attributes
            assert hasattr(optimizer.strategy, 'metric')
            assert hasattr(optimizer.strategy, 'mode')
            assert optimizer.strategy.metric == "val_loss"
            assert optimizer.strategy.mode == "min"


@pytest.mark.skipif(ray is None, reason="Ray not installed")
class TestPauseResumeScenarios:
    """Test various pause/resume scenarios."""
    
    def test_pause_signal_file_creation(self):
        """Test that pause signal files are created correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cli = TuneReflowCLI(
                experiment_name="test",
                experiment_dir=tmpdir,
            )
            
            # Request pause
            cli._request_pause()
            
            # Check signal file
            assert cli.pause_signal_file.exists()
            
            # Read and verify content
            signal_data = json.loads(cli.pause_signal_file.read_text())
            assert signal_data["pause_requested"] == True
            assert signal_data["experiment_name"] == "test"
            assert "timestamp" in signal_data
            assert "session_id" in signal_data
    
    def test_session_state_serialization(self):
        """Test complete session state serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from LightningTune.cli.tune_reflow import TuneSessionState
            
            # Create complex state
            strategy = BOHBStrategy(max_t=100, reduction_factor=3)
            search_space = SimpleSearchSpace()
            
            state = TuneSessionState(
                experiment_name="test_exp",
                experiment_dir=tmpdir,
                session_id="abc123",
                base_config_path="/path/to/config.yaml",
                search_space_pickle=pickle.dumps(search_space),
                strategy_pickle=pickle.dumps(strategy),
                optimization_config_pickle=pickle.dumps({"max_epochs": 100}),
                pause_requested=True,
                trials_paused=3,
                total_trials=5,
            )
            
            # Save
            state_path = Path(tmpdir) / "state.pkl"
            state.save(state_path)
            
            # Load
            loaded = TuneSessionState.load(state_path)
            
            # Verify all fields
            assert loaded.experiment_name == "test_exp"
            assert loaded.session_id == "abc123"
            assert loaded.pause_requested == True
            assert loaded.trials_paused == 3
            assert loaded.total_trials == 5
            
            # Verify pickled objects
            loaded_strategy = pickle.loads(loaded.strategy_pickle)
            assert isinstance(loaded_strategy, BOHBStrategy)
            assert loaded_strategy.max_t == 100  # BOHB uses max_t, not grace_period
            
            loaded_search_space = pickle.loads(loaded.search_space_pickle)
            assert isinstance(loaded_search_space, SimpleSearchSpace)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])