"""
Unit tests for pause/resume functionality.
"""

import pytest
import pickle
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import sys
import tempfile

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from LightningTune.cli.tune_reflow import TuneSessionState, TuneReflowCLI
from LightningTune.callbacks.tune_pause_callback import TunePauseCallback
from LightningTune.core.strategies import BOHBStrategy
from LightningTune.core.config import SearchSpace


# Module-level class for pickling tests
class TestSearchSpaceForPickle(SearchSpace):
    """Test search space that can be pickled."""
    def get_search_space(self):
        return {"lr": [0.001, 0.01]}
    
    def get_metric_config(self):
        return {"metric": "val_loss", "mode": "min"}


class TestTuneSessionState:
    """Test session state serialization."""
    
    def test_init(self):
        """Test initialization."""
        state = TuneSessionState(
            experiment_name="test_exp",
            experiment_dir="/tmp/experiments",
            session_id="12345",
            base_config_path="/path/to/config.yaml",
            search_space_pickle=b"search_space",
            strategy_pickle=b"strategy",
            optimization_config_pickle=b"config",
        )
        
        assert state.experiment_name == "test_exp"
        assert state.experiment_dir == "/tmp/experiments"
        assert state.session_id == "12345"
        assert state.base_config_path == "/path/to/config.yaml"
        assert state.search_space_pickle == b"search_space"
        assert state.strategy_pickle == b"strategy"
        assert state.optimization_config_pickle == b"config"
        assert state.pause_requested == False
        assert state.trials_paused == 0
    
    def test_save_load(self):
        """Test saving and loading state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.pkl"
            
            # Create state
            state = TuneSessionState(
                experiment_name="my_exp",
                experiment_dir=tmpdir,
                session_id="abc123",
                base_config_path="config.yaml",
                search_space_pickle=pickle.dumps({"lr": [0.001, 0.01]}),
                strategy_pickle=pickle.dumps(BOHBStrategy(max_t=100)),
                optimization_config_pickle=pickle.dumps({"max_epochs": 100}),
                pause_requested=True,
                trials_paused=3,
                total_trials=4,
            )
            
            # Save
            state.save(state_path)
            
            # Check files exist
            assert state_path.exists()
            assert state_path.with_suffix('.summary.yaml').exists()
            
            # Load
            loaded_state = TuneSessionState.load(state_path)
            
            # Verify
            assert loaded_state.experiment_name == "my_exp"
            assert loaded_state.session_id == "abc123"
            assert loaded_state.pause_requested == True
            assert loaded_state.trials_paused == 3
            assert loaded_state.total_trials == 4
            
            # Verify pickled objects
            search_space = pickle.loads(loaded_state.search_space_pickle)
            assert search_space == {"lr": [0.001, 0.01]}
            
            strategy = pickle.loads(loaded_state.strategy_pickle)
            assert isinstance(strategy, BOHBStrategy)


class TestTuneReflowCLI:
    """Test CLI for pause/resume."""
    
    def test_init(self):
        """Test initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cli = TuneReflowCLI(
                experiment_name="test",
                experiment_dir=tmpdir,
                session_id="test123",
            )
            
            assert cli.experiment_name == "test"
            assert cli.experiment_dir == Path(tmpdir)
            assert cli.session_id == "test123"
            assert cli.enable_pause == True
            
            # Check session directory created
            expected_dir = Path(tmpdir) / "test" / "session_test123"
            assert expected_dir.exists()
    
    @pytest.mark.skip(reason="Ray is now a required dependency")
    def test_init_no_ray(self):
        """Test initialization without Ray."""
        with pytest.raises(ImportError, match="Ray is required"):
            TuneReflowCLI(experiment_name="test")
    
    def test_save_optimizer_config(self):
        """Test saving optimizer configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cli = TuneReflowCLI(
                experiment_name="test",
                experiment_dir=tmpdir,
            )
            
            # Mock optimizer
            mock_optimizer = MagicMock()
            mock_optimizer.base_config_path = "config.yaml"
            mock_optimizer.search_space = {"lr": [0.001, 0.01]}
            mock_optimizer.strategy = BOHBStrategy(max_t=100)
            mock_optimizer.optimization_config = {"max_epochs": 50}
            
            # Save config
            state = cli.save_optimizer_config(mock_optimizer)
            
            # Verify state
            assert state.experiment_name == "test"
            assert state.base_config_path == "config.yaml"
            assert pickle.loads(state.search_space_pickle) == {"lr": [0.001, 0.01]}
            assert isinstance(pickle.loads(state.strategy_pickle), BOHBStrategy)
            assert pickle.loads(state.optimization_config_pickle) == {"max_epochs": 50}
            
            # Check file saved
            assert cli.state_file.exists()
    
    def test_restore_optimizer(self):
        """Test restoring optimizer from state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cli = TuneReflowCLI(
                experiment_name="test",
                experiment_dir=tmpdir,
            )
            
            # Create state
            state = TuneSessionState(
                experiment_name="test",
                experiment_dir=tmpdir,
                session_id="123",
                base_config_path="config.yaml",
                search_space_pickle=pickle.dumps({"lr": [0.001, 0.01]}),
                strategy_pickle=pickle.dumps(BOHBStrategy(max_t=50, reduction_factor=2)),
                optimization_config_pickle=pickle.dumps({"max_epochs": 100}),
            )
            
            # Restore
            with patch('LightningTune.core.optimizer.ConfigDrivenOptimizer') as MockOptimizer:
                optimizer = cli.restore_optimizer(state)
                
                # Check optimizer created with correct args
                MockOptimizer.assert_called_once()
                call_args = MockOptimizer.call_args
                
                assert call_args[1]['base_config_path'] == "config.yaml"
                assert call_args[1]['search_space'] == {"lr": [0.001, 0.01]}
                assert isinstance(call_args[1]['strategy'], BOHBStrategy)
                assert call_args[1]['strategy'].max_t == 50
                assert call_args[1]['optimization_config'] == {"max_epochs": 100}
    
    def test_pause_signal_file(self):
        """Test pause signal file creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cli = TuneReflowCLI(
                experiment_name="test",
                experiment_dir=tmpdir,
            )
            
            # Request pause
            cli._request_pause()
            
            # Check signal file created
            assert cli.pause_signal_file.exists()
            
            # Check content
            signal_data = json.loads(cli.pause_signal_file.read_text())
            assert signal_data["pause_requested"] == True
            assert signal_data["experiment_name"] == "test"
            assert "timestamp" in signal_data
    
    def test_resume_class_method(self):
        """Test resume class method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create session directory and state
            session_dir = Path(tmpdir) / "test" / "session_123"
            session_dir.mkdir(parents=True)
            
            state = TuneSessionState(
                experiment_name="test",
                experiment_dir=tmpdir,
                session_id="123",
                base_config_path="config.yaml",
                search_space_pickle=b"search",
                strategy_pickle=b"strategy",
                optimization_config_pickle=b"config",
            )
            
            state_file = session_dir / "session_state.pkl"
            state.save(state_file)
            
            # Resume
            cli = TuneReflowCLI.resume(session_dir)
            
            assert cli.experiment_name == "test"
            assert cli.session_id == "123"
            assert cli.state == state
            assert cli.session_dir == session_dir


class TestTunePauseCallback:
    """Test pause callback."""
    
    def test_init(self):
        """Test initialization."""
        pause_file = Path("/tmp/pause.signal")
        callback = TunePauseCallback(
            pause_signal_file=pause_file,
            check_interval=2.0,
            verbose=True,
        )
        
        assert callback.pause_signal_file == pause_file
        assert callback.check_interval == 2.0
        assert callback.verbose == True
        assert callback._last_check_time == 0
        assert callback._pause_executed == False
    
    def test_check_pause_signal(self):
        """Test pause signal detection."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.signal', delete=False) as f:
            pause_file = Path(f.name)
            
            # Write signal
            signal_data = {
                "pause_requested": True,
                "timestamp": time.time(),
            }
            f.write(json.dumps(signal_data))
            f.flush()
            
            # Create callback
            callback = TunePauseCallback(pause_signal_file=pause_file)
            
            # Check signal
            assert callback._check_pause_signal() == True
            
            # Clean up
            pause_file.unlink()
    
    def test_no_pause_signal(self):
        """Test when no pause signal exists."""
        callback = TunePauseCallback(
            pause_signal_file=Path("/tmp/nonexistent.signal")
        )
        
        assert callback._check_pause_signal() == False
    
    def test_execute_pause(self):
        """Test pause execution."""
        callback = TunePauseCallback()
        
        # Mock trainer and module
        mock_trainer = MagicMock()
        mock_module = MagicMock()
        
        # Execute pause
        callback._execute_pause(mock_trainer, mock_module)
        
        # Check trainer stopped
        assert mock_trainer.should_stop == True
        
        # Check checkpoint reported
        # Note: Ray train reporting not tested here since ray is not imported
        # This would be tested in E2E tests with actual Ray
    
    def test_on_validation_end(self):
        """Test validation end hook."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.signal') as f:
            pause_file = Path(f.name)
            
            callback = TunePauseCallback(
                pause_signal_file=pause_file,
                check_interval=0,  # Check immediately
            )
            
            # Mock trainer and module
            mock_trainer = MagicMock()
            mock_module = MagicMock()
            
            # No signal yet
            callback.on_validation_end(mock_trainer, mock_module)
            assert mock_trainer.should_stop != True
            
            # Write signal
            signal_data = {"pause_requested": True, "timestamp": time.time()}
            f.write(json.dumps(signal_data))
            f.flush()
            
            # Now should pause
            with patch.object(callback, '_execute_pause') as mock_execute:
                callback.on_validation_end(mock_trainer, mock_module)
                mock_execute.assert_called_once_with(mock_trainer, mock_module)


class TestPickleCompatibility:
    """Test that all components can be pickled for pause/resume."""
    
    def test_pickle_search_space(self):
        """Test search space pickling."""
        search_space = TestSearchSpaceForPickle()
        
        pickled = pickle.dumps(search_space)
        restored = pickle.loads(pickled)
        
        assert restored.get_search_space() == {"lr": [0.001, 0.01]}
        assert restored.get_metric_config() == {"metric": "val_loss", "mode": "min"}
    
    def test_pickle_optimization_config(self):
        """Test optimization config pickling."""
        from LightningTune.core.strategies import OptimizationConfig
        
        config = OptimizationConfig(
            max_epochs=100,
            max_concurrent_trials=4,
            experiment_name="test",
        )
        
        pickled = pickle.dumps(config)
        restored = pickle.loads(pickled)
        
        assert restored.max_epochs == 100
        assert restored.max_concurrent_trials == 4
        assert restored.experiment_name == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])