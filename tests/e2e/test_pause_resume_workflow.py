"""
End-to-end test for pause/resume workflow.

This test simulates a complete pause/resume cycle.
"""

import pytest
import tempfile
import yaml
import pickle
import time
import json
import subprocess
import signal
from pathlib import Path
import sys
import os
from unittest.mock import MagicMock

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import ray
    from ray import tune
except ImportError:
    ray = None
    tune = None

from LightningTune import (
    ConfigDrivenOptimizer,
    SearchSpace,
    OptimizationConfig,
    TuneReflowCLI,
)
from LightningTune.core.strategies import BOHBStrategy
from LightningTune.cli.tune_reflow import TuneSessionState


class MinimalSearchSpace(SearchSpace):
    """Minimal search space for testing."""
    
    def get_search_space(self):
        if tune:
            return {
                "model.init_args.learning_rate": tune.choice([0.001, 0.01]),
            }
        else:
            return {
                "model.init_args.learning_rate": [0.001, 0.01],
            }
    
    def get_metric_config(self):
        return {"metric": "val_loss", "mode": "min"}


@pytest.mark.skipif(True, reason="Requires LightningTune to be importable by Ray workers")
class TestPauseResumeWorkflow:
    """Test complete pause/resume workflow."""
    
    def test_complete_pause_resume_cycle(self):
        """Test a complete pause and resume cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config file
            config = {
                "model": {
                    "class_path": "tests.fixtures.dummy_model.DummyModel",
                    "init_args": {"learning_rate": 0.001}
                },
                "data": {
                    "class_path": "tests.fixtures.dummy_model.DummyDataModule",
                    "init_args": {"num_samples": 50, "batch_size": 32}
                },
                "trainer": {
                    "max_epochs": 2,
                    "accelerator": "cpu",
                    "enable_progress_bar": False,
                    "logger": False,
                }
            }
            
            config_path = Path(tmpdir) / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # PHASE 1: Start optimization and pause
            experiment_name = "pause_resume_test"
            cli = TuneReflowCLI(
                experiment_name=experiment_name,
                experiment_dir=tmpdir,
            )
            
            strategy = BOHBStrategy(max_t=2, reduction_factor=2)
            search_space = MinimalSearchSpace()
            
            optimizer = ConfigDrivenOptimizer(
                base_config_path=str(config_path),
                search_space=search_space,
                strategy=strategy,
                optimization_config=OptimizationConfig(
                    max_epochs=2,
                    max_concurrent_trials=1,
                    experiment_name=experiment_name,
                    experiment_dir=tmpdir,
                ),
            )
            
            # Save state before running
            state = cli.save_optimizer_config(optimizer)
            
            # Verify state was saved
            assert cli.state_file.exists()
            assert state.experiment_name == experiment_name
            
            # Verify pickled objects
            restored_search_space = pickle.loads(state.search_space_pickle)
            assert isinstance(restored_search_space, MinimalSearchSpace)
            
            restored_strategy = pickle.loads(state.strategy_pickle)
            assert isinstance(restored_strategy, BOHBStrategy)
            assert restored_strategy.grace_period == 1
            
            # PHASE 2: Test resume command generation
            resume_command = cli._generate_resume_command()
            assert "--resume-session" in resume_command
            assert str(cli.session_dir) in resume_command
            
            # PHASE 3: Test loading from saved state
            cli2 = TuneReflowCLI.resume(session_dir=cli.session_dir)
            
            # Verify loaded correctly
            assert cli2.experiment_name == experiment_name
            assert cli2.session_id == cli.session_id
            
            # Restore optimizer
            restored_optimizer = cli2.restore_optimizer(state)
            
            # Verify optimizer restored correctly
            assert restored_optimizer.base_config_path == str(config_path)
            assert isinstance(restored_optimizer.strategy, BOHBStrategy)
            assert restored_optimizer.strategy.grace_period == 1
            
            # Cleanup
            ray.shutdown()
    
    def test_pause_signal_detection(self):
        """Test that pause signals are detected correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from LightningTune.callbacks.tune_pause_callback import TunePauseCallback
            
            pause_signal_file = Path(tmpdir) / "pause.signal"
            
            # Create callback
            callback = TunePauseCallback(
                pause_signal_file=pause_signal_file,
                check_interval=0,  # Check immediately
            )
            
            # Initially no signal
            assert callback._check_pause_signal() == False
            
            # Write pause signal
            signal_data = {
                "pause_requested": True,
                "timestamp": time.time(),
                "experiment_name": "test",
                "session_id": "123",
            }
            pause_signal_file.write_text(json.dumps(signal_data))
            
            # Now should detect signal
            assert callback._check_pause_signal() == True
            
            # Test that pause only executes once
            mock_trainer = MagicMock()
            mock_module = MagicMock()
            
            callback._execute_pause(mock_trainer, mock_module)
            assert callback._pause_executed == True
            
            # Should not execute again
            assert callback._check_pause_signal() == False
    
    def test_session_state_completeness(self):
        """Test that session state contains everything needed for resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create all necessary components
            strategy = BOHBStrategy(
                max_t=100,
                reduction_factor=3,
                metric="custom_metric",
                mode="max",
            )
            
            search_space = MinimalSearchSpace()
            
            optimization_config = OptimizationConfig(
                max_epochs=50,
                max_concurrent_trials=4,
                experiment_name="complete_test",
                experiment_dir=tmpdir,
                resources_per_trial={"cpu": 2, "gpu": 0.5},
            )
            
            # Create state
            state = TuneSessionState(
                experiment_name="complete_test",
                experiment_dir=tmpdir,
                session_id="full_123",
                base_config_path="/path/to/config.yaml",
                search_space_pickle=pickle.dumps(search_space),
                strategy_pickle=pickle.dumps(strategy),
                optimization_config_pickle=pickle.dumps(optimization_config),
                pause_requested=True,
                trials_paused=2,
                total_trials=4,
                checkpoint_dir=str(Path(tmpdir) / "checkpoints"),
                original_argv=["python", "optimize.py", "--arg1", "value1"],
            )
            
            # Save and load
            state_path = Path(tmpdir) / "complete_state.pkl"
            state.save(state_path)
            loaded_state = TuneSessionState.load(state_path)
            
            # Verify everything preserved
            assert loaded_state.experiment_name == "complete_test"
            assert loaded_state.session_id == "full_123"
            assert loaded_state.base_config_path == "/path/to/config.yaml"
            assert loaded_state.pause_requested == True
            assert loaded_state.trials_paused == 2
            assert loaded_state.total_trials == 4
            assert loaded_state.original_argv == ["python", "optimize.py", "--arg1", "value1"]
            
            # Verify complex objects
            loaded_strategy = pickle.loads(loaded_state.strategy_pickle)
            assert loaded_strategy.grace_period == 10
            assert loaded_strategy.reduction_factor == 3
            assert loaded_strategy.max_t == 100
            assert loaded_strategy.metric == "custom_metric"
            assert loaded_strategy.mode == "max"
            
            loaded_config = pickle.loads(loaded_state.optimization_config_pickle)
            assert loaded_config.max_epochs == 50
            assert loaded_config.max_concurrent_trials == 4
            assert loaded_config.resources_per_trial["cpu"] == 2
    
    def test_cli_workflow_simulation(self):
        """Simulate the CLI workflow for pause/resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test script that uses the CLI
            test_script = Path(tmpdir) / "test_optimize.py"
            test_script.write_text("""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from LightningTune import ConfigDrivenOptimizer, TuneReflowCLI
from LightningTune.core.strategies import BOHBStrategy
from tests.e2e.test_pause_resume_workflow import MinimalSearchSpace

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--resume-session", type=str)
args = parser.parse_args()

if args.resume_session:
    cli = TuneReflowCLI.resume(session_dir=args.resume_session)
    print(f"RESUMED: {cli.session_id}")
else:
    cli = TuneReflowCLI(experiment_name="cli_test", experiment_dir="{tmpdir}")
    strategy = BOHBStrategy(max_t=10, reduction_factor=2)
    optimizer = ConfigDrivenOptimizer(
        base_config_path=args.config,
        search_space=MinimalSearchSpace(),
        strategy=strategy,
    )
    cli.save_optimizer_config(optimizer)
    print(f"CREATED: {cli.session_id}")
    print(f"SESSION_DIR: {cli.session_dir}")
""".replace("{tmpdir}", tmpdir))
            
            # Create config
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("""
model:
  class_path: tests.fixtures.dummy_model.DummyModel
data:
  class_path: tests.fixtures.dummy_model.DummyDataModule
trainer:
  max_epochs: 1
""")
            
            # Run initial session
            result = subprocess.run(
                [sys.executable, str(test_script), "--config", str(config_path)],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )
            
            # Parse output
            output_lines = result.stdout.strip().split('\n')
            session_id = None
            session_dir = None
            
            for line in output_lines:
                if line.startswith("CREATED:"):
                    session_id = line.split(":")[1].strip()
                elif line.startswith("SESSION_DIR:"):
                    session_dir = line.split(":", 1)[1].strip()
            
            assert session_id is not None
            assert session_dir is not None
            assert Path(session_dir).exists()
            
            # Test resume
            result2 = subprocess.run(
                [sys.executable, str(test_script), 
                 "--config", str(config_path),
                 "--resume-session", session_dir],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )
            
            # Verify resumed correctly
            assert f"RESUMED: {session_id}" in result2.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])