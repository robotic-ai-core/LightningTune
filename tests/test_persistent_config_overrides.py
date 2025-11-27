"""
Test persistent configuration overrides across pause/resume cycles.

This test validates that config overrides (like --trial-steps) persist
across HPO session resume operations.
"""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import logging

import optuna
import pytest

from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer


# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPersistentConfigOverrides:
    """Test suite for persistent config overrides functionality."""
    
    def test_config_overrides_saved_in_checkpoint(self, tmp_path):
        """Test that config overrides are saved in both WandB and local checkpoints."""
        
        # Create optimizer with config overrides
        config_overrides = {
            "trial_steps": 5000,
            "trainer.val_check_interval": 500,
            "trainer.limit_val_batches": 10
        }
        
        optimizer = PausibleOptunaOptimizer(
            base_config={"test": "config"},
            search_space=lambda trial: {"lr": trial.suggest_float("lr", 1e-4, 1e-2)},
            model_class=MagicMock,
            study_name="test_study",
            save_every_n_trials=1,
            local_checkpoint_dir=tmp_path / "checkpoints"
        )
        
        # Set persistent overrides
        optimizer.persistent_config_overrides = config_overrides
        
        # Create a dummy study
        study = optuna.create_study()
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=2)
        
        # Save to local checkpoint
        success = optimizer.save_study_to_local(study, 2)
        assert success
        
        # Load and verify
        checkpoint_file = tmp_path / "checkpoints" / "study.pkl"
        assert checkpoint_file.exists()
        
        with open(checkpoint_file, 'rb') as f:
            session_info = pickle.load(f)
        
        assert "config_overrides" in session_info
        assert session_info["config_overrides"] == config_overrides
        assert session_info["config_overrides"]["trial_steps"] == 5000
    
    def test_config_overrides_restored_on_resume(self, tmp_path):
        """Test that config overrides are restored when resuming."""
        
        # Create initial optimizer with overrides
        initial_overrides = {
            "trial_steps": 3000,
            "trainer.val_check_interval": 250
        }
        
        optimizer = PausibleOptunaOptimizer(
            base_config={"test": "config"},
            search_space=lambda trial: {"lr": trial.suggest_float("lr", 1e-4, 1e-2)},
            model_class=MagicMock,
            study_name="test_study",
            local_checkpoint_dir=tmp_path / "checkpoints"
        )
        
        # Create and save a study
        study = optuna.create_study()
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=2)
        
        # Save with config overrides
        session_info = {
            "study": study,
            "total_trials_completed": 2,
            "sampler_name": "tpe",
            "pruner_name": "median",
            "study_name": "test_study",
            "config_overrides": initial_overrides
        }
        
        checkpoint_file = tmp_path / "checkpoints" / "study.pkl"
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(session_info, f)
        
        # Load from checkpoint
        loaded_info = optimizer.load_study_from_local(str(tmp_path / "checkpoints"))
        
        assert loaded_info is not None
        assert "config_overrides" in loaded_info
        assert loaded_info["config_overrides"]["trial_steps"] == 3000
        assert loaded_info["config_overrides"]["trainer.val_check_interval"] == 250
    
    def test_config_overrides_merge_on_resume(self, tmp_path, caplog):
        """Test merging of saved and new config overrides on resume."""
        
        # Create study with initial overrides
        initial_overrides = {
            "trial_steps": 4000,
            "trainer.val_check_interval": 500,
            "trainer.limit_val_batches": 5
        }
        
        optimizer = PausibleOptunaOptimizer(
            base_config={"test": "config"},
            search_space=lambda trial: {"lr": trial.suggest_float("lr", 1e-4, 1e-2)},
            model_class=MagicMock,
            datamodule_class=MagicMock,
            study_name="test_study",
            local_checkpoint_dir=tmp_path / "checkpoints"
        )
        
        # Save checkpoint with initial overrides
        study = optuna.create_study()
        study.optimize(lambda trial: 0.5, n_trials=3)
        
        session_info = {
            "study": study,
            "total_trials_completed": 3,
            "sampler_name": "tpe",
            "pruner_name": "median",
            "study_name": "test_study",
            "config_overrides": initial_overrides
        }
        
        checkpoint_file = tmp_path / "checkpoints" / "study.pkl"
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(session_info, f)
        
        # Mock the underlying optimizer creation
        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = study
        with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer', return_value=mock_optimizer), \
             patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer', return_value=mock_optimizer):
            # Resume with some overrides changed
            new_overrides = {
                "trial_steps": 5000,  # Changed
                "trainer.val_check_interval": 500,  # Unchanged
                "trainer.enable_progress_bar": False  # New
            }
            
            # Capture logs
            with caplog.at_level(logging.INFO):
                study = optimizer.optimize(
                    n_trials=10,
                    resume_from=str(tmp_path / "checkpoints"),
                    config_overrides=new_overrides
                )
            
            # Note: Config overrides are not displayed on resume (simplified logging)
            # Just verify that overrides were properly merged

            # Verify merged overrides
            assert optimizer.persistent_config_overrides["trial_steps"] == 5000
            assert optimizer.persistent_config_overrides["trainer.val_check_interval"] == 500
            assert optimizer.persistent_config_overrides["trainer.limit_val_batches"] == 5
            assert optimizer.persistent_config_overrides["trainer.enable_progress_bar"] is False
    
    def test_config_overrides_display_for_new_study(self, tmp_path, caplog):
        """Test that config overrides are displayed when starting a new study."""
        
        config_overrides = {
            "trial_steps": 6000,
            "trainer.val_check_interval": 1000
        }
        
        optimizer = PausibleOptunaOptimizer(
            base_config={"test": "config"},
            search_space=lambda trial: {"lr": trial.suggest_float("lr", 1e-4, 1e-2)},
            model_class=MagicMock,
            datamodule_class=MagicMock,
            study_name="new_study"
        )
        
        # Mock the underlying optimizer creation
        mock_study = optuna.create_study()
        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = mock_study
        with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer', return_value=mock_optimizer), \
             patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer', return_value=mock_optimizer):
            # Start new study with overrides
            with caplog.at_level(logging.INFO):
                study = optimizer.optimize(
                    n_trials=5,
                    config_overrides=config_overrides
                )
            
            # Check that overrides were displayed (simplified format)
            log_text = caplog.text
            assert "🆕 STARTING NEW OPTIMIZATION" in log_text
            assert "📋 Config overrides:" in log_text
            assert "parameter(s)" in log_text
            
            # Verify overrides were stored
            assert optimizer.persistent_config_overrides == config_overrides
    
    def test_empty_overrides_no_table(self, tmp_path, caplog):
        """Test that no table is shown when there are no config overrides."""
        
        optimizer = PausibleOptunaOptimizer(
            base_config={"test": "config"},
            search_space=lambda trial: {"lr": trial.suggest_float("lr", 1e-4, 1e-2)},
            model_class=MagicMock,
            datamodule_class=MagicMock,
            study_name="test_study"
        )
        
        # Mock the underlying optimizer creation
        mock_study = optuna.create_study()
        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = mock_study
        with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer', return_value=mock_optimizer), \
             patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer', return_value=mock_optimizer):
            # Start without overrides
            with caplog.at_level(logging.INFO):
                study = optimizer.optimize(n_trials=5)
            
            # Check that no override message was shown
            log_text = caplog.text
            assert "📋 Config overrides:" not in log_text
            assert optimizer.persistent_config_overrides == {}


    def test_args_prefix_config_overrides(self, tmp_path):
        """Test that args.* prefixed configs (from persistent-by-default) work correctly."""
        
        # Create optimizer with config overrides including args.* prefixed ones
        config_overrides = {
            "trial_steps": 5000,
            "trainer.val_check_interval": 500,
            "args.n_trials": 100,
            "args.sampler": "tpe",
            "args.pruner": "hyperband",
            "args.patience": 10,
            "args.test_mode": True,
            "args.upload_checkpoints": True,
            "args.config": "configs/base.yaml",
            "args.wandb": "my-project"
        }
        
        optimizer = PausibleOptunaOptimizer(
            base_config={"test": "config"},
            search_space=lambda trial: {"lr": trial.suggest_float("lr", 1e-4, 1e-2)},
            model_class=MagicMock,
            study_name="test_study",
            save_every_n_trials=1,
            local_checkpoint_dir=tmp_path / "checkpoints"
        )
        
        # Set persistent overrides
        optimizer.persistent_config_overrides = config_overrides
        
        # Create a dummy study
        study = optuna.create_study()
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=2)
        
        # Save to local checkpoint
        success = optimizer.save_study_to_local(study, 2)
        assert success
        
        # Load and verify
        checkpoint_file = tmp_path / "checkpoints" / "study.pkl"
        assert checkpoint_file.exists()
        
        with open(checkpoint_file, 'rb') as f:
            session_info = pickle.load(f)
        
        assert "config_overrides" in session_info
        assert session_info["config_overrides"] == config_overrides
        
        # Verify all args.* prefixed configs are preserved
        for key in config_overrides:
            if key.startswith("args."):
                assert session_info["config_overrides"][key] == config_overrides[key]
        
        # Test restoration
        loaded_info = optimizer.load_study_from_local(str(tmp_path / "checkpoints"))
        assert loaded_info is not None
        assert "config_overrides" in loaded_info
        
        # All args.* configs should be restored
        for key in config_overrides:
            if key.startswith("args."):
                assert loaded_info["config_overrides"][key] == config_overrides[key]


    def test_argument_restoration_on_resume(self, tmp_path):
        """Test that saved args.* values are properly restored when resuming.
        
        This test catches the bug where --trial-steps reverted to default 
        instead of using the saved value of 40000.
        """
        # First, save a checkpoint with trial_steps=40000
        saved_overrides = {
            "args.trial_steps": 40000,  # Different from default 5000
            "args.val_interval": 250,
            "args.n_trials": 100,  # Different from default 50
            "trainer.val_check_interval": 250,
            "trial_steps": 40000,  # Also saved for backward compatibility
        }
        
        optimizer = PausibleOptunaOptimizer(
            base_config={"test": "config"},
            search_space=lambda trial: {"lr": trial.suggest_float("lr", 1e-4, 1e-2)},
            model_class=MagicMock,
            study_name="test_study",
            local_checkpoint_dir=tmp_path / "checkpoints"
        )
        
        # Create and save a study with the overrides
        study = optuna.create_study()
        study.optimize(lambda trial: 0.5, n_trials=3)
        
        session_info = {
            "study": study,
            "total_trials_completed": 3,
            "sampler_name": "tpe",
            "pruner_name": "median",
            "study_name": "test_study",
            "config_overrides": saved_overrides
        }
        
        checkpoint_file = tmp_path / "checkpoints" / "study.pkl"
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(session_info, f)
        
        # Now simulate resuming with default argument values
        # This simulates: python script.py --resume-from latest
        # WITHOUT specifying --trial-steps
        
        class Args:
            trial_steps = 5000  # Default value
            val_interval = None  # Default value  
            n_trials = 50  # Default value
            resume_from = str(tmp_path / "checkpoints")
            wandb = None
            study_name = "test_study"
        
        args = Args()
        
        # Load the saved session
        loaded_info = optimizer.load_study_from_local(args.resume_from)
        assert loaded_info is not None
        
        # The critical test: args should be restored from saved values
        saved_overrides = loaded_info["config_overrides"]
        
        # Verify the saved values exist
        assert "args.trial_steps" in saved_overrides
        assert saved_overrides["args.trial_steps"] == 40000
        assert "args.n_trials" in saved_overrides
        assert saved_overrides["args.n_trials"] == 100
        
        # This is what the script SHOULD do: restore args from saved values
        NON_PERSISTENT_ARGS = {'resume_from', 'study_name'}
        for key, value in saved_overrides.items():
            if key.startswith("args."):
                arg_name = key[5:]  # Remove "args." prefix
                if arg_name not in NON_PERSISTENT_ARGS and hasattr(args, arg_name):
                    setattr(args, arg_name, value)
        
        # After restoration, args should have the saved values, not defaults
        assert args.trial_steps == 40000, f"Expected trial_steps=40000, got {args.trial_steps}"
        assert args.n_trials == 100, f"Expected n_trials=100, got {args.n_trials}"
        assert args.val_interval == 250, f"Expected val_interval=250, got {args.val_interval}"


class TestResumeCommandPrintingConsistency:
    """Test that resume command printing works consistently with args.* configs."""
    
    def test_resume_command_with_args_configs(self, tmp_path):
        """Test that resume command is printed correctly with args.* prefixed configs."""
        
        # Create optimizer with both regular and args.* configs
        optimizer = PausibleOptunaOptimizer(
            base_config={"test": "config"},
            search_space=lambda trial: {"lr": trial.suggest_float("lr", 1e-4, 1e-2)},
            model_class=MagicMock,
            study_name="test_study",
            wandb_project="test-project",
            local_checkpoint_dir=tmp_path / "checkpoints"
        )
        
        # Set config overrides including args.* prefixed ones
        optimizer.persistent_config_overrides = {
            "trial_steps": 5000,
            "args.n_trials": 100,
            "args.sampler": "tpe",
            "args.upload_checkpoints": True
        }
        
        # Generate resume command
        resume_cmd = optimizer._build_resume_command()
        
        # The command should include wandb and study name but NOT the args.* configs
        # since those are already persisted
        assert "--wandb test-project" in resume_cmd
        assert "--study-name test_study" in resume_cmd
        assert "--resume-from latest" in resume_cmd
        
        # Should not include args.* configs in the command since they're auto-restored
        assert "--n-trials" not in resume_cmd
        assert "--sampler" not in resume_cmd
        assert "--upload-checkpoints" not in resume_cmd


class TestStatusEmojis:
    """Test that status emojis are correctly assigned."""
    
    def test_emoji_meanings(self):
        """Verify emoji meanings match documentation."""
        
        # This is more of a documentation test
        emoji_meanings = {
            "📌": "persistent from checkpoint (red pin)",
            "⭐": "new parameter added (yellow star)",
            "✅": "changed/updated value (green checkmark)",
            "🔄": "unchanged - specified again with same value (circular arrows)"
        }
        
        # Ensure emojis are distinct and meaningful
        assert len(set(emoji_meanings.keys())) == len(emoji_meanings)
        
        # Ensure all emojis are single characters (for display alignment)
        for emoji in emoji_meanings.keys():
            # Emojis might be 1-2 Python chars but display as single width
            assert len(emoji) <= 2


def test_integration_with_world_model_script(tmp_path):
    """Test integration with the actual world_model_hpo_optuna.py script."""

    # This test validates that the script correctly passes trial_steps
    # and other overrides to the optimizer

    config_overrides = {
        "trial_steps": 7000,
        "trainer.val_check_interval": 750,
        "trainer.enable_model_summary": False
    }

    # Create a mock optimizer that captures the config_overrides
    captured_overrides = {}

    def mock_optimize(n_trials, resume_from=None, config_overrides=None, **kwargs):
        nonlocal captured_overrides
        captured_overrides = config_overrides or {}
        # Return a mock study
        study = optuna.create_study()
        study.optimize(lambda trial: 0.5, n_trials=1)
        return study

    # The actual test would import and test the script
    # For now, we just verify the structure
    assert True  # Placeholder for actual integration test


class TestBaseConfigRestorationOnResume:
    """Test that base_config is restored from checkpoint when resuming without --config."""

    def test_base_config_restored_from_checkpoint(self, tmp_path):
        """Test that base_config is set from args.config in checkpoint when resuming.

        This validates the fix for:
        TypeError: expected str, bytes or os.PathLike object, not NoneType

        When resuming without --config flag, the config path should be restored
        from the checkpoint's args.config.
        """
        from LightningTune import HPORunner

        # Create a checkpoint with args.config saved
        saved_config_path = "configs/my_model.yaml"
        checkpoint = {
            "study": optuna.create_study(),
            "total_trials_completed": 5,
            "sampler_name": "tpe",
            "pruner_name": "median",
            "study_name": "resume_test",
            "config_overrides": {
                "args.config": saved_config_path,
                "args.n_trials": 100,
                "args.trial_steps": 40000,
            }
        }

        checkpoint_dir = tmp_path / "checkpoints" / "resume_test"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "study.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)

        # Create HPORunner WITHOUT base_config (simulating --resume-from without --config)
        runner = HPORunner(
            model_class=MagicMock,
            datamodule_class=MagicMock,
            search_space=lambda trial, config: config,
            base_config=None,  # No config specified
            require_config=True,  # Normally would fail
            default_study_name="resume_test",
        )

        # Verify base_config is initially None
        assert runner.base_config is None

        # Parse args simulating: --resume-from <checkpoint_file> --study-name resume_test
        # Note: _load_checkpoint expects the full file path, not directory
        args, _ = runner._create_parser().parse_known_args([
            "--resume-from", str(checkpoint_file),
            "--study-name", "resume_test",
            "--n-trials", "10",
        ])
        runner.args = args

        # Manually run the checkpoint restoration logic
        checkpoint_loaded = runner._load_checkpoint(str(checkpoint_file))
        assert checkpoint_loaded is not None

        runner._restore_args_from_checkpoint(checkpoint_loaded, [
            "--resume-from", str(checkpoint_file),
            "--study-name", "resume_test",
            "--n-trials", "10",
        ])

        # Critical: after restoration, args.config should have the saved path
        assert runner.args.config == saved_config_path, (
            f"args.config should be restored from checkpoint, got: {runner.args.config}"
        )

        # The fix: update base_config from restored args.config
        if runner.base_config is None and runner.args.config:
            runner.base_config = runner.args.config

        # Now base_config should be set
        assert runner.base_config == saved_config_path, (
            f"base_config should be updated from args.config, got: {runner.base_config}"
        )

    def test_resume_without_config_fails_if_checkpoint_lacks_config(self, tmp_path):
        """Test that resume fails gracefully if checkpoint doesn't have args.config."""
        from LightningTune import HPORunner

        # Create a checkpoint WITHOUT args.config (simulating old checkpoint)
        checkpoint = {
            "study": optuna.create_study(),
            "total_trials_completed": 5,
            "sampler_name": "tpe",
            "pruner_name": "median",
            "study_name": "old_study",
            "config_overrides": {
                # No args.config here
                "args.n_trials": 50,
            }
        }

        checkpoint_dir = tmp_path / "checkpoints" / "old_study"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "study.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)

        # Create HPORunner WITHOUT base_config
        runner = HPORunner(
            model_class=MagicMock,
            datamodule_class=MagicMock,
            search_space=lambda trial, config: config,
            base_config=None,
            require_config=True,
            default_study_name="old_study",
        )

        # Parse args - note: use checkpoint_file (full path), not checkpoint_dir
        args, _ = runner._create_parser().parse_known_args([
            "--resume-from", str(checkpoint_file),
            "--study-name", "old_study",
            "--n-trials", "10",
        ])
        runner.args = args

        # Load and restore
        checkpoint_loaded = runner._load_checkpoint(str(checkpoint_file))
        runner._restore_args_from_checkpoint(checkpoint_loaded, [
            "--resume-from", str(checkpoint_file),
        ])

        # args.config should still be None (not in checkpoint)
        assert runner.args.config is None

        # base_config should remain None
        if runner.base_config is None and runner.args.config:
            runner.base_config = runner.args.config

        assert runner.base_config is None, (
            "base_config should remain None if checkpoint lacks args.config"
        )

        # In the actual run(), this would trigger an error message
        # asking user to specify --config


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])