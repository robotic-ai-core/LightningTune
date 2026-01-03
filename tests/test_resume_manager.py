"""
Tests for ResumeManager.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import optuna


class TestResumeManager:
    """Tests for ResumeManager class."""

    def test_initialization(self, tmp_path):
        """Test basic initialization."""
        from LightningTune.resume import ResumeManager

        manager = ResumeManager(
            local_checkpoint_dir=tmp_path,
            wandb_project="test-project",
            study_name="test_study",
        )

        assert manager.local_checkpoint_dir == tmp_path
        assert manager.wandb_project == "test-project"
        assert manager.study_name == "test_study"
        assert manager.persistence is not None

    def test_initialization_local_only(self, tmp_path):
        """Test initialization with local only."""
        from LightningTune.resume import ResumeManager
        from LightningTune.persistence import LocalPersistence

        manager = ResumeManager(local_checkpoint_dir=tmp_path)

        assert isinstance(manager.persistence, LocalPersistence)

    def test_initialization_no_persistence(self):
        """Test initialization without persistence."""
        from LightningTune.resume import ResumeManager

        manager = ResumeManager()

        assert manager.persistence is None

    def test_save_checkpoint(self, tmp_path):
        """Test saving a checkpoint."""
        from LightningTune.resume import ResumeManager

        manager = ResumeManager(
            local_checkpoint_dir=tmp_path / "checkpoints",
            study_name="save_test",
        )

        study = optuna.create_study()
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=3)

        result = manager.save_checkpoint(
            study=study,
            total_trials_completed=3,
            sampler_name="tpe",
            pruner_name="median",
            config_overrides={"key": "value"},
        )

        assert result is True
        assert (tmp_path / "checkpoints" / "study.pkl").exists()

    def test_load_checkpoint(self, tmp_path):
        """Test loading a checkpoint."""
        from LightningTune.resume import ResumeManager

        checkpoint_dir = tmp_path / "checkpoints"
        manager = ResumeManager(
            local_checkpoint_dir=checkpoint_dir,
            study_name="load_test",
        )

        # Save first
        study = optuna.create_study()
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=2)
        manager.save_checkpoint(
            study=study,
            total_trials_completed=2,
            sampler_name="tpe",
            pruner_name="median",
        )

        # Load
        session = manager.load_checkpoint("latest")

        assert session is not None
        assert session["study_name"] == "load_test"
        assert session["total_trials_completed"] == 2

    def test_restore_args(self, tmp_path):
        """Test restoring arguments from checkpoint."""
        from LightningTune.resume import ResumeManager

        manager = ResumeManager(
            local_checkpoint_dir=tmp_path,
            persist_args=True,
        )

        # Create mock args object
        class Args:
            n_trials = 10
            save_every = 5

        args = Args()

        checkpoint = {
            "config_overrides": {
                "args.n_trials": 100,
                "args.save_every": 20,
            },
        }

        # Restore with no CLI args specified
        with patch('sys.argv', ['script.py']):
            restored, overridden = manager.restore_args(args, checkpoint)

        assert restored == 2
        assert args.n_trials == 100
        assert args.save_every == 20

    def test_restore_args_with_cli_override(self, tmp_path):
        """Test that CLI args override saved args."""
        from LightningTune.resume import ResumeManager

        manager = ResumeManager(persist_args=True)

        class Args:
            n_trials = 50

        args = Args()

        checkpoint = {
            "config_overrides": {
                "args.n_trials": 100,
            },
        }

        # CLI specifies n_trials
        with patch('sys.argv', ['script.py', '--n-trials', '50']):
            restored, overridden = manager.restore_args(args, checkpoint)

        # Should not restore n_trials since it was specified on CLI
        assert args.n_trials == 50
        assert overridden == 1

    def test_handle_n_trials_extension(self, tmp_path):
        """Test n_trials extension logic."""
        from LightningTune.resume import ResumeManager

        manager = ResumeManager()

        checkpoint = {
            "config_overrides": {"args.n_trials": 50},
        }

        # Extend: current > saved
        with patch('sys.argv', ['script.py']):
            final, extended = manager.handle_n_trials_extension(100, checkpoint)
        assert final == 100
        assert extended is True

        # Use saved: current < saved, not CLI specified
        with patch('sys.argv', ['script.py']):
            final, extended = manager.handle_n_trials_extension(30, checkpoint)
        assert final == 50
        assert extended is False

        # Respect CLI: current < saved, but CLI specified
        with patch('sys.argv', ['script.py', '--n-trials', '30']):
            final, extended = manager.handle_n_trials_extension(30, checkpoint)
        assert final == 30
        assert extended is False

    def test_build_resume_command(self, tmp_path):
        """Test building resume command."""
        from LightningTune.resume import ResumeManager

        checkpoint_dir = tmp_path / "checkpoints"
        manager = ResumeManager(
            local_checkpoint_dir=checkpoint_dir,
            wandb_project="my-project",
            study_name="my_study",
        )

        cmd = manager.build_resume_command(
            original_argv=['script.py', '--trial-steps', '1000'],
            default_script='hpo.py',
            use_local=True,
        )

        assert 'python' in cmd
        assert 'script.py' in cmd
        assert '--wandb' in cmd
        assert 'my-project' in cmd
        assert '--study-name' in cmd
        assert 'my_study' in cmd
        assert '--trial-steps' in cmd
        assert '1000' in cmd
        assert '--resume-from' in cmd
        assert str(checkpoint_dir) in cmd

    def test_build_wandb_resume_command(self, tmp_path):
        """Test building WandB resume command."""
        from LightningTune.resume import ResumeManager

        manager = ResumeManager(
            local_checkpoint_dir=tmp_path,
            wandb_project="my-project",
            study_name="my_study",
        )

        cmd = manager.build_wandb_resume_command()

        assert '--resume-from' in cmd
        assert 'latest' in cmd

    def test_get_trials_since_last_upload(self):
        """Test calculating trials since last upload."""
        from LightningTune.resume import ResumeManager

        manager = ResumeManager()

        # No checkpoint - all trials are new
        assert manager.get_trials_since_last_upload(10, None) == 10

        # With checkpoint
        checkpoint = {"last_wandb_upload_trial_count": 5}
        assert manager.get_trials_since_last_upload(10, checkpoint) == 5

        # No last_wandb_upload_trial_count in checkpoint
        checkpoint = {}
        assert manager.get_trials_since_last_upload(10, checkpoint) == 10

    def test_should_upload_to_wandb(self):
        """Test upload decision logic."""
        from LightningTune.resume import ResumeManager

        # No WandB project - never upload
        manager = ResumeManager()
        assert manager.should_upload_to_wandb(10, 5) is False

        # With WandB project
        manager = ResumeManager(
            wandb_project="test",
            study_name="test",
        )

        # Not enough trials
        checkpoint = {"last_wandb_upload_trial_count": 5}
        assert manager.should_upload_to_wandb(8, 5, checkpoint) is False

        # Enough trials
        assert manager.should_upload_to_wandb(10, 5, checkpoint) is True


class TestCreateResumeManagerFromArgs:
    """Tests for create_resume_manager_from_args helper."""

    def test_basic_creation(self, tmp_path):
        """Test creating ResumeManager from args."""
        from LightningTune.resume import create_resume_manager_from_args

        class Args:
            wandb = "my-project"
            study_name = "my_study"
            persist_args = True

        args = Args()
        manager = create_resume_manager_from_args(args, default_checkpoint_dir=tmp_path)

        assert manager.wandb_project == "my-project"
        assert manager.study_name == "my_study"
        assert manager.local_checkpoint_dir == tmp_path
        assert manager.persist_args is True

    def test_with_checkpoint_dir_in_args(self, tmp_path):
        """Test using checkpoint_dir from args."""
        from LightningTune.resume import create_resume_manager_from_args

        class Args:
            checkpoint_dir = tmp_path / "custom"
            wandb = None
            study_name = None
            persist_args = True

        args = Args()
        manager = create_resume_manager_from_args(args)

        assert manager.local_checkpoint_dir == tmp_path / "custom"


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
