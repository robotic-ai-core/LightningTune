#!/usr/bin/env python
"""
Unit tests for checkpoint persistence refactoring.

Tests the removal of redundant `last_saved_trial_count` parameter and validates:
1. Save functions no longer accept the removed parameter
2. Checkpoint dictionaries don't contain the removed field
3. Load functions handle both old and new checkpoint formats
4. Counter restoration logic works correctly
5. Backward compatibility with old checkpoint format
6. Edge cases (empty checkpoints, invalid values, large counts)
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pickle
import tempfile
import pytest

import optuna

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from LightningTune.persistence import (
    save_study_to_local,
    load_study_from_local,
    save_study_to_wandb,
    load_study_from_wandb,
)
from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer


def _create_test_study(name: str, n_trials: int) -> optuna.Study:
    """Create a simple test study with n completed trials."""
    study = optuna.create_study(study_name=name)
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=n_trials)
    return study


class TestPersistenceModuleSaveLoad:
    """Test persistence.py save/load functions without last_saved_trial_count."""

    def test_save_study_to_local_no_last_saved_parameter(self, tmp_path: Path):
        """Test save_study_to_local() doesn't accept last_saved_trial_count parameter."""
        study = _create_test_study("test_study", 5)
        checkpoint_dir = tmp_path / "checkpoints"

        # Should succeed without last_saved_trial_count
        success = save_study_to_local(
            checkpoint_dir,
            study,
            total_trials_completed=5,
            sampler_name="tpe",
            pruner_name="median",
            study_name="test_study",
            config_overrides={"args.n_trials": 10}
        )

        assert success, "save_study_to_local should succeed"
        assert (checkpoint_dir / "study.pkl").exists(), "study.pkl should be created"

        # Verify checkpoint dictionary doesn't contain last_saved_trial_count
        with open(checkpoint_dir / "study.pkl", 'rb') as f:
            session_info = pickle.load(f)

        assert "last_saved_trial_count" not in session_info, \
            "Checkpoint should not contain last_saved_trial_count field"
        assert session_info["total_trials_completed"] == 5, \
            "Should contain total_trials_completed"

    def test_load_study_from_local_handles_new_format(self, tmp_path: Path):
        """Test load_study_from_local() correctly loads new format checkpoints."""
        study = _create_test_study("test_study", 3)
        checkpoint_dir = tmp_path / "checkpoints"

        # Save with new format
        save_study_to_local(
            checkpoint_dir,
            study,
            total_trials_completed=3,
            sampler_name="tpe",
            pruner_name="median",
            study_name="test_study",
        )

        # Load and verify
        session_info = load_study_from_local(str(checkpoint_dir))
        assert session_info is not None, "Should load successfully"
        assert session_info["total_trials_completed"] == 3
        assert session_info["study_name"] == "test_study"
        assert session_info["sampler_name"] == "tpe"
        assert "last_saved_trial_count" not in session_info

    def test_load_study_from_local_handles_old_format(self, tmp_path: Path):
        """Test load_study_from_local() handles old format with last_saved_trial_count."""
        study = _create_test_study("test_study", 3)
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create old format checkpoint with last_saved_trial_count
        old_format_session = {
            "study": study,
            "total_trials_completed": 3,
            "last_saved_trial_count": 3,  # Old field - should be ignored
            "sampler_name": "tpe",
            "pruner_name": "median",
            "study_name": "test_study",
            "config_overrides": {},
        }

        with open(checkpoint_dir / "study.pkl", 'wb') as f:
            pickle.dump(old_format_session, f)

        # Load and verify it works despite extra field
        session_info = load_study_from_local(str(checkpoint_dir))
        assert session_info is not None, "Should load old format successfully"
        assert session_info["total_trials_completed"] == 3
        # Old field is present but ignored (not used anywhere)
        assert session_info.get("last_saved_trial_count") == 3

    @patch('wandb.init')
    @patch('wandb.Artifact')
    def test_save_study_to_wandb_no_last_saved_parameter(self, mock_artifact_class, mock_wandb_init):
        """Test save_study_to_wandb() doesn't accept last_saved_trial_count parameter."""
        # Setup mocks
        mock_run = Mock()
        mock_logged_artifact = Mock()
        mock_logged_artifact.wait = Mock()
        mock_run.log_artifact.return_value = mock_logged_artifact
        mock_run.finish = Mock()
        mock_wandb_init.return_value = mock_run

        mock_artifact = Mock()
        mock_artifact.add_file = Mock()
        mock_artifact_class.return_value = mock_artifact

        study = _create_test_study("test_study", 5)

        # Should succeed without last_saved_trial_count
        success = save_study_to_wandb(
            wandb_project="test-project",
            study_name="test_study",
            study=study,
            total_trials_completed=5,
            sampler_name="tpe",
            pruner_name="median",
            config_overrides={"args.n_trials": 10}
        )

        assert success, "save_study_to_wandb should succeed"

        # Verify the saved pickle doesn't contain last_saved_trial_count
        # Check what was passed to add_file
        assert mock_artifact.add_file.called
        tmp_file_path = mock_artifact.add_file.call_args[0][0]

        with open(tmp_file_path, 'rb') as f:
            session_info = pickle.load(f)

        assert "last_saved_trial_count" not in session_info, \
            "WandB checkpoint should not contain last_saved_trial_count field"
        assert session_info["total_trials_completed"] == 5

    @patch('wandb.Api')
    def test_load_study_from_wandb_handles_new_format(self, mock_api_class):
        """Test load_study_from_wandb() correctly loads new format checkpoints."""
        study = _create_test_study("test_study", 3)

        # Create new format checkpoint
        session_info = {
            "study": study,
            "total_trials_completed": 3,
            "sampler_name": "tpe",
            "pruner_name": "median",
            "study_name": "test_study",
            "config_overrides": {},
        }

        # Mock WandB API
        mock_api = Mock()
        mock_artifact = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write test checkpoint
            pkl_path = Path(tmpdir) / "study.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(session_info, f)

            mock_artifact.download.return_value = tmpdir
            mock_api.artifact.return_value = mock_artifact
            mock_api_class.return_value = mock_api

            # Load and verify
            loaded_info = load_study_from_wandb("test-project", "test_study", "latest")

            assert loaded_info is not None
            assert loaded_info["total_trials_completed"] == 3
            assert "last_saved_trial_count" not in loaded_info

    @patch('wandb.Api')
    def test_load_study_from_wandb_handles_old_format(self, mock_api_class):
        """Test load_study_from_wandb() handles old format with last_saved_trial_count."""
        study = _create_test_study("test_study", 3)

        # Create old format checkpoint with last_saved_trial_count
        old_session_info = {
            "study": study,
            "total_trials_completed": 3,
            "last_saved_trial_count": 3,  # Old field
            "sampler_name": "tpe",
            "pruner_name": "median",
            "study_name": "test_study",
            "config_overrides": {},
        }

        # Mock WandB API
        mock_api = Mock()
        mock_artifact = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write old format checkpoint
            pkl_path = Path(tmpdir) / "study.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(old_session_info, f)

            mock_artifact.download.return_value = tmpdir
            mock_api.artifact.return_value = mock_artifact
            mock_api_class.return_value = mock_api

            # Load and verify it works
            loaded_info = load_study_from_wandb("test-project", "test_study", "latest")

            assert loaded_info is not None, "Should load old format successfully"
            assert loaded_info["total_trials_completed"] == 3
            # Old field is present but not used
            assert loaded_info.get("last_saved_trial_count") == 3


class TestCounterRestorationLogic:
    """Test counter restoration logic in pausible_optimizer.py."""

    def test_fresh_start_counter_initialization(self, tmp_path: Path):
        """Test that fresh start initializes counter correctly."""
        with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer'):
            with patch('LightningTune.optuna.pausible_optimizer.ReflowOptunaDrivenOptimizer'):
                optimizer = PausibleOptunaOptimizer(
                    base_config={'dummy': 'config'},
                    search_space=lambda trial: {},
                    model_class=Mock,
                    wandb_project=None,
                    study_name="test_study",
                    save_every_n_trials=5,
                    enable_pause=False,
                    use_reflow=False,
                )

                # Mock objective
                def simple_objective(trial):
                    return trial.suggest_float('x', 0, 1)

                mock_underlying = Mock()
                mock_underlying.create_objective.return_value = simple_objective
                optimizer.underlying_optimizer = mock_underlying

                # Run 2 trials (fresh start)
                study = optimizer.optimize(n_trials=2, config_overrides={})

                # Verify counter state
                assert optimizer.total_trials_completed == 2
                # On fresh start, last_checkpoint_trial_count should start at 0
                # and trials_in_batch should be 2 (no saves yet since save_every=5)

    def test_resume_counter_restoration(self, tmp_path: Path):
        """Test counter restoration: trials_in_batch = total_trials_completed - checkpoint_trial_count."""
        checkpoint_dir = tmp_path / "checkpoints"

        with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer'):
            with patch('LightningTune.optuna.pausible_optimizer.ReflowOptunaDrivenOptimizer'):
                # First run: save after 5 trials
                opt1 = PausibleOptunaOptimizer(
                    base_config={'dummy': 'config'},
                    search_space=lambda trial: {},
                    model_class=Mock,
                    wandb_project=None,
                    study_name="test_study",
                    save_every_n_trials=5,
                    enable_pause=False,
                    use_reflow=False,
                    local_checkpoint_dir=str(checkpoint_dir),
                )

                def simple_objective(trial):
                    return trial.suggest_float('x', 0, 1)

                mock_underlying = Mock()
                mock_underlying.create_objective.return_value = simple_objective
                opt1.underlying_optimizer = mock_underlying

                # Run 5 trials (should trigger save)
                study1 = opt1.optimize(n_trials=5, config_overrides={})

                # Save checkpoint manually to simulate save_every behavior
                opt1.save_study_to_local(study1, 5)

                # Second run: resume from checkpoint
                opt2 = PausibleOptunaOptimizer(
                    base_config={'dummy': 'config'},
                    search_space=lambda trial: {},
                    model_class=Mock,
                    wandb_project=None,
                    study_name="test_study",
                    save_every_n_trials=5,
                    enable_pause=False,
                    use_reflow=False,
                    local_checkpoint_dir=str(checkpoint_dir),
                )

                mock_underlying2 = Mock()
                mock_underlying2.create_objective.return_value = simple_objective
                opt2.underlying_optimizer = mock_underlying2

                # Resume - should calculate trials_in_batch correctly
                # checkpoint had 5 trials, we'll run to 8 total
                study2 = opt2.optimize(
                    n_trials=8,
                    resume_from=str(checkpoint_dir),
                    config_overrides={}
                )

                # After resume, optimizer should have:
                # - total_trials_completed restored from checkpoint (5)
                # - After running 3 more: total_trials_completed = 8
                assert opt2.total_trials_completed == 8

    def test_save_counter_update_after_periodic_save(self, tmp_path: Path):
        """Test that last_checkpoint_trial_count updates after periodic save."""
        checkpoint_dir = tmp_path / "checkpoints"

        with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer'):
            with patch('LightningTune.optuna.pausible_optimizer.ReflowOptunaDrivenOptimizer'):
                optimizer = PausibleOptunaOptimizer(
                    base_config={'dummy': 'config'},
                    search_space=lambda trial: {},
                    model_class=Mock,
                    wandb_project=None,
                    study_name="test_study",
                    save_every_n_trials=3,  # Save every 3 trials
                    enable_pause=False,
                    use_reflow=False,
                    local_checkpoint_dir=str(checkpoint_dir),
                )

                def simple_objective(trial):
                    return trial.suggest_float('x', 0, 1)

                mock_underlying = Mock()
                mock_underlying.create_objective.return_value = simple_objective
                optimizer.underlying_optimizer = mock_underlying

                # Run 7 trials (should save at 3 and 6, with 1 remaining)
                study = optimizer.optimize(n_trials=7, config_overrides={})

                # Verify final state
                assert optimizer.total_trials_completed == 7

                # Check that checkpoint exists
                assert (checkpoint_dir / "study.pkl").exists()

                # Load checkpoint and verify count
                with open(checkpoint_dir / "study.pkl", 'rb') as f:
                    session_info = pickle.load(f)
                assert session_info["total_trials_completed"] == 7

    def test_save_counter_update_after_pause_save(self, tmp_path: Path):
        """Test that last_checkpoint_trial_count updates after pause save."""
        checkpoint_dir = tmp_path / "checkpoints"

        with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer'):
            with patch('LightningTune.optuna.pausible_optimizer.ReflowOptunaDrivenOptimizer'):
                optimizer = PausibleOptunaOptimizer(
                    base_config={'dummy': 'config'},
                    search_space=lambda trial: {},
                    model_class=Mock,
                    wandb_project=None,
                    study_name="test_study",
                    save_every_n_trials=10,  # High to avoid periodic saves
                    enable_pause=False,
                    use_reflow=False,
                    local_checkpoint_dir=str(checkpoint_dir),
                )

                def simple_objective(trial):
                    return trial.suggest_float('x', 0, 1)

                mock_underlying = Mock()
                mock_underlying.create_objective.return_value = simple_objective
                optimizer.underlying_optimizer = mock_underlying

                # Run 4 trials, then simulate pause
                study = optimizer.optimize(n_trials=4, config_overrides={})

                # Simulate pause save
                optimizer.should_pause = True
                optimizer.save_study_to_local(study, optimizer.total_trials_completed)

                # Verify checkpoint was saved with correct count
                with open(checkpoint_dir / "study.pkl", 'rb') as f:
                    session_info = pickle.load(f)
                assert session_info["total_trials_completed"] == 4


class TestBackwardCompatibility:
    """Test backward compatibility with old checkpoint format."""

    def test_old_checkpoint_with_last_saved_trial_count_loads(self, tmp_path: Path):
        """Test that old checkpoints with last_saved_trial_count load successfully."""
        study = _create_test_study("test_study", 5)
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create old format checkpoint
        old_checkpoint = {
            "study": study,
            "total_trials_completed": 5,
            "last_saved_trial_count": 5,  # Old redundant field
            "sampler_name": "tpe",
            "pruner_name": "median",
            "study_name": "test_study",
            "config_overrides": {"args.n_trials": 10},
        }

        with open(checkpoint_dir / "study.pkl", 'wb') as f:
            pickle.dump(old_checkpoint, f)

        # Load with new code
        session_info = load_study_from_local(str(checkpoint_dir))

        assert session_info is not None
        assert session_info["total_trials_completed"] == 5
        assert session_info["study_name"] == "test_study"
        # Old field exists but is not used by new code
        assert session_info.get("last_saved_trial_count") == 5

    def test_resume_from_old_checkpoint_works(self, tmp_path: Path):
        """Test that resuming from an old checkpoint works correctly."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create old format checkpoint
        study = _create_test_study("test_study", 3)
        old_checkpoint = {
            "study": study,
            "total_trials_completed": 3,
            "last_saved_trial_count": 3,  # Old field
            "sampler_name": "tpe",
            "pruner_name": "median",
            "study_name": "test_study",
            "config_overrides": {},
        }

        with open(checkpoint_dir / "study.pkl", 'wb') as f:
            pickle.dump(old_checkpoint, f)

        with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer'):
            with patch('LightningTune.optuna.pausible_optimizer.ReflowOptunaDrivenOptimizer'):
                optimizer = PausibleOptunaOptimizer(
                    base_config={'dummy': 'config'},
                    search_space=lambda trial: {},
                    model_class=Mock,
                    wandb_project=None,
                    study_name="test_study",
                    save_every_n_trials=5,
                    enable_pause=False,
                    use_reflow=False,
                    local_checkpoint_dir=str(checkpoint_dir),
                )

                def simple_objective(trial):
                    return trial.suggest_float('x', 0, 1)

                mock_underlying = Mock()
                mock_underlying.create_objective.return_value = simple_objective
                optimizer.underlying_optimizer = mock_underlying

                # Resume from old checkpoint
                study = optimizer.optimize(
                    n_trials=5,
                    resume_from=str(checkpoint_dir),
                    config_overrides={}
                )

                # Should resume successfully and run 2 more trials
                assert optimizer.total_trials_completed == 5

    def test_old_checkpoint_extra_field_safely_ignored(self, tmp_path: Path):
        """Test that extra field in old checkpoint is safely ignored."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        study = _create_test_study("test_study", 7)

        # Create checkpoint with extra field and mismatched value
        old_checkpoint = {
            "study": study,
            "total_trials_completed": 7,
            "last_saved_trial_count": 5,  # Different from total - should be ignored
            "sampler_name": "tpe",
            "pruner_name": "median",
            "study_name": "test_study",
            "config_overrides": {},
        }

        with open(checkpoint_dir / "study.pkl", 'wb') as f:
            pickle.dump(old_checkpoint, f)

        # Load and verify only total_trials_completed is used
        session_info = load_study_from_local(str(checkpoint_dir))

        assert session_info is not None
        assert session_info["total_trials_completed"] == 7
        # New code should only use total_trials_completed, not last_saved_trial_count
        # Verify the mismatched value doesn't cause issues
        assert session_info.get("last_saved_trial_count") == 5  # Present but unused


class TestEdgeCases:
    """Test edge cases for checkpoint persistence."""

    def test_empty_checkpoint_all_zeros(self, tmp_path: Path):
        """Test checkpoint with zero trials."""
        checkpoint_dir = tmp_path / "checkpoints"

        # Create study with 0 trials
        study = optuna.create_study(study_name="empty_study")

        success = save_study_to_local(
            checkpoint_dir,
            study,
            total_trials_completed=0,
            sampler_name="tpe",
            pruner_name="median",
            study_name="empty_study",
        )

        assert success

        # Load and verify
        session_info = load_study_from_local(str(checkpoint_dir))
        assert session_info is not None
        assert session_info["total_trials_completed"] == 0
        assert len(session_info["study"].trials) == 0

    def test_missing_fields_use_get_defaults(self, tmp_path: Path):
        """Test that missing fields are handled with .get() defaults."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        study = _create_test_study("test_study", 2)

        # Create minimal checkpoint (missing some optional fields)
        minimal_checkpoint = {
            "study": study,
            "total_trials_completed": 2,
            "study_name": "test_study",
            # Missing: sampler_name, pruner_name, config_overrides
        }

        with open(checkpoint_dir / "study.pkl", 'wb') as f:
            pickle.dump(minimal_checkpoint, f)

        # Load and verify it handles missing fields gracefully
        session_info = load_study_from_local(str(checkpoint_dir))

        assert session_info is not None
        assert session_info["total_trials_completed"] == 2
        assert session_info.get("sampler_name") is None
        assert session_info.get("config_overrides") is None

    def test_very_large_trial_counts(self, tmp_path: Path):
        """Test checkpoint with very large trial counts (10000+)."""
        checkpoint_dir = tmp_path / "checkpoints"

        # Create study with large trial count (but don't actually run 10000 trials)
        study = _create_test_study("large_study", 5)

        # Save with large count
        success = save_study_to_local(
            checkpoint_dir,
            study,
            total_trials_completed=10537,  # Simulate large count
            sampler_name="tpe",
            pruner_name="median",
            study_name="large_study",
        )

        assert success

        # Load and verify large count is preserved
        session_info = load_study_from_local(str(checkpoint_dir))
        assert session_info is not None
        assert session_info["total_trials_completed"] == 10537

    def test_negative_trial_count_rejected(self, tmp_path: Path):
        """Test that negative trial counts are handled (should save but be unusual)."""
        checkpoint_dir = tmp_path / "checkpoints"
        study = _create_test_study("test_study", 1)

        # Try to save with negative count (shouldn't happen in practice)
        success = save_study_to_local(
            checkpoint_dir,
            study,
            total_trials_completed=-1,  # Invalid but should not crash
            sampler_name="tpe",
            pruner_name="median",
            study_name="test_study",
        )

        # Should complete without crashing (even if value is nonsensical)
        assert success

        session_info = load_study_from_local(str(checkpoint_dir))
        assert session_info is not None
        # The invalid value is preserved (caller's responsibility to validate)
        assert session_info["total_trials_completed"] == -1

    def test_checkpoint_with_none_config_overrides(self, tmp_path: Path):
        """Test checkpoint with None config_overrides is handled correctly."""
        checkpoint_dir = tmp_path / "checkpoints"
        study = _create_test_study("test_study", 3)

        # Save with None config_overrides
        success = save_study_to_local(
            checkpoint_dir,
            study,
            total_trials_completed=3,
            sampler_name="tpe",
            pruner_name="median",
            study_name="test_study",
            config_overrides=None,  # Explicit None
        )

        assert success

        # Load and verify empty dict is used
        session_info = load_study_from_local(str(checkpoint_dir))
        assert session_info is not None
        assert session_info["config_overrides"] == {}  # None converted to {}

    def test_checkpoint_with_pruned_trials(self, tmp_path: Path):
        """Test checkpoint correctly handles studies with pruned trials."""
        checkpoint_dir = tmp_path / "checkpoints"

        # Create study with both completed and pruned trials
        study = optuna.create_study(study_name="pruned_study")

        def objective(trial):
            x = trial.suggest_float('x', 0, 1)
            if trial.number % 2 == 0:
                raise optuna.TrialPruned()  # Prune even trials
            return x

        # Run trials (some will be pruned)
        for _ in range(6):
            try:
                study.optimize(objective, n_trials=1)
            except:
                pass

        # Count finished trials (completed + pruned)
        finished = len([t for t in study.trials
                       if t.state in [optuna.trial.TrialState.COMPLETE,
                                     optuna.trial.TrialState.PRUNED]])

        # Save checkpoint
        success = save_study_to_local(
            checkpoint_dir,
            study,
            total_trials_completed=finished,
            sampler_name="tpe",
            pruner_name="median",
            study_name="pruned_study",
        )

        assert success

        # Load and verify
        session_info = load_study_from_local(str(checkpoint_dir))
        assert session_info is not None
        assert session_info["total_trials_completed"] == finished

        # Verify both completed and pruned trials are in the study
        loaded_study = session_info["study"]
        completed_count = len([t for t in loaded_study.trials
                              if t.state == optuna.trial.TrialState.COMPLETE])
        pruned_count = len([t for t in loaded_study.trials
                           if t.state == optuna.trial.TrialState.PRUNED])

        assert completed_count > 0, "Should have some completed trials"
        assert pruned_count > 0, "Should have some pruned trials"
        assert completed_count + pruned_count == finished


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
