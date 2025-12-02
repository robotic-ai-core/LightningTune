"""
Test that --upload-every (save_every_n_trials) correctly limits WandB artifact uploads.

This test validates that WandB artifacts are only uploaded when the trial count reaches
the configured threshold, NOT after every trial.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import pytest
import optuna
import tempfile

# Add LightningTune root to path
lightningtune_root = Path(__file__).parent.parent
sys.path.insert(0, str(lightningtune_root))

from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer


class TestUploadEveryNTrials:
    """Test that WandB uploads respect save_every_n_trials setting."""

    def test_wandb_uploads_only_at_threshold(self):
        """
        Test that persist_save_study_to_wandb is called only when trials_in_batch >= save_every_n_trials.

        With save_every_n_trials=3 and 10 trials:
        - Expected WandB uploads at: trial 3, 6, 9, and final save at 10
        - Total: 4 uploads (NOT 10)
        """
        wandb_save_calls = []
        local_save_calls = []

        def track_wandb_save(wandb_project, *, study_name=None, study=None, total_trials_completed=None, **kwargs):
            """Track when WandB saves happen."""
            wandb_save_calls.append(total_trials_completed)
            return True

        def track_local_save(checkpoint_dir, study, total_trials_completed, **kwargs):
            """Track when local saves happen."""
            local_save_calls.append(total_trials_completed)
            return True

        # Create optimizer with save_every_n_trials=3
        optimizer = PausibleOptunaOptimizer(
            base_config={'dummy': 'config'},
            search_space=lambda trial: {'x': trial.suggest_float('x', 0, 1)},
            model_class=Mock,
            datamodule_class=Mock,
            wandb_project="test-project",
            save_every_n_trials=3,  # Upload to WandB every 3 trials
            enable_pause=False,
            use_reflow=False,
            restart_on_save=False,  # Don't restart to see full trial sequence
        )

        with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb', side_effect=track_wandb_save):
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local', side_effect=track_local_save):
                # Run 10 trials
                study = optimizer.optimize(
                    n_trials=10,
                    config_overrides={},
                    callbacks=[]
                )

        # Local saves happen every trial PLUS a final save at end (10 + 1 = 11)
        # Note: The final local save is intentional to ensure checkpoint is always current
        assert len(local_save_calls) >= 10, f"Expected at least 10 local saves, got {len(local_save_calls)}: {local_save_calls}"

        # WandB uploads should only happen at thresholds: 3, 6, 9, and final save at 10
        # That's 4 uploads total, NOT 10
        expected_wandb_saves = [3, 6, 9, 10]
        assert wandb_save_calls == expected_wandb_saves, \
            f"Expected WandB uploads at {expected_wandb_saves}, got {wandb_save_calls}"

    def test_wandb_uploads_with_save_every_5(self):
        """Test with save_every_n_trials=5 and 12 trials."""
        wandb_save_calls = []

        def track_wandb_save(wandb_project, *, study_name=None, study=None, total_trials_completed=None, **kwargs):
            wandb_save_calls.append(total_trials_completed)
            return True

        def track_local_save(*args, **kwargs):
            return True

        optimizer = PausibleOptunaOptimizer(
            base_config={'dummy': 'config'},
            search_space=lambda trial: {'x': trial.suggest_float('x', 0, 1)},
            model_class=Mock,
            datamodule_class=Mock,
            wandb_project="test-project",
            save_every_n_trials=5,
            enable_pause=False,
            use_reflow=False,
            restart_on_save=False,
        )

        with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb', side_effect=track_wandb_save):
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local', side_effect=track_local_save):
                study = optimizer.optimize(
                    n_trials=12,
                    config_overrides={},
                    callbacks=[]
                )

        # WandB uploads should happen at: 5, 10, and final save at 12
        expected_wandb_saves = [5, 10, 12]
        assert wandb_save_calls == expected_wandb_saves, \
            f"Expected WandB uploads at {expected_wandb_saves}, got {wandb_save_calls}"

    def test_wandb_uploads_with_save_every_10(self):
        """Test with save_every_n_trials=10 and 10 trials - should upload only once at end."""
        wandb_save_calls = []

        def track_wandb_save(wandb_project, *, study_name=None, study=None, total_trials_completed=None, **kwargs):
            wandb_save_calls.append(total_trials_completed)
            return True

        def track_local_save(*args, **kwargs):
            return True

        optimizer = PausibleOptunaOptimizer(
            base_config={'dummy': 'config'},
            search_space=lambda trial: {'x': trial.suggest_float('x', 0, 1)},
            model_class=Mock,
            datamodule_class=Mock,
            wandb_project="test-project",
            save_every_n_trials=10,
            enable_pause=False,
            use_reflow=False,
            restart_on_save=False,
        )

        with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb', side_effect=track_wandb_save):
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local', side_effect=track_local_save):
                study = optimizer.optimize(
                    n_trials=10,
                    config_overrides={},
                    callbacks=[]
                )

        # WandB upload should happen only at trial 10
        expected_wandb_saves = [10]
        assert wandb_save_calls == expected_wandb_saves, \
            f"Expected WandB uploads at {expected_wandb_saves}, got {wandb_save_calls}"

    def test_no_wandb_when_project_is_none(self):
        """Test that WandB save is never called when wandb_project is None."""
        wandb_save_calls = []
        local_save_calls = []

        def track_wandb_save(*args, **kwargs):
            wandb_save_calls.append(1)
            return True

        def track_local_save(checkpoint_dir, study, total_trials_completed, **kwargs):
            local_save_calls.append(total_trials_completed)
            return True

        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = PausibleOptunaOptimizer(
                base_config={'dummy': 'config'},
                search_space=lambda trial: {'x': trial.suggest_float('x', 0, 1)},
                model_class=Mock,
                datamodule_class=Mock,
                wandb_project=None,  # No WandB
                local_checkpoint_dir=Path(tmpdir),
                save_every_n_trials=2,
                enable_pause=False,
                use_reflow=False,
                restart_on_save=False,
            )

            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb', side_effect=track_wandb_save):
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local', side_effect=track_local_save):
                    study = optimizer.optimize(
                        n_trials=6,
                        config_overrides={},
                        callbacks=[]
                    )

        # WandB should never be called
        assert len(wandb_save_calls) == 0, f"WandB save should not be called when project is None"

        # Local saves should still happen (6 in loop + 1 final = 7)
        assert len(local_save_calls) >= 6, f"Expected at least 6 local saves, got {len(local_save_calls)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
