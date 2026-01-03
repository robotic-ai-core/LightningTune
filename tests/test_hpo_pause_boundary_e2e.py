#!/usr/bin/env python
"""
End-to-end test for HPO pause boundary behavior.

This test validates that HPO uses TRIAL-BOUNDARY pause only:
1. PauseCallback is disabled (no validation-boundary pause)
2. Trial-boundary pause works via PausibleOptunaOptimizer
3. FlowProgressBarCallback is used (Lightning's default progress bar disabled)
"""

import os
import sys
from unittest.mock import patch
import pytest

# Add LightningTune root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightning.pytorch import LightningModule
import torch


class SimpleModel(LightningModule):
    """Simple model for testing."""

    def __init__(self, **kwargs):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x = torch.randn(4, 10)
        loss = self(x).mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = torch.randn(4, 10)
        loss = self(x).mean()
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class TestHPOPauseBoundary:
    """Test that HPO uses trial-boundary pause only."""

    def test_pause_callbacks_filtered_from_config(self, tmp_path):
        """
        Verify PauseCallback and EarlyPauseCallback are filtered from config,
        but FlowProgressBarCallback is added for progress display.
        """
        from LightningTune.optuna.optimizer import OptunaDrivenOptimizer
        import optuna

        # Config that explicitly includes PauseCallback (like world_model_pusht.yaml)
        config_with_pause_callback = {
            'trainer': {
                'max_epochs': 1,
                'callbacks': [
                    {'class_path': 'lightning_reflow.callbacks.pause.PauseCallback', 'init_args': {}},
                    {'class_path': 'lightning_reflow.callbacks.pause.EarlyPauseCallback', 'init_args': {}},
                ]
            }
        }

        optimizer = OptunaDrivenOptimizer(
            base_config=config_with_pause_callback,
            search_space=lambda trial, config: config,
            model_class=SimpleModel,
            datamodule_class=None,
            study_name="test_filter",
            save_checkpoints=False,
        )

        study = optuna.create_study()
        trial = study.ask()
        callbacks = optimizer._prepare_callbacks(trial, config_with_pause_callback)

        try:
            from lightning_reflow.callbacks.pause import PauseCallback, EarlyPauseCallback
            from lightning_reflow.callbacks.monitoring import FlowProgressBarCallback

            # Verify NO PauseCallback or EarlyPauseCallback
            pause_callbacks = [cb for cb in callbacks if isinstance(cb, (PauseCallback, EarlyPauseCallback))]
            assert len(pause_callbacks) == 0, (
                f"PauseCallback should be filtered out but found: {pause_callbacks}"
            )

            # Verify FlowProgressBarCallback IS present (since PauseCallback inherits from it)
            flow_progress = [cb for cb in callbacks if isinstance(cb, FlowProgressBarCallback)]
            assert len(flow_progress) >= 1, (
                f"FlowProgressBarCallback should be added, but found none"
            )
        except ImportError:
            pytest.skip("LightningReflow not available")

    def test_trial_boundary_pause_completes_current_trial(self, tmp_path):
        """
        Test that trial-boundary pause:
        1. Allows current trial to complete after 'p' is pressed
        2. Pauses at trial boundary (not mid-trial)
        3. Saves study state for resume
        """
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        trial_phases = []  # Track (trial_num, phase) tuples
        saved_checkpoint = [None]

        def tracked_objective(trial):
            trial_num = trial.number + 1
            trial_phases.append((trial_num, "start"))

            # Simulate 'p' press in the MIDDLE of trial 2
            if trial_num == 2:
                trial_phases.append((trial_num, "pause_requested"))
                optimizer._pause_requested = True

            trial_phases.append((trial_num, "end"))
            return trial_num * 0.01

        checkpoint_dir = tmp_path / "checkpoints" / "trial_pause_test"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial, config: config,
            model_class=SimpleModel,
            datamodule_class=None,
            save_every_n_trials=10,
            wandb_project=None,
            study_name="test_trial_pause",
            enable_pause=True,
            local_checkpoint_dir=checkpoint_dir,
        )

        with patch.object(optimizer, 'underlying_optimizer', create=True) as mock_opt:
            mock_opt.create_objective.return_value = tracked_objective
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                def capture_save(path, study, *args, **kwargs):
                    saved_checkpoint[0] = {'n_trials': len(study.trials)}
                    return True
                mock_local.side_effect = capture_save

                optimizer.optimize(n_trials=10)

        # Trial 2 should have completed (start -> pause_requested -> end)
        trial_2_phases = [phase for (num, phase) in trial_phases if num == 2]
        assert trial_2_phases == ["start", "pause_requested", "end"], (
            f"Trial 2 should complete even after pause requested. Got: {trial_2_phases}"
        )

        # Pause should have happened after trial 2 completed
        assert optimizer.should_pause, "Should be paused after trial 2"

        # Study should have been saved with correct trial count
        assert saved_checkpoint[0] is not None, "Checkpoint should be saved on pause"
        assert saved_checkpoint[0]['n_trials'] == 2, (
            f"Checkpoint should have 2 completed trials, got {saved_checkpoint[0]['n_trials']}"
        )

    def test_enable_pause_default_is_true(self):
        """Verify that enable_pause defaults to True in HPORunner."""
        from LightningTune.hpo_runner import HPORunner
        from unittest.mock import MagicMock

        runner = HPORunner(
            model_class=MagicMock,
            datamodule_class=None,
            search_space=lambda trial: {},
        )

        assert runner.enable_pause is True, "HPORunner.enable_pause should default to True"

    def test_lightning_progress_bar_disabled(self):
        """
        Verify HPORunner disables Lightning's default progress bar.
        This allows FlowProgressBarCallback to provide the UI instead.
        """
        from LightningTune.hpo_runner import HPORunner
        import inspect

        source = inspect.getsource(HPORunner)

        # Check that enable_progress_bar is set to False
        assert 'enable_progress_bar"] = False' in source, (
            "HPORunner should set trainer.enable_progress_bar = False"
        )
        assert 'enable_progress_bar"] = True' not in source, (
            "HPORunner should NOT set trainer.enable_progress_bar = True"
        )


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
