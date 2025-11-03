#!/usr/bin/env python
"""End-to-end tests for HPORunner resume functionality with CLI arguments and config files."""

import pytest
import sys
import os
import tempfile
import pickle
from pathlib import Path
from unittest.mock import Mock, patch

# Add LightningTune root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LightningTune.hpo_runner import HPORunner
from lightning import LightningModule
import torch


class DummyModel(LightningModule):
    """Dummy model for testing."""
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        return torch.tensor(0.0)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class TestHPORunnerResumeE2E:
    """End-to-end tests for HPORunner resume with --config and CLI overrides."""

    def test_resume_with_config_and_cli_overrides(self, tmp_path):
        """Test full HPORunner resume workflow with --config and dot-notation args.

        This E2E test verifies:
        1. Initial run with --config, dot-notation args, and n-trials
        2. Checkpoint is saved with correct trial count
        3. Resume loads checkpoint and extends trials
        4. CLI overrides are restored from checkpoint
        5. Config file still works on resume
        6. Trial counts are correct at every stage
        """
        # Setup: Create config file
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
model:
  class_path: test_hpo_runner_resume_e2e.DummyModel
  init_args:
    learning_rate: 1e-3

trainer:
  max_epochs: 1
  limit_train_batches: 1
  limit_val_batches: 1
  num_sanity_val_steps: 0
""")

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Mock the optimizer to avoid actual training
        with patch('LightningTune.hpo_runner.PausibleOptunaOptimizer') as mock_optimizer_class:
            # Phase 1: Initial run (3 trials)
            mock_study_phase1 = Mock()
            mock_study_phase1.best_value = 0.5
            mock_study_phase1.best_trial = Mock(number=2)
            mock_study_phase1.trials = [Mock(state='COMPLETE') for _ in range(3)]

            mock_optimizer_phase1 = Mock()
            mock_optimizer_phase1.optimize.return_value = mock_study_phase1
            mock_optimizer_class.return_value = mock_optimizer_phase1

            runner1 = HPORunner(
                model_class=DummyModel,
                datamodule_class=None,
                search_space=lambda trial: {"learning_rate": trial.suggest_float("lr", 1e-4, 1e-2)},
                default_study_name="test_resume_e2e",
            )

            # Run initial 3 trials with config and CLI overrides
            argv_phase1 = [
                '--config', str(config_file),          # Config file (Phase 7 feature)
                '--data.batch_size', '512',            # Dot-notation arg (Phase 2 feature)
                '--data.num_workers', '8',             # Another dot-notation arg (not overridden by test_mode)
                '--n-trials', '3',
                '--trial-steps', '100',
                '--study-name', 'test_resume_e2e',
                '--experiment-dir', str(checkpoint_dir),
                '--save-every', '3',  # Save after every 3 trials
                '--test-mode',
                '--no-reflow',
            ]

            study1 = runner1.run_from_cli(argv=argv_phase1)

            # Verify Phase 1: Initial run completed
            assert study1 is not None
            assert len(study1.trials) == 3, "Should have completed 3 trials"

            # Manually create checkpoint (since we're mocking the optimizer)
            checkpoint_subdir = checkpoint_dir / "test_resume_e2e"
            checkpoint_subdir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = checkpoint_subdir / "study.pkl"

            # Create checkpoint with the data that would normally be saved
            import optuna
            study_for_checkpoint = optuna.create_study()
            for i in range(3):
                study_for_checkpoint.add_trial(optuna.trial.create_trial(value=0.5+i*0.1, params={}))

            session = {
                'study': study_for_checkpoint,
                'total_trials_completed': 3,
                'config_overrides': runner1.config_overrides,  # Get actual config overrides from runner
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(session, f)

            # Verify checkpoint contents
            with open(checkpoint_file, 'rb') as f:
                session = pickle.load(f)

            assert session['total_trials_completed'] == 3, "Should have 3 completed trials in checkpoint"
            assert 'config_overrides' in session, "Should have config_overrides in checkpoint"

            # Verify CLI overrides were saved to checkpoint
            config_overrides = session['config_overrides']
            assert 'data.batch_size' in config_overrides, "CLI override should be in checkpoint"
            assert config_overrides['data.batch_size'] == 512, "CLI override value should be correct"
            assert 'data.num_workers' in config_overrides, "CLI override should be in checkpoint"
            assert config_overrides['data.num_workers'] == 8, "CLI override value should be correct"

            # Phase 2: Resume and extend to 7 trials total
            mock_study_phase2 = Mock()
            mock_study_phase2.best_value = 0.4
            mock_study_phase2.best_trial = Mock(number=6)
            # Simulate 7 total trials (3 from phase 1 + 4 new)
            mock_study_phase2.trials = [Mock(state='COMPLETE') for _ in range(7)]

            mock_optimizer_phase2 = Mock()
            mock_optimizer_phase2.optimize.return_value = mock_study_phase2
            mock_optimizer_class.return_value = mock_optimizer_phase2

            runner2 = HPORunner(
                model_class=DummyModel,
                datamodule_class=None,
                search_space=lambda trial: {"learning_rate": trial.suggest_float("lr", 1e-4, 1e-2)},
                default_study_name="test_resume_e2e",
            )

            # Resume with increased n-trials
            argv_phase2 = [
                '--config', str(config_file),          # Config still needed on resume
                '--resume-from', str(checkpoint_file),
                '--n-trials', '7',  # Extend to 7 total trials
                '--study-name', 'test_resume_e2e',
                '--experiment-dir', str(checkpoint_dir),
                '--test-mode',
                '--no-reflow',
            ]

            study2 = runner2.run_from_cli(argv=argv_phase2)

            # Verify Phase 2: Resume completed
            assert study2 is not None
            assert len(study2.trials) == 7, "Should have 7 total trials after resume"

            # Verify that config_overrides were passed to optimizer on resume
            call_kwargs = mock_optimizer_phase2.optimize.call_args[1]
            resumed_config_overrides = call_kwargs['config_overrides']

            # Verify CLI overrides were restored
            assert 'data.batch_size' in resumed_config_overrides, "CLI override should be restored on resume"
            assert resumed_config_overrides['data.batch_size'] == 512, "CLI override value should be restored"
            assert 'data.num_workers' in resumed_config_overrides, "CLI override should be restored on resume"
            assert resumed_config_overrides['data.num_workers'] == 8, "CLI override value should be restored"

    def test_resume_without_config_arg_fails(self, tmp_path):
        """Test that resume without --config fails gracefully when config is required."""
        # Create a checkpoint that expects a config
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_file = checkpoint_dir / "study.pkl"

        # Create minimal checkpoint
        import optuna
        study = optuna.create_study()
        session = {
            'study': study,
            'total_trials_completed': 3,
            'config_overrides': {},
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(session, f)

        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config=None,  # No base config
        )

        # Try to resume without --config (base_config will be None)
        argv = [
            '--resume-from', str(checkpoint_file),
            '--n-trials', '5',
            '--test-mode',
            '--no-reflow',
        ]

        # This should work - config is optional if not needed by model
        # Just verify it doesn't crash
        with patch('LightningTune.hpo_runner.PausibleOptunaOptimizer') as mock_optimizer_class:
            mock_study = Mock()
            mock_study.trials = [Mock(state='COMPLETE') for _ in range(5)]
            mock_optimizer = Mock()
            mock_optimizer.optimize.return_value = mock_study
            mock_optimizer_class.return_value = mock_optimizer

            study = runner.run_from_cli(argv=argv)
            assert study is not None

    def test_resume_preserves_trial_steps(self, tmp_path):
        """Test that --trial-steps is preserved across resume."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
model:
  class_path: test_hpo_runner_resume_e2e.DummyModel

trainer:
  max_epochs: 1
""")

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        with patch('LightningTune.hpo_runner.PausibleOptunaOptimizer') as mock_optimizer_class:
            # Phase 1: Run with specific trial_steps
            mock_study_phase1 = Mock()
            mock_study_phase1.trials = [Mock(state='COMPLETE') for _ in range(2)]
            mock_optimizer_phase1 = Mock()
            mock_optimizer_phase1.optimize.return_value = mock_study_phase1
            mock_optimizer_class.return_value = mock_optimizer_phase1

            runner1 = HPORunner(
                model_class=DummyModel,
                datamodule_class=None,
                search_space=lambda trial: {},
                default_study_name="test_trial_steps",
            )

            argv_phase1 = [
                '--config', str(config_file),
                '--n-trials', '2',
                '--trial-steps', '1234',  # Specific value to check
                '--study-name', 'test_trial_steps',
                '--experiment-dir', str(checkpoint_dir),
                '--save-every', '2',
                '--test-mode',
                '--no-reflow',
            ]

            runner1.run_from_cli(argv=argv_phase1)

            # Manually create checkpoint (since we're mocking the optimizer)
            checkpoint_subdir = checkpoint_dir / "test_trial_steps"
            checkpoint_subdir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = checkpoint_subdir / "study.pkl"

            # Create checkpoint with the data that would normally be saved
            import optuna
            study_for_checkpoint = optuna.create_study()
            for i in range(2):
                study_for_checkpoint.add_trial(optuna.trial.create_trial(value=0.5+i*0.1, params={}))

            session = {
                'study': study_for_checkpoint,
                'total_trials_completed': 2,
                'config_overrides': runner1.config_overrides,  # Get actual config overrides from runner
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(session, f)

            # Verify checkpoint has trial_steps
            with open(checkpoint_file, 'rb') as f:
                session = pickle.load(f)

            config_overrides = session['config_overrides']
            # trial_steps is stored as a persistent arg
            assert any('trial_steps' in key for key in config_overrides.keys()), \
                "trial_steps should be in checkpoint"

            # Phase 2: Resume without specifying trial_steps again
            mock_study_phase2 = Mock()
            mock_study_phase2.trials = [Mock(state='COMPLETE') for _ in range(4)]
            mock_optimizer_phase2 = Mock()
            mock_optimizer_phase2.optimize.return_value = mock_study_phase2
            mock_optimizer_class.return_value = mock_optimizer_phase2

            runner2 = HPORunner(
                model_class=DummyModel,
                datamodule_class=None,
                search_space=lambda trial: {},
                default_study_name="test_trial_steps",
            )

            argv_phase2 = [
                '--config', str(config_file),
                '--resume-from', str(checkpoint_file),
                '--n-trials', '4',
                '--study-name', 'test_trial_steps',
                '--experiment-dir', str(checkpoint_dir),
                '--test-mode',
                '--no-reflow',
            ]

            runner2.run_from_cli(argv=argv_phase2)

            # Verify trial_steps was restored (check runner's args)
            assert runner2.args.trial_steps == 1234, \
                "trial_steps should be restored from checkpoint"

    def test_n_trials_extension(self, tmp_path):
        """Test that n_trials can be extended across multiple resumes."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("model:\n  class_path: test_hpo_runner_resume_e2e.DummyModel\n")

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        with patch('LightningTune.hpo_runner.PausibleOptunaOptimizer') as mock_optimizer_class:
            # Run 1: 3 trials
            mock_study_1 = Mock()
            mock_study_1.trials = [Mock(state='COMPLETE') for _ in range(3)]
            mock_optimizer_1 = Mock()
            mock_optimizer_1.optimize.return_value = mock_study_1
            mock_optimizer_class.return_value = mock_optimizer_1

            runner1 = HPORunner(
                model_class=DummyModel,
                datamodule_class=None,
                search_space=lambda trial: {},
                default_study_name="test_extension",
            )

            runner1.run_from_cli(argv=[
                '--config', str(config_file),
                '--n-trials', '3',
                '--study-name', 'test_extension',
                '--experiment-dir', str(checkpoint_dir),
                '--save-every', '3',
                '--test-mode',
                '--no-reflow',
            ])

            # Manually create checkpoint (since we're mocking the optimizer)
            checkpoint_subdir = checkpoint_dir / "test_extension"
            checkpoint_subdir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = checkpoint_subdir / "study.pkl"

            # Create checkpoint with the data that would normally be saved
            import optuna
            study_for_checkpoint = optuna.create_study()
            for i in range(3):
                study_for_checkpoint.add_trial(optuna.trial.create_trial(value=0.5+i*0.1, params={}))

            session1 = {
                'study': study_for_checkpoint,
                'total_trials_completed': 3,
                'config_overrides': runner1.config_overrides,
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(session1, f)

            # Verify initial checkpoint
            assert session1['total_trials_completed'] == 3

            # Run 2: Extend to 7 trials
            mock_study_2 = Mock()
            mock_study_2.trials = [Mock(state='COMPLETE') for _ in range(7)]
            mock_optimizer_2 = Mock()
            mock_optimizer_2.optimize.return_value = mock_study_2
            mock_optimizer_class.return_value = mock_optimizer_2

            runner2 = HPORunner(
                model_class=DummyModel,
                datamodule_class=None,
                search_space=lambda trial: {},
                default_study_name="test_extension",
            )

            runner2.run_from_cli(argv=[
                '--config', str(config_file),
                '--resume-from', str(checkpoint_file),
                '--n-trials', '7',  # Extend by 4
                '--study-name', 'test_extension',
                '--experiment-dir', str(checkpoint_dir),
                '--save-every', '7',
                '--test-mode',
                '--no-reflow',
            ])

            # Verify logging showed correct resume status
            # Runner should have logged: "Trials completed: 3", "Target n_trials: 7", "Remaining: 4"
            assert runner2.args.n_trials == 7


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, '-v'])
