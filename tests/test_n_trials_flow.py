"""
Test that n_trials flows correctly from CLI through to optimization loop.

This test reproduces a bug where running with --n-trials 100 resulted in
only 3 trials being executed.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import optuna
from argparse import Namespace


class TestNTrialsFlow:
    """Test n_trials value flow from CLI to optimization loop."""

    def test_n_trials_passed_to_optimizer(self):
        """Test that n_trials from CLI is correctly passed to optimizer.optimize()."""
        from LightningTune.hpo_runner import HPORunner
        from lightning.pytorch import LightningModule

        # Create a minimal model class
        class DummyModel(LightningModule):
            def __init__(self, **kwargs):
                super().__init__()
                self.save_hyperparameters()
            def training_step(self, batch, batch_idx):
                return {"loss": 0.1}
            def configure_optimizers(self):
                return None

        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={},
        )

        # Mock the optimizer to capture the n_trials value
        with patch('LightningTune.hpo_runner.PausibleOptunaOptimizer') as mock_optimizer_class:
            mock_study = Mock()
            mock_optimizer = Mock()
            mock_optimizer.optimize.return_value = mock_study
            mock_optimizer_class.return_value = mock_optimizer

            # Run with explicit n_trials
            argv = ['--n-trials', '100', '--test-mode', '--no-reflow']
            runner.run_from_cli(argv=argv)

            # Verify n_trials was passed correctly
            mock_optimizer.optimize.assert_called_once()
            call_kwargs = mock_optimizer.optimize.call_args

            # n_trials should be passed as first positional arg or keyword arg
            if call_kwargs.args:
                actual_n_trials = call_kwargs.args[0]
            else:
                actual_n_trials = call_kwargs.kwargs.get('n_trials')

            assert actual_n_trials == 100, f"Expected n_trials=100, got {actual_n_trials}"

    def test_n_trials_not_overwritten_by_save_every(self):
        """Test that n_trials is not confused with save_every_n_trials."""
        from LightningTune.hpo_runner import HPORunner
        from lightning.pytorch import LightningModule

        class DummyModel(LightningModule):
            def __init__(self, **kwargs):
                super().__init__()
                self.save_hyperparameters()
            def training_step(self, batch, batch_idx):
                return {"loss": 0.1}
            def configure_optimizers(self):
                return None

        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={},
        )

        with patch('LightningTune.hpo_runner.PausibleOptunaOptimizer') as mock_optimizer_class:
            mock_study = Mock()
            mock_optimizer = Mock()
            mock_optimizer.optimize.return_value = mock_study
            mock_optimizer_class.return_value = mock_optimizer

            # Run with n_trials=100 and save_every=3 (default)
            argv = ['--n-trials', '100', '--save-every', '3', '--test-mode', '--no-reflow']
            runner.run_from_cli(argv=argv)

            # Verify optimizer was created with correct save_every
            create_call = mock_optimizer_class.call_args
            assert create_call.kwargs.get('save_every_n_trials') == 3

            # Verify optimize was called with correct n_trials
            optimize_call = mock_optimizer.optimize.call_args
            if optimize_call.args:
                actual_n_trials = optimize_call.args[0]
            else:
                actual_n_trials = optimize_call.kwargs.get('n_trials')

            assert actual_n_trials == 100, f"n_trials should be 100, not save_every (3)"

    def test_optimizer_loop_runs_correct_number_of_trials(self):
        """Test that the optimization loop actually runs the specified number of trials."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer
        from unittest.mock import patch, MagicMock

        # Track how many trials were run
        trials_run = []

        def counting_objective(trial):
            trials_run.append(trial.number)
            return len(trials_run) * 0.01

        optimizer = PausibleOptunaOptimizer(
            base_config={"dummy": "config"},
            search_space=lambda trial: {},
            model_class=None,
            datamodule_class=None,
            save_every_n_trials=3,  # Default
        )

        # Patch the underlying optimizer to use our counting objective
        with patch.object(optimizer, 'use_reflow', False):
            # Create a mock for OptunaDrivenOptimizer that uses our objective
            original_class = None
            def mock_optimizer_init(self_opt, *args, **kwargs):
                # Store config but use our objective
                self_opt.base_config = kwargs.get('base_config', {})
                self_opt.search_space = kwargs.get('search_space', lambda t: {})

            with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer') as MockOptimizer:
                mock_instance = MagicMock()
                mock_instance.create_objective.return_value = counting_objective
                MockOptimizer.return_value = mock_instance

                # Run optimization with n_trials=10
                n_trials = 10
                study = optimizer.optimize(n_trials=n_trials)

                # Verify correct number of trials ran
                assert len(trials_run) == n_trials, \
                    f"Expected {n_trials} trials, but only {len(trials_run)} ran: {trials_run}"
                assert optimizer.total_trials_completed == n_trials

    def test_n_trials_10_runs_10_trials(self):
        """Concrete test: n_trials=10 should run exactly 10 trials."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        trial_count = [0]

        def counting_objective(trial):
            trial_count[0] += 1
            return trial_count[0] * 0.01

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=None,
            datamodule_class=None,
            save_every_n_trials=3,
        )

        with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer') as MockOpt:
            mock_instance = MagicMock()
            mock_instance.create_objective.return_value = counting_objective
            MockOpt.return_value = mock_instance

            study = optimizer.optimize(n_trials=10)

            assert trial_count[0] == 10, f"Expected 10 trials, got {trial_count[0]}"
            assert optimizer.total_trials_completed == 10

    def test_n_trials_100_with_save_every_3(self):
        """Test that n_trials=100 with save_every=3 runs all 100 trials."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        trial_count = [0]

        def counting_objective(trial):
            trial_count[0] += 1
            return trial_count[0] * 0.001

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=None,
            datamodule_class=None,
            save_every_n_trials=3,  # This should NOT limit trials to 3
        )

        with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer') as MockOpt:
            mock_instance = MagicMock()
            mock_instance.create_objective.return_value = counting_objective
            MockOpt.return_value = mock_instance

            # Run with n_trials=100
            study = optimizer.optimize(n_trials=100)

            # Should have run all 100 trials
            assert trial_count[0] == 100, \
                f"Expected 100 trials but only {trial_count[0]} ran. " \
                f"Bug: save_every_n_trials (3) may be confused with n_trials."
            assert optimizer.total_trials_completed == 100

    def test_fresh_start_n_trials_not_from_checkpoint(self):
        """Test that fresh start uses CLI n_trials, not some cached value."""
        from LightningTune.hpo_runner import HPORunner
        from lightning.pytorch import LightningModule

        class DummyModel(LightningModule):
            def __init__(self, **kwargs):
                super().__init__()
            def training_step(self, batch, batch_idx):
                return {"loss": 0.1}
            def configure_optimizers(self):
                return None

        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={},
        )

        with patch('LightningTune.hpo_runner.PausibleOptunaOptimizer') as mock_optimizer_class:
            mock_study = Mock()
            mock_optimizer = Mock()
            mock_optimizer.optimize.return_value = mock_study
            mock_optimizer_class.return_value = mock_optimizer

            # Fresh start (no --resume-from)
            argv = ['--n-trials', '50', '--test-mode', '--no-reflow']
            runner.run_from_cli(argv=argv)

            # n_trials should be 50, not some default or cached value
            optimize_call = mock_optimizer.optimize.call_args
            if optimize_call.args:
                actual_n_trials = optimize_call.args[0]
            else:
                actual_n_trials = optimize_call.kwargs.get('n_trials')

            assert actual_n_trials == 50, f"Fresh start should use CLI n_trials=50, got {actual_n_trials}"


    def test_n_trials_with_wandb_project(self):
        """Test n_trials when wandb_project is set (mimics user's scenario)."""
        from LightningTune.hpo_runner import HPORunner
        from lightning.pytorch import LightningModule

        class DummyModel(LightningModule):
            def __init__(self, **kwargs):
                super().__init__()
            def training_step(self, batch, batch_idx):
                return {"loss": 0.1}
            def configure_optimizers(self):
                return None

        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={},
        )

        with patch('LightningTune.hpo_runner.PausibleOptunaOptimizer') as mock_optimizer_class:
            mock_study = Mock()
            mock_optimizer = Mock()
            mock_optimizer.optimize.return_value = mock_study
            mock_optimizer_class.return_value = mock_optimizer

            # Mimic user's command with wandb
            argv = ['--n-trials', '100', '--wandb', 'test_project', '--test-mode', '--no-reflow']
            runner.run_from_cli(argv=argv)

            # Verify n_trials was passed correctly
            optimize_call = mock_optimizer.optimize.call_args
            if optimize_call.args:
                actual_n_trials = optimize_call.args[0]
            else:
                actual_n_trials = optimize_call.kwargs.get('n_trials')

            assert actual_n_trials == 100, f"With --wandb, expected n_trials=100, got {actual_n_trials}"

    def test_args_n_trials_value_after_parsing(self):
        """Test that self.args.n_trials has correct value after parsing."""
        from LightningTune.hpo_runner import HPORunner
        from lightning.pytorch import LightningModule

        class DummyModel(LightningModule):
            def __init__(self, **kwargs):
                super().__init__()
            def training_step(self, batch, batch_idx):
                return {"loss": 0.1}
            def configure_optimizers(self):
                return None

        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={},
        )

        with patch('LightningTune.hpo_runner.PausibleOptunaOptimizer') as mock_optimizer_class:
            mock_study = Mock()
            mock_optimizer = Mock()
            mock_optimizer.optimize.return_value = mock_study
            mock_optimizer_class.return_value = mock_optimizer

            argv = ['--n-trials', '100', '--wandb', 'test_project', '--trial-steps', '40000', '--test-mode', '--no-reflow']
            runner.run_from_cli(argv=argv)

            # Check args values
            assert runner.args.n_trials == 100, f"args.n_trials should be 100, got {runner.args.n_trials}"
            assert runner.args.wandb == 'test_project'
            assert runner.args.trial_steps == 40000

    def test_n_trials_different_values(self):
        """Test various n_trials values are correctly passed."""
        from LightningTune.hpo_runner import HPORunner
        from lightning.pytorch import LightningModule

        class DummyModel(LightningModule):
            def __init__(self, **kwargs):
                super().__init__()
            def training_step(self, batch, batch_idx):
                return {"loss": 0.1}
            def configure_optimizers(self):
                return None

        for expected_n_trials in [3, 10, 50, 100, 200]:
            runner = HPORunner(
                model_class=DummyModel,
                datamodule_class=None,
                search_space=lambda trial: {},
                base_config={},
            )

            with patch('LightningTune.hpo_runner.PausibleOptunaOptimizer') as mock_optimizer_class:
                mock_study = Mock()
                mock_optimizer = Mock()
                mock_optimizer.optimize.return_value = mock_study
                mock_optimizer_class.return_value = mock_optimizer

                argv = ['--n-trials', str(expected_n_trials), '--test-mode', '--no-reflow']
                runner.run_from_cli(argv=argv)

                optimize_call = mock_optimizer.optimize.call_args
                if optimize_call.args:
                    actual_n_trials = optimize_call.args[0]
                else:
                    actual_n_trials = optimize_call.kwargs.get('n_trials')

                assert actual_n_trials == expected_n_trials, \
                    f"Expected n_trials={expected_n_trials}, got {actual_n_trials}"


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
