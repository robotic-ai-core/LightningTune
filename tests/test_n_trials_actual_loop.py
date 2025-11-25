"""
Test that the optimization loop actually runs the correct number of trials.
Uses mocked trials to verify the loop behavior.
"""

import pytest
import optuna
from unittest.mock import patch, MagicMock


class TestActualTrialExecution:
    """Test actual trial execution count."""

    def test_100_trials_with_save_every_3(self):
        """
        Reproduce the bug: n_trials=100, save_every=3 should run 100 trials.

        Bug report: User ran with --n-trials 100 but only 3 trials executed.
        """
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        trial_count = [0]

        def simple_objective(trial):
            trial_count[0] += 1
            return trial_count[0] * 0.001

        # Create optimizer with save_every=3 (the suspected culprit)
        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=None,
            datamodule_class=None,
            save_every_n_trials=3,
            wandb_project=None,  # No actual WandB
        )

        # Patch to use our simple objective
        with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer') as MockOpt:
            mock_instance = MagicMock()
            mock_instance.create_objective.return_value = simple_objective
            MockOpt.return_value = mock_instance

            # Run with n_trials=100
            study = optimizer.optimize(n_trials=100)

            # THE KEY ASSERTION
            assert trial_count[0] == 100, \
                f"BUG REPRODUCED: Expected 100 trials but only {trial_count[0]} ran!"
            assert optimizer.total_trials_completed == 100

    def test_10_trials_basic(self):
        """Basic test: 10 trials should run 10 times."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        trial_count = [0]

        def simple_objective(trial):
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
            mock_instance.create_objective.return_value = simple_objective
            MockOpt.return_value = mock_instance

            study = optimizer.optimize(n_trials=10)

            assert trial_count[0] == 10, f"Expected 10 trials, got {trial_count[0]}"
            assert optimizer.total_trials_completed == 10

    def test_save_every_equals_n_trials(self):
        """Test when save_every equals n_trials (both 3)."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        trial_count = [0]

        def simple_objective(trial):
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
            mock_instance.create_objective.return_value = simple_objective
            MockOpt.return_value = mock_instance

            # This is the suspicious case - save_every=3 and n_trials=3
            study = optimizer.optimize(n_trials=3)

            assert trial_count[0] == 3, f"Expected 3 trials, got {trial_count[0]}"

    def test_with_wandb_project_set(self):
        """Test with wandb_project set (mimics user's scenario)."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        trial_count = [0]

        def simple_objective(trial):
            trial_count[0] += 1
            return trial_count[0] * 0.001

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=None,
            datamodule_class=None,
            save_every_n_trials=3,
            wandb_project="test_project",  # Set like user's command
        )

        # Mock both the optimizer and WandB saves
        with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer') as MockOpt:
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_save:
                mock_save.return_value = True  # Simulate successful saves

                mock_instance = MagicMock()
                mock_instance.create_objective.return_value = simple_objective
                MockOpt.return_value = mock_instance

                study = optimizer.optimize(n_trials=100)

                assert trial_count[0] == 100, \
                    f"With wandb_project set: Expected 100 trials, got {trial_count[0]}"


    def test_hpo_runner_e2e_100_trials(self):
        """End-to-end test: HPORunner with n_trials=100 should run 100 trials."""
        from LightningTune.hpo_runner import HPORunner
        from lightning.pytorch import LightningModule

        trial_count = [0]

        def simple_objective(trial):
            trial_count[0] += 1
            return trial_count[0] * 0.001

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

        # Patch the optimizer class to inject our objective
        with patch('LightningTune.hpo_runner.PausibleOptunaOptimizer') as MockPausible:
            with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer') as MockOpt:
                # Set up the mock chain
                mock_opt_instance = MagicMock()
                mock_opt_instance.create_objective.return_value = simple_objective
                MockOpt.return_value = mock_opt_instance

                # Create a real PausibleOptunaOptimizer but with mocked underlying optimizer
                from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

                def create_real_optimizer(*args, **kwargs):
                    return PausibleOptunaOptimizer(*args, **kwargs)

                MockPausible.side_effect = create_real_optimizer

                # Run with exact user command args
                argv = ['--n-trials', '100', '--wandb', 'test_project', '--save-every', '3',
                        '--test-mode', '--no-reflow']

                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_save:
                    mock_save.return_value = True
                    study = runner.run_from_cli(argv=argv)

                # THE KEY ASSERTION
                assert trial_count[0] == 100, \
                    f"HPORunner E2E: Expected 100 trials but only {trial_count[0]} ran!"

    def test_exact_user_scenario(self):
        """
        Exact reproduction of user's scenario:
        python scripts/world_model_hpo.py --n-trials 100 --wandb pusht_hpo_l1_prenorm --trial-steps 40000
        """
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        trial_count = [0]

        def simple_objective(trial):
            trial_count[0] += 1
            return trial_count[0] * 0.001

        # Exact user scenario: wandb project set, save_every=3 (default)
        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=None,
            datamodule_class=None,
            save_every_n_trials=3,  # Default value
            wandb_project="pusht_hpo_l1_prenorm",  # User's project
            study_name="world_model_pusht_hpo",  # Default study name
        )

        with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer') as MockOpt:
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_save:
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                    mock_save.return_value = True
                    mock_local.return_value = True

                    mock_instance = MagicMock()
                    mock_instance.create_objective.return_value = simple_objective
                    MockOpt.return_value = mock_instance

                    # Run with n_trials=100
                    study = optimizer.optimize(n_trials=100)

                    # Verify all 100 trials ran
                    assert trial_count[0] == 100, \
                        f"User scenario: Expected 100 trials but only {trial_count[0]} ran!"
                    assert optimizer.total_trials_completed == 100

                    # Verify saves happened at correct intervals (every 3 trials)
                    # With 100 trials and save_every=3, we expect:
                    # - 33 periodic saves (at trials 3, 6, ..., 99)
                    # - 1 final save after trial 100 completes (since 100 > 99)
                    # Total: 33-34 saves depending on whether final save is triggered
                    expected_periodic_saves = 100 // 3
                    assert mock_save.call_count >= expected_periodic_saves, \
                        f"Expected at least {expected_periodic_saves} WandB saves, got {mock_save.call_count}"
                    assert mock_save.call_count <= expected_periodic_saves + 1, \
                        f"Expected at most {expected_periodic_saves + 1} WandB saves, got {mock_save.call_count}"

    def test_reflow_path_with_real_lightningmodule(self):
        """
        Test that OptunaDrivenOptimizer (which uses LightningReflow) is used.

        After consolidation, OptunaDrivenOptimizer always uses LightningReflow
        for training execution. This test verifies the optimizer is created
        and runs correctly with a real LightningModule.
        """
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer
        from lightning.pytorch import LightningModule

        trial_count = [0]

        def simple_objective(trial):
            trial_count[0] += 1
            return trial_count[0] * 0.001

        # Create a real LightningModule class
        class DummyModel(LightningModule):
            def __init__(self, **kwargs):
                super().__init__()
            def training_step(self, batch, batch_idx):
                return {"loss": 0.1}
            def configure_optimizers(self):
                return None

        # User scenario with real model class
        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=DummyModel,
            datamodule_class=None,
            save_every_n_trials=3,
            wandb_project="pusht_hpo_l1_prenorm",
            study_name="world_model_pusht_hpo",
        )

        # Patch OptunaDrivenOptimizer (now the only optimizer, uses LightningReflow internally)
        with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer') as MockOpt:
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_save:
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                    mock_save.return_value = True
                    mock_local.return_value = True

                    mock_opt_instance = MagicMock()
                    mock_opt_instance.create_objective.return_value = simple_objective
                    MockOpt.return_value = mock_opt_instance

                    # Run with n_trials=100
                    study = optimizer.optimize(n_trials=100)

                    # Verify all 100 trials ran
                    assert trial_count[0] == 100, \
                        f"Expected 100 trials but only {trial_count[0]} ran!"
                    assert optimizer.total_trials_completed == 100

                    # Verify OptunaDrivenOptimizer was used
                    assert MockOpt.called, "OptunaDrivenOptimizer should be used"

    def test_resume_command_printed_on_pause(self):
        """
        Test that the correct resume command is printed when optimization is paused.
        """
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer
        import sys

        trial_count = [0]

        def simple_objective(trial):
            trial_count[0] += 1
            return trial_count[0] * 0.001

        # Simulate original argv
        original_argv = sys.argv.copy()
        sys.argv = ['scripts/world_model_hpo.py', '--n-trials', '100',
                    '--wandb', 'test_project', '--study-name', 'my_study',
                    '--trial-steps', '40000']

        try:
            optimizer = PausibleOptunaOptimizer(
                base_config={},
                search_space=lambda trial: {},
                model_class=None,
                datamodule_class=None,
                save_every_n_trials=3,
                wandb_project="test_project",
                study_name="my_study",
            )

            with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer') as MockOpt:
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_save:
                    with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                        mock_save.return_value = True
                        mock_local.return_value = True

                        mock_instance = MagicMock()
                        mock_instance.create_objective.return_value = simple_objective
                        MockOpt.return_value = mock_instance

                        # Trigger pause after 5 trials
                        def pause_after_5(trial):
                            result = simple_objective(trial)
                            if trial_count[0] >= 5:
                                optimizer._pause_requested = True
                            return result

                        mock_instance.create_objective.return_value = pause_after_5

                        # Run optimization
                        study = optimizer.optimize(n_trials=100)

                        # Verify pause occurred
                        assert optimizer.should_pause, "Optimization should have paused"
                        assert trial_count[0] == 5, f"Expected 5 trials, got {trial_count[0]}"

                        # Check the resume command
                        resume_cmd = optimizer._build_resume_command()
                        assert "scripts/world_model_hpo.py" in resume_cmd
                        assert "--wandb test_project" in resume_cmd
                        assert "--study-name my_study" in resume_cmd
                        assert "--trial-steps 40000" in resume_cmd
                        assert "--resume-from latest" in resume_cmd
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
