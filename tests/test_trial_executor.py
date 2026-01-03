"""
Tests for TrialExecutor and related classes.
"""

import gc
import pytest
from unittest.mock import MagicMock, patch
import optuna


class TestTrialExecutorConfig:
    """Tests for TrialExecutorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from LightningTune.optuna.trial_executor import TrialExecutorConfig

        config = TrialExecutorConfig()

        assert config.n_trials == 50
        assert config.save_every_n_trials == 10
        assert config.restart_on_save is False
        assert config.restart_every_trial is True
        assert config.enable_pause is True
        assert config.cleanup_between_trials is True

    def test_custom_values(self):
        """Test custom configuration."""
        from LightningTune.optuna.trial_executor import TrialExecutorConfig

        config = TrialExecutorConfig(
            n_trials=100,
            save_every_n_trials=5,
            restart_on_save=True,
            enable_pause=False,
        )

        assert config.n_trials == 100
        assert config.save_every_n_trials == 5
        assert config.restart_on_save is True
        assert config.enable_pause is False


class TestTrialExecutorState:
    """Tests for TrialExecutorState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        from LightningTune.optuna.trial_executor import TrialExecutorState

        state = TrialExecutorState()

        assert state.trials_completed == 0
        assert state.trials_completed_this_run == 0
        assert state.pause_requested is False
        assert state.quit_requested is False
        assert state.should_exit_for_restart is False
        assert state.last_wandb_upload_count == 0


class TestSimpleTrialExecutor:
    """Tests for SimpleTrialExecutor class."""

    def test_run_basic(self):
        """Test basic trial execution."""
        from LightningTune.optuna.trial_executor import SimpleTrialExecutor

        study = optuna.create_study()

        def objective(trial):
            x = trial.suggest_float("x", 0, 10)
            return x ** 2

        executor = SimpleTrialExecutor(study, objective, n_trials=5)
        result = executor.run()

        assert len(result.trials) == 5
        assert result.best_value is not None

    def test_run_with_callback(self):
        """Test execution with trial completion callback."""
        from LightningTune.optuna.trial_executor import SimpleTrialExecutor

        study = optuna.create_study()
        completed_trials = []

        def on_complete(trial, count):
            completed_trials.append((trial.number, count))

        executor = SimpleTrialExecutor(
            study,
            lambda trial: trial.suggest_float("x", 0, 10),
            n_trials=3,
            on_trial_complete=on_complete,
        )
        executor.run()

        assert len(completed_trials) == 3
        assert completed_trials[0] == (0, 1)
        assert completed_trials[1] == (1, 2)
        assert completed_trials[2] == (2, 3)

    def test_run_handles_pruned_trials(self):
        """Test handling of pruned trials."""
        from LightningTune.optuna.trial_executor import SimpleTrialExecutor

        study = optuna.create_study()

        def objective(trial):
            x = trial.suggest_float("x", 0, 10)
            if trial.number == 1:
                raise optuna.TrialPruned()
            return x ** 2

        executor = SimpleTrialExecutor(study, objective, n_trials=3)
        result = executor.run()

        assert len(result.trials) == 3
        assert result.trials[1].state == optuna.trial.TrialState.PRUNED

    def test_run_handles_failed_trials(self):
        """Test handling of failed trials."""
        from LightningTune.optuna.trial_executor import SimpleTrialExecutor

        study = optuna.create_study()

        def objective(trial):
            x = trial.suggest_float("x", 0, 10)
            if trial.number == 1:
                raise ValueError("Test error")
            return x ** 2

        executor = SimpleTrialExecutor(study, objective, n_trials=3)
        result = executor.run()

        assert len(result.trials) == 3
        assert result.trials[1].state == optuna.trial.TrialState.FAIL


class TestTrialExecutor:
    """Tests for TrialExecutor class."""

    def test_run_basic(self):
        """Test basic execution."""
        from LightningTune.optuna.trial_executor import TrialExecutor, TrialExecutorConfig

        study = optuna.create_study()
        config = TrialExecutorConfig(n_trials=3, enable_pause=False)

        executor = TrialExecutor(
            study=study,
            objective=lambda trial: trial.suggest_float("x", 0, 10),
            config=config,
        )

        result = executor.run()
        assert len(result.trials) == 3

    def test_run_from_initial_trials(self):
        """Test resuming from initial trial count."""
        from LightningTune.optuna.trial_executor import TrialExecutor, TrialExecutorConfig

        study = optuna.create_study()
        # Pre-populate with 2 trials
        study.optimize(lambda trial: trial.suggest_float("x", 0, 10), n_trials=2)

        config = TrialExecutorConfig(n_trials=5, enable_pause=False)

        executor = TrialExecutor(
            study=study,
            objective=lambda trial: trial.suggest_float("x", 0, 10),
            config=config,
        )

        result = executor.run(initial_trials_completed=2)

        # Should have run 3 more trials (5 - 2)
        assert len(result.trials) == 5
        assert executor.state.trials_completed == 5
        assert executor.state.trials_completed_this_run == 3

    def test_pause_property(self):
        """Test pause property with and without keyboard handler."""
        from LightningTune.optuna.trial_executor import TrialExecutor, TrialExecutorConfig

        study = optuna.create_study()
        config = TrialExecutorConfig(n_trials=3, enable_pause=False)

        # Without keyboard handler
        executor = TrialExecutor(
            study=study,
            objective=lambda trial: 0,
            config=config,
        )
        assert executor.should_pause is False

        executor.should_pause = True
        assert executor.should_pause is True

    def test_pause_property_with_handler(self):
        """Test pause property with keyboard handler."""
        from LightningTune.optuna.trial_executor import TrialExecutor, TrialExecutorConfig
        from LightningTune.input.keyboard_handler import HPOKeyboardHandler

        study = optuna.create_study()
        config = TrialExecutorConfig(n_trials=3, enable_pause=True)
        handler = HPOKeyboardHandler()

        executor = TrialExecutor(
            study=study,
            objective=lambda trial: 0,
            config=config,
            keyboard_handler=handler,
        )

        assert executor.should_pause is False

        handler.pause_requested = True
        assert executor.should_pause is True

    def test_cleanup_memory(self):
        """Test memory cleanup between trials."""
        from LightningTune.optuna.trial_executor import TrialExecutor, TrialExecutorConfig

        study = optuna.create_study()
        config = TrialExecutorConfig(n_trials=2, cleanup_between_trials=True, enable_pause=False)

        gc_collected = []

        # Mock gc.collect to track calls
        with patch('gc.collect', side_effect=lambda: gc_collected.append(True)):
            executor = TrialExecutor(
                study=study,
                objective=lambda trial: trial.suggest_float("x", 0, 10),
                config=config,
            )
            executor.run()

        # Should have been called after each trial
        assert len(gc_collected) >= 2

    def test_trial_complete_callback(self):
        """Test trial completion callback."""
        from LightningTune.optuna.trial_executor import TrialExecutor, TrialExecutorConfig

        study = optuna.create_study()
        config = TrialExecutorConfig(n_trials=3, enable_pause=False)
        callbacks = []

        executor = TrialExecutor(
            study=study,
            objective=lambda trial: trial.suggest_float("x", 0, 10),
            config=config,
            on_trial_complete=lambda trial, count: callbacks.append((trial.number, count)),
        )
        executor.run()

        assert len(callbacks) == 3
        assert callbacks[0] == (0, 1)
        assert callbacks[2] == (2, 3)

    def test_should_save_checkpoint(self):
        """Test checkpoint save decision."""
        from LightningTune.optuna.trial_executor import TrialExecutor, TrialExecutorConfig

        study = optuna.create_study()
        config = TrialExecutorConfig(
            n_trials=100,
            save_every_n_trials=5,
            enable_pause=False,
        )

        executor = TrialExecutor(
            study=study,
            objective=lambda trial: 0,
            config=config,
        )

        # Initially no save
        executor.state.trials_completed = 3
        executor.state.last_wandb_upload_count = 0
        assert executor._should_save_checkpoint() is False

        # At 5 trials, should save
        executor.state.trials_completed = 5
        assert executor._should_save_checkpoint() is True

        # After save at 5, shouldn't save at 6
        executor.state.last_wandb_upload_count = 5
        executor.state.trials_completed = 6
        assert executor._should_save_checkpoint() is False

        # At 10, should save again
        executor.state.trials_completed = 10
        assert executor._should_save_checkpoint() is True

    def test_restart_every_trial(self):
        """Test restart_every_trial behavior."""
        from LightningTune.optuna.trial_executor import TrialExecutor, TrialExecutorConfig

        study = optuna.create_study()
        config = TrialExecutorConfig(
            n_trials=10,
            restart_on_save=True,
            restart_every_trial=True,
            enable_pause=False,
        )

        executor = TrialExecutor(
            study=study,
            objective=lambda trial: trial.suggest_float("x", 0, 10),
            config=config,
        )

        result = executor.run()

        # Should only run 1 trial then exit for restart
        assert executor.state.trials_completed == 1
        assert executor.state.should_exit_for_restart is True


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
