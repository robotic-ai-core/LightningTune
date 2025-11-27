#!/usr/bin/env python
"""Test cases for pause/cancel toggle functionality.

Note: The KeyboardMonitor class was removed as part of consolidation.
PausibleOptunaOptimizer now uses ImprovedKeyboardHandler from LightningReflow.
These tests focus on the pause functionality in PausibleOptunaOptimizer.
"""

import pytest
import time
import threading
import logging
from unittest.mock import MagicMock, patch
import sys
import os

# Add LightningTune root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPauseToggle:
    """Test pause/cancel toggle functionality.

    Note: KeyboardMonitor was removed. Tests now use mocks for pause state.
    Consolidated: Previously 6 separate tests that all tested the same flag toggling.
    """

    def test_pause_state_toggle_and_persistence(self):
        """Test pause state toggling, persistence, and thread-safe callback.

        Consolidated test covering:
        - Initial state is False
        - State can be set to True
        - State persists across multiple reads
        - State can be cleared
        - Toggle sequence works correctly
        - Thread-safe callback can modify state
        """
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test"
        )

        # Initial state is False
        assert not optimizer._pause_requested

        # Set pause
        optimizer._pause_requested = True
        assert optimizer._pause_requested

        # State persists across multiple reads
        assert optimizer._pause_requested
        assert optimizer._pause_requested

        # Clear pause
        optimizer._pause_requested = False
        assert not optimizer._pause_requested

        # Toggle sequence
        optimizer._pause_requested = True
        assert optimizer._pause_requested
        optimizer._pause_requested = False
        assert not optimizer._pause_requested
        optimizer._pause_requested = True
        assert optimizer._pause_requested

        # Thread-safe callback can set pause
        def set_pause():
            with optimizer._pause_lock:
                optimizer._pause_requested = True
        set_pause()
        assert optimizer._pause_requested

    def test_pause_related_code_exists(self):
        """Verify pause-related code exists in optimizer (sanity check)."""
        from LightningTune.optuna import pausible_optimizer
        import inspect

        source = inspect.getsource(pausible_optimizer.PausibleOptunaOptimizer)

        # Check for essential pause-related content
        assert "pause" in source.lower(), "Missing pause-related code"
        assert "trial" in source.lower(), "Missing trial-related code"


class TestOptimizeLoopPause:
    """Test pause behavior during the optimize loop.

    Consolidated: Previously 7 tests, reduced to 2 meaningful tests.
    - Removed test_pause_flag_read_correctly (covered by TestPauseToggle)
    - Removed test_callback_toggles_pause_state (covered by TestPauseToggle)
    - Merged test_pause_with_polling_active_path, test_pause_with_manual_polling_no_polling_thread,
      and test_pause_detection_with_polling_thread_race_condition (all just run N trials without pause)
    """

    def test_pause_stops_optimization_early(self):
        """Test that setting pause flag stops optimization before target trials.

        Consolidated test that verifies:
        - Pause triggered during trial stops optimization
        - Pause detected after N trials where N < target
        """
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        call_count = [0]

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test_pause_optimize"
        )

        def test_objective(trial):
            call_count[0] += 1
            # Set pause after 3 trials (target is 10)
            if call_count[0] >= 3:
                optimizer._pause_requested = True
            return 0.1 * call_count[0]

        with patch.object(optimizer, 'underlying_optimizer', create=True) as mock_opt:
            mock_opt.create_objective.return_value = test_objective
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_save:
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                    mock_save.return_value = True
                    mock_local.return_value = True

                    # Run optimization with 10 trials (should stop at 3)
                    study = optimizer.optimize(n_trials=10)

                    # Should have paused after 3 trials, not run all 10
                    assert call_count[0] >= 3
                    assert call_count[0] < 10, "Pause should have stopped optimization early"

    def test_optimization_runs_all_trials_without_pause(self):
        """Test that optimization runs all trials when pause is disabled or not triggered."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        call_count = [0]

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=False,  # Pause disabled
            wandb_project=None,
            study_name="test_no_pause"
        )

        def test_objective(trial):
            call_count[0] += 1
            time.sleep(0.01)  # Small delay to test thread safety
            return call_count[0] * 0.1

        with patch.object(optimizer, 'underlying_optimizer', create=True) as mock_opt:
            mock_opt.create_objective.return_value = test_objective
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_save:
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                    mock_save.return_value = True
                    mock_local.return_value = True

                    # Run all 5 trials
                    study = optimizer.optimize(n_trials=5)
                    assert call_count[0] == 5


def test_resume_command_includes_original_cli(monkeypatch):
    """Test that resume command includes original CLI arguments."""
    from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

    # Set up original argv
    original_argv = ['script.py', '--n-trials', '100', '--wandb', 'my-project']
    monkeypatch.setattr('sys.argv', original_argv)

    optimizer = PausibleOptunaOptimizer(
        base_config={},
        search_space=lambda trial: {},
        model_class=MagicMock,
        enable_pause=True,
        wandb_project="my-project",
        study_name="test"
    )

    # Verify original argv was captured
    assert optimizer._original_argv == original_argv


class TestPauseWithPrunedTrials:
    """Test pause functionality works with pruned trials."""

    def test_pause_after_pruned_trial(self):
        """Test that pause check runs after pruned trials.

        Regression test: Previously, with restart_every_trial=True,
        sys.exit(42) was called BEFORE the pause check, making 'p' ineffective.
        """
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer
        import optuna

        # Track trial execution
        trial_results = []

        def test_objective(trial):
            trial_num = len(trial_results) + 1
            # Prune trials 2 and 3
            if trial_num in [2, 3]:
                trial_results.append(('pruned', trial_num))
                raise optuna.TrialPruned()
            trial_results.append(('complete', trial_num))
            return trial_num * 0.1

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test_pruned_pause",
            restart_on_save=False,  # Disable restart for this test
        )

        with patch.object(optimizer, 'underlying_optimizer', create=True) as mock_opt:
            mock_opt.create_objective.return_value = test_objective
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_wandb:
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                    mock_wandb.return_value = True
                    mock_local.return_value = True

                    # Set pause after trial 2 (which will be pruned)
                    original_update = optimizer._update_pause_from_keyboard
                    def mock_update():
                        # Return True (pause) after trial 2
                        if len(trial_results) >= 2:
                            return True
                        return original_update()

                    optimizer._update_pause_from_keyboard = mock_update

                    # Run optimization
                    study = optimizer.optimize(n_trials=10)

                    # Should have paused after trial 2 (pruned)
                    assert len(trial_results) == 2, f"Should pause after 2 trials, got {len(trial_results)}"
                    assert trial_results[1] == ('pruned', 2), "Second trial should be pruned"
                    assert optimizer.should_pause, "should_pause flag should be True"

    def test_pause_check_runs_before_restart(self):
        """Test that pause check runs BEFORE restart logic.

        This verifies the fix: pause check must happen before sys.exit(42).
        """
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer
        import optuna

        trial_count = [0]
        exit_called = [False]
        pause_checked = [False]

        def test_objective(trial):
            trial_count[0] += 1
            return 0.1

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test_pause_before_restart",
            restart_on_save=True,  # Enable restart
            restart_every_trial=True,  # Restart after every trial
        )

        with patch.object(optimizer, 'underlying_optimizer', create=True) as mock_opt:
            mock_opt.create_objective.return_value = test_objective
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_wandb:
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                    mock_wandb.return_value = True
                    mock_local.return_value = True

                    # Track if pause check is called
                    original_update = optimizer._update_pause_from_keyboard
                    def mock_update():
                        pause_checked[0] = True
                        return True  # Always request pause

                    optimizer._update_pause_from_keyboard = mock_update

                    # Patch sys.exit to not actually exit but track if called
                    with patch('sys.exit') as mock_exit:
                        def track_exit(code):
                            exit_called[0] = True
                            # Don't actually exit, just track
                        mock_exit.side_effect = track_exit

                        # Run optimization
                        study = optimizer.optimize(n_trials=5)

                        # Pause check should have been called
                        assert pause_checked[0], "Pause check should be called"
                        # sys.exit should NOT have been called because pause was triggered first
                        assert not exit_called[0], "sys.exit should not be called when pause is requested"
                        # Should have paused
                        assert optimizer.should_pause, "Optimizer should be in paused state"


class TestPauseExitBehavior:
    """Test that pause causes clean exit (code 0), not restart (code 42)."""

    def test_pause_exits_with_code_0_not_42(self):
        """Test that when pause is triggered, subprocess exits with 0, not 42.

        This is a critical regression test. The bug was:
        - With restart_every_trial=True, after each trial sys.exit(42) was called
        - This caused the parent process to restart the subprocess
        - Even after pressing 'p', trials kept repeating

        The fix ensures pause check runs BEFORE restart logic, and when pause
        is triggered, the function returns normally (exit 0) instead of calling
        sys.exit(42).
        """
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer
        import sys

        trial_count = [0]
        exit_codes = []

        def test_objective(trial):
            trial_count[0] += 1
            return 0.1 * trial_count[0]

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test_exit_code",
            restart_on_save=True,
            restart_every_trial=True,
        )

        with patch.object(optimizer, 'underlying_optimizer', create=True) as mock_opt:
            mock_opt.create_objective.return_value = test_objective
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_wandb:
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                    mock_wandb.return_value = True
                    mock_local.return_value = True

                    # Simulate pause being pressed after first trial
                    def mock_pause_check():
                        return trial_count[0] >= 1  # Pause after first trial

                    optimizer._update_pause_from_keyboard = mock_pause_check

                    # Track sys.exit calls
                    with patch.object(sys, 'exit') as mock_exit:
                        def track_exit(code):
                            exit_codes.append(code)
                            # Raise SystemExit to stop execution
                            raise SystemExit(code)
                        mock_exit.side_effect = track_exit

                        # Run optimization - should NOT call sys.exit(42)
                        try:
                            study = optimizer.optimize(n_trials=10)
                        except SystemExit as e:
                            # If we get here with code 42, the test fails
                            pytest.fail(f"sys.exit({e.code}) was called - pause should prevent restart!")

                        # Function returned normally (no sys.exit called)
                        assert len(exit_codes) == 0, f"sys.exit was called with codes: {exit_codes}"
                        assert optimizer.should_pause, "Optimizer should be in paused state"
                        assert trial_count[0] == 1, f"Should have run exactly 1 trial before pause, got {trial_count[0]}"

    def test_normal_restart_calls_exit_42(self):
        """Test that without pause, restart_every_trial causes sys.exit(42)."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer
        import sys

        trial_count = [0]
        exit_code = [None]

        def test_objective(trial):
            trial_count[0] += 1
            return 0.1 * trial_count[0]

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test_restart_exit",
            restart_on_save=True,
            restart_every_trial=True,
        )

        with patch.object(optimizer, 'underlying_optimizer', create=True) as mock_opt:
            mock_opt.create_objective.return_value = test_objective
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_wandb:
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                    mock_wandb.return_value = True
                    mock_local.return_value = True

                    # NO pause - should trigger restart
                    optimizer._update_pause_from_keyboard = lambda: False

                    with patch.object(sys, 'exit') as mock_exit:
                        def track_exit(code):
                            exit_code[0] = code
                            raise SystemExit(code)
                        mock_exit.side_effect = track_exit

                        # Run optimization - should call sys.exit(42) after first trial
                        with pytest.raises(SystemExit) as exc_info:
                            optimizer.optimize(n_trials=10)

                        assert exc_info.value.code == 42, f"Expected exit code 42 for restart, got {exc_info.value.code}"
                        assert trial_count[0] == 1, "Should have run exactly 1 trial before restart"


def test_keyboard_interrupt_propagates(monkeypatch):
    """Test that KeyboardInterrupt propagates correctly."""
    from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer
    import optuna

    call_count = [0]
    def test_objective(trial):
        call_count[0] += 1
        if call_count[0] >= 2:
            raise KeyboardInterrupt()
        return call_count[0] * 0.1

    optimizer = PausibleOptunaOptimizer(
        base_config={},
        search_space=lambda trial: {},
        model_class=MagicMock,
        enable_pause=True,
        wandb_project=None,
        study_name="test_keyboard_interrupt"
    )

    with patch.object(optimizer, 'underlying_optimizer', create=True) as mock_opt:
        mock_opt.create_objective.return_value = test_objective
        with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_save:
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                mock_save.return_value = True
                mock_local.return_value = True

                # Should raise KeyboardInterrupt
                with pytest.raises(KeyboardInterrupt):
                    optimizer.optimize(n_trials=5)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, '-v'])
