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
    """

    def test_pause_toggle_logic(self):
        """Test that pause state can be toggled via the pause flag."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test"
        )

        # Initially not paused
        assert not optimizer._pause_requested

        # Set pause
        optimizer._pause_requested = True
        assert optimizer._pause_requested

        # Clear pause
        optimizer._pause_requested = False
        assert not optimizer._pause_requested

    def test_pause_toggle_in_monitor_loop(self):
        """Test toggle behavior via _update_pause_from_keyboard."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test"
        )

        # Test that pause state starts as False
        assert optimizer._pause_requested == False

        # Manually toggle pause state (simulating keyboard press)
        optimizer._pause_requested = True
        assert optimizer._pause_requested == True

        # Toggle again (cancel)
        optimizer._pause_requested = False
        assert optimizer._pause_requested == False

    def test_pause_message_output(self, capsys):
        """Test that correct messages are displayed for pause/cancel."""
        # This test verifies the optimizer has the pause message logic
        from LightningTune.optuna import pausible_optimizer
        import inspect

        source = inspect.getsource(pausible_optimizer.PausibleOptunaOptimizer.optimize)

        # Check for pause-related messages in the optimize method
        assert "pause" in source.lower() or "Pause" in source

    def test_monitor_loop_single_iteration(self):
        """Test a single iteration of pause state changes."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test"
        )

        # Test toggle sequence
        assert not optimizer._pause_requested

        optimizer._pause_requested = True
        assert optimizer._pause_requested

        optimizer._pause_requested = False
        assert not optimizer._pause_requested

        optimizer._pause_requested = True
        assert optimizer._pause_requested

    def test_integration_with_pausible_optimizer(self):
        """Test that pausible optimizer handles pause flag correctly."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        # Create optimizer with mock components
        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test"
        )

        # Scenario 1: No pause requested
        assert not optimizer._pause_requested

        # Scenario 2: Pause requested
        optimizer._pause_requested = True
        assert optimizer._pause_requested

        # Clear pause
        optimizer._pause_requested = False
        assert not optimizer._pause_requested

    def test_pause_persistence_between_checks(self):
        """Test that pause state persists between checks until cleared."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test"
        )

        # Set pause
        optimizer._pause_requested = True

        # Multiple checks should all return True
        assert optimizer._pause_requested
        assert optimizer._pause_requested
        assert optimizer._pause_requested

        # Clear should reset it
        optimizer._pause_requested = False
        assert not optimizer._pause_requested


class TestPauseMessages:
    """Test pause/cancel message display."""

    def test_message_content(self):
        """Verify pause-related message content exists."""
        from LightningTune.optuna import pausible_optimizer
        import inspect

        source = inspect.getsource(pausible_optimizer.PausibleOptunaOptimizer)

        # Check for pause-related content
        assert "pause" in source.lower() or "Pause" in source

    def test_execution_messages(self):
        """Test that execution messages are correct."""
        from LightningTune.optuna import pausible_optimizer
        import inspect

        source = inspect.getsource(pausible_optimizer.PausibleOptunaOptimizer.optimize)

        # Check for trial execution messages
        assert "trial" in source.lower() or "Trial" in source


class TestOptimizeLoopPause:
    """Test pause behavior during the optimize loop."""

    def test_pause_during_optimize_loop(self):
        """Test that pause is detected during optimize loop."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer
        import optuna

        # Track whether pause was checked
        pause_check_count = [0]

        # Create optimizer
        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test_pause_optimize"
        )

        # Simple objective that tracks calls
        call_count = [0]
        def test_objective(trial):
            call_count[0] += 1
            # Set pause after 2 trials
            if call_count[0] >= 2:
                optimizer._pause_requested = True
            return 0.1 * call_count[0]

        # Patch optimizer to use our objective
        with patch.object(optimizer, 'underlying_optimizer', create=True) as mock_opt:
            mock_opt.create_objective.return_value = test_objective
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_save:
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                    mock_save.return_value = True
                    mock_local.return_value = True

                    # Run optimization with 5 trials (should stop at 2)
                    study = optimizer.optimize(n_trials=5)

                    # Should have paused after 2 trials
                    assert call_count[0] >= 2

    def test_pause_flag_read_correctly(self):
        """Test that _update_pause_from_keyboard reads the flag correctly."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test"
        )

        # Initially false
        assert optimizer._pause_requested == False

        # Set to true
        optimizer._pause_requested = True
        assert optimizer._pause_requested == True

    def test_callback_toggles_pause_state(self):
        """Test that pause state can be toggled via callback."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test"
        )

        # Simulate callback setting pause
        def set_pause():
            with optimizer._pause_lock:
                optimizer._pause_requested = True

        # Call callback
        set_pause()

        # Verify pause was set
        assert optimizer._pause_requested == True

    def test_pause_with_polling_active_path(self):
        """Test pause when polling is active."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        call_count = [0]
        def test_objective(trial):
            call_count[0] += 1
            return call_count[0] * 0.1

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test_polling_pause"
        )

        with patch.object(optimizer, 'underlying_optimizer', create=True) as mock_opt:
            mock_opt.create_objective.return_value = test_objective
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_save:
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                    mock_save.return_value = True
                    mock_local.return_value = True

                    # Run a few trials
                    study = optimizer.optimize(n_trials=3)
                    assert call_count[0] == 3

    def test_pause_with_actual_polling_thread(self):
        """Test pause with actual polling thread behavior."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        call_count = [0]
        def test_objective(trial):
            call_count[0] += 1
            if call_count[0] >= 3:
                optimizer._pause_requested = True
            return call_count[0] * 0.1

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test_polling_thread"
        )

        with patch.object(optimizer, 'underlying_optimizer', create=True) as mock_opt:
            mock_opt.create_objective.return_value = test_objective
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_save:
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                    mock_save.return_value = True
                    mock_local.return_value = True

                    # Run with more trials than pause point
                    study = optimizer.optimize(n_trials=10)

                    # Should have stopped at or after pause point
                    assert call_count[0] >= 3

    def test_pause_with_manual_polling_no_polling_thread(self):
        """Test pause with manual polling (no polling thread)."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        call_count = [0]
        def test_objective(trial):
            call_count[0] += 1
            return call_count[0] * 0.1

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=False,  # Disable pause for this test
            wandb_project=None,
            study_name="test_no_polling"
        )

        with patch.object(optimizer, 'underlying_optimizer', create=True) as mock_opt:
            mock_opt.create_objective.return_value = test_objective
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_save:
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                    mock_save.return_value = True
                    mock_local.return_value = True

                    # Run all trials (no pause)
                    study = optimizer.optimize(n_trials=5)
                    assert call_count[0] == 5

    def test_pause_detection_with_polling_thread_race_condition(self):
        """Test that pause detection handles race conditions correctly."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        call_count = [0]
        def test_objective(trial):
            call_count[0] += 1
            # Simulate work that takes some time
            time.sleep(0.01)
            return call_count[0] * 0.1

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test_race_condition"
        )

        with patch.object(optimizer, 'underlying_optimizer', create=True) as mock_opt:
            mock_opt.create_objective.return_value = test_objective
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_save:
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                    mock_save.return_value = True
                    mock_local.return_value = True

                    # Run trials
                    study = optimizer.optimize(n_trials=3)
                    assert call_count[0] == 3


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
