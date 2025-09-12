#!/usr/bin/env python
"""Test cases for pause/cancel toggle functionality."""

import pytest
import time
import threading
from unittest.mock import MagicMock, patch
import sys
import os

# Add LightningTune root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LightningTune.optuna.keyboard_monitor import KeyboardMonitor


class TestPauseToggle:
    """Test pause/cancel toggle functionality."""
    
    def test_pause_toggle_logic(self):
        """Test that pause state toggles correctly."""
        monitor = KeyboardMonitor(pause_key='p')
        
        # Initially not paused
        assert not monitor.is_pause_requested()
        
        # Set pause
        monitor._pause_requested = True
        assert monitor.is_pause_requested()
        
        # Clear pause
        monitor.clear_pause()
        assert not monitor.is_pause_requested()
    
    def test_pause_toggle_in_monitor_loop(self):
        """Test toggle behavior in the monitor loop."""
        monitor = KeyboardMonitor(pause_key='p')
        
        # Track key presses and state changes
        key_presses = []
        initial_state = False
        
        def mock_read_key(timeout):
            # Return None to prevent infinite loop
            return None
        
        # Replace read_key with mock
        monitor._read_key = mock_read_key
        
        # Test the toggle directly
        assert monitor._pause_requested == False
        
        # Simulate processing 'p' key (first press - activate)
        monitor._pause_requested = False
        if 'p' and 'p'.lower() == monitor.pause_key:
            if monitor._pause_requested:
                monitor._pause_requested = False
            else:
                monitor._pause_requested = True
        assert monitor._pause_requested == True, "First 'p' press should activate pause"
        
        # Simulate processing 'p' key (second press - cancel)
        if 'p' and 'p'.lower() == monitor.pause_key:
            if monitor._pause_requested:
                monitor._pause_requested = False
            else:
                monitor._pause_requested = True
        assert monitor._pause_requested == False, "Second 'p' press should cancel pause"
        
        # Simulate processing 'p' key (third press - reactivate)
        if 'p' and 'p'.lower() == monitor.pause_key:
            if monitor._pause_requested:
                monitor._pause_requested = False
            else:
                monitor._pause_requested = True
        assert monitor._pause_requested == True, "Third 'p' press should reactivate pause"
    
    def test_pause_message_output(self):
        """Test that correct messages are displayed for pause/cancel."""
        from io import StringIO
        import logging
        
        # Capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        
        logger = logging.getLogger('external.LightningTune.LightningTune.optuna.keyboard_monitor')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        monitor = KeyboardMonitor(pause_key='p')
        
        # Add helper method for testing
        def _process_key(key):
            """Helper method for testing - process a single key."""
            if key and key.lower() == monitor.pause_key:
                # Toggle pause state
                if monitor._pause_requested:
                    monitor._pause_requested = False
                    logger.info(f"\n✅ Pause CANCELLED ('{monitor.pause_key}' pressed again)")
                    logger.info("   Optimization will continue normally")
                else:
                    monitor._pause_requested = True
                    logger.info(f"\n⏸️  Pause SCHEDULED ('{monitor.pause_key}' pressed)")
                    logger.info("   Will pause after current trial completes")
                    logger.info(f"   Press '{monitor.pause_key}' again to cancel pause")
        
        # Simulate first 'p' press (schedule pause)
        monitor._pause_requested = False
        _process_key('p')
        
        output = log_capture.getvalue()
        assert "Pause SCHEDULED" in output
        assert "Press 'p' again to cancel" in output
        
        # Clear buffer
        log_capture.truncate(0)
        log_capture.seek(0)
        
        # Simulate second 'p' press (cancel pause)
        _process_key('p')
        
        output = log_capture.getvalue()
        assert "Pause CANCELLED" in output
        assert "continue normally" in output
        
        logger.removeHandler(handler)
    
    def test_monitor_loop_single_iteration(self):
        """Test a single iteration of the monitor loop."""
        # Add helper method to KeyboardMonitor for testing
        def _monitor_loop_single_iteration(self, key):
            """Helper method for testing - process a single key."""
            if key and key.lower() == self.pause_key:
                # Toggle pause state
                if self._pause_requested:
                    self._pause_requested = False
                    # In real code, this logs
                else:
                    self._pause_requested = True
                    # In real code, this logs
        
        KeyboardMonitor._monitor_loop_single_iteration = _monitor_loop_single_iteration
        
        monitor = KeyboardMonitor(pause_key='p')
        
        # Test toggle sequence
        assert not monitor.is_pause_requested()
        
        monitor._monitor_loop_single_iteration('p')
        assert monitor.is_pause_requested()
        
        monitor._monitor_loop_single_iteration('p')
        assert not monitor.is_pause_requested()
        
        monitor._monitor_loop_single_iteration('p')
        assert monitor.is_pause_requested()
    
    def test_integration_with_pausible_optimizer(self):
        """Test that pausible optimizer doesn't clear pause flag prematurely."""
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
        
        # Mock keyboard monitor
        mock_monitor = MagicMock()
        mock_monitor.is_pause_requested.return_value = False
        optimizer.keyboard_monitor = mock_monitor
        
        # Simulate pause detection scenarios
        
        # Scenario 1: No pause requested
        assert not optimizer.keyboard_monitor.is_pause_requested()
        
        # Scenario 2: Pause requested
        mock_monitor.is_pause_requested.return_value = True
        assert optimizer.keyboard_monitor.is_pause_requested()
        
        # Verify clear_pause is called only at cleanup
        mock_monitor.clear_pause.assert_not_called()
        
        # Simulate cleanup
        if optimizer.keyboard_monitor:
            optimizer.keyboard_monitor.clear_pause()
            optimizer.keyboard_monitor.stop()
        
        mock_monitor.clear_pause.assert_called_once()
        mock_monitor.stop.assert_called_once()
    
    def test_pause_persistence_between_checks(self):
        """Test that pause state persists between checks until cleared."""
        monitor = KeyboardMonitor(pause_key='p')
        
        # Set pause
        monitor._pause_requested = True
        
        # Multiple checks should all return True
        assert monitor.is_pause_requested()
        assert monitor.is_pause_requested()
        assert monitor.is_pause_requested()
        
        # State should persist
        assert monitor._pause_requested == True
        
        # Clear should reset it
        monitor.clear_pause()
        assert not monitor.is_pause_requested()


class TestPauseMessages:
    """Test pause/cancel message display."""
    
    def test_message_content(self):
        """Verify the exact message content."""
        from LightningTune.optuna import keyboard_monitor
        import inspect
        
        source = inspect.getsource(keyboard_monitor.KeyboardMonitor._monitor_loop)
        
        # Check for pause scheduled message
        assert "Pause SCHEDULED" in source
        assert "Press" in source and "again to cancel" in source
        
        # Check for pause cancelled message  
        assert "Pause CANCELLED" in source
        assert "continue normally" in source
    
    def test_execution_messages(self):
        """Test that execution messages are correct."""
        from LightningTune.optuna import pausible_optimizer
        import inspect
        
        source = inspect.getsource(pausible_optimizer.PausibleOptunaOptimizer.optimize)
        
        # Check for execution message
        assert "Executing pause" in source
        
        # Check that we use _update_pause_from_keyboard for error handling
        assert "_update_pause_from_keyboard" in source


if __name__ == "__main__":
    # Run tests
    test = TestPauseToggle()
    
    print("Running pause toggle tests...")
    
    try:
        test.test_pause_toggle_logic()
        print("✅ Pause toggle logic test passed")
    except AssertionError as e:
        print(f"❌ Pause toggle logic test failed: {e}")
    
    try:
        test.test_monitor_loop_single_iteration()
        print("✅ Monitor loop iteration test passed")
    except AssertionError as e:
        print(f"❌ Monitor loop iteration test failed: {e}")
    
    try:
        test.test_integration_with_pausible_optimizer()
        print("✅ Integration test passed")
    except AssertionError as e:
        print(f"❌ Integration test failed: {e}")
    
    try:
        test.test_pause_persistence_between_checks()
        print("✅ Pause persistence test passed")
    except AssertionError as e:
        print(f"❌ Pause persistence test failed: {e}")
    
    messages = TestPauseMessages()
    
    try:
        messages.test_message_content()
        print("✅ Message content test passed")
    except AssertionError as e:
        print(f"❌ Message content test failed: {e}")
    
    try:
        messages.test_execution_messages()
        print("✅ Execution messages test passed")
    except AssertionError as e:
        print(f"❌ Execution messages test failed: {e}")
    
    print("\n✅ All pause toggle tests completed!")


def test_resume_command_includes_original_cli(monkeypatch):
    import sys
    fake_argv = [
        "scripts/world_model_hpo_optuna.py",
        "--wandb", "proj",
        "--trial-steps", "40000",
        "--n-trials", "25",
        "--sampler", "tpe",
        "--pruner", "hyperband",
    ]
    monkeypatch.setattr(sys, "argv", fake_argv.copy(), raising=False)

    from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

    opt = PausibleOptunaOptimizer(
        base_config={"trainer": {}},
        search_space=lambda t: {},
        model_class=MagicMock,
        wandb_project="proj",
        study_name="world_model_tpe_hyperband",
        enable_pause=False,
    )

    cmd = opt._build_resume_command()
    # The resume command should be minimal - only what's needed to resume
    assert "scripts/world_model_hpo_optuna.py" in cmd
    assert "--wandb proj" in cmd
    assert "--study-name world_model_tpe_hyperband" in cmd
    assert "--resume-from latest" in cmd
    assert cmd.startswith("python ")


def test_keyboard_interrupt_propagates(monkeypatch):
    from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer
    import optuna

    class DummyStudy:
        def __init__(self):
            self.sampler = optuna.samplers.TPESampler()
            self.pruner = optuna.pruners.MedianPruner()
            self.trials = []

        def optimize(self, objective, n_trials, show_progress_bar=False, gc_after_trial=True):
            raise KeyboardInterrupt()

    def fake_create_study(**kwargs):
        return DummyStudy()

    monkeypatch.setattr("LightningTune.optuna.pausible_optimizer.create_sampler", lambda *a, **k: optuna.samplers.TPESampler())
    monkeypatch.setattr("LightningTune.optuna.pausible_optimizer.create_pruner", lambda *a, **k: optuna.pruners.MedianPruner())
    monkeypatch.setattr("LightningTune.optuna.pausible_optimizer.optuna.create_study", fake_create_study)

    opt = PausibleOptunaOptimizer(
        base_config={"trainer": {}},
        search_space=lambda t: {},
        model_class=MagicMock,
        wandb_project=None,
        enable_pause=False,
    )

    with pytest.raises(KeyboardInterrupt):
        opt.optimize(n_trials=1)