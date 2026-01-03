"""
Tests for HPOKeyboardHandler.
"""

import os
import pytest
import threading
import time
from unittest.mock import patch, MagicMock


class TestHPOKeyboardHandler:
    """Tests for HPOKeyboardHandler class."""

    def test_initialization(self):
        """Test handler initializes with default values."""
        from LightningTune.input.keyboard_handler import HPOKeyboardHandler

        handler = HPOKeyboardHandler()

        assert handler.pause_requested is False
        assert handler.quit_requested is False
        assert handler.is_running is False
        assert handler._pause_key == 'p'
        assert handler._quit_key == 'q'

    def test_custom_callbacks(self):
        """Test handler accepts custom callbacks."""
        from LightningTune.input.keyboard_handler import HPOKeyboardHandler

        pause_called = []
        quit_called = []

        def on_pause():
            pause_called.append(True)

        def on_quit():
            quit_called.append(True)

        handler = HPOKeyboardHandler(
            on_pause=on_pause,
            on_quit=on_quit,
            pause_key='x',
            quit_key='z',
        )

        assert handler._pause_key == 'x'
        assert handler._quit_key == 'z'

    def test_pause_requested_thread_safe(self):
        """Test pause_requested property is thread-safe."""
        from LightningTune.input.keyboard_handler import HPOKeyboardHandler

        handler = HPOKeyboardHandler()
        results = []

        def set_pause():
            for _ in range(100):
                handler.pause_requested = True
                results.append(handler.pause_requested)

        def get_pause():
            for _ in range(100):
                results.append(handler.pause_requested)

        t1 = threading.Thread(target=set_pause)
        t2 = threading.Thread(target=get_pause)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # All results should be booleans
        assert all(isinstance(r, bool) for r in results)

    def test_handle_key_input_pause(self):
        """Test handling pause key input."""
        from LightningTune.input.keyboard_handler import HPOKeyboardHandler

        pause_called = []

        handler = HPOKeyboardHandler(on_pause=lambda: pause_called.append(True))
        handler._debounce_interval = 0  # Disable debounce for testing

        # First press - should pause
        result = handler._handle_key_input('p')
        assert result is True
        assert handler.pause_requested is True
        assert len(pause_called) == 1

    def test_handle_key_input_pause_toggle(self):
        """Test pause toggle behavior."""
        from LightningTune.input.keyboard_handler import HPOKeyboardHandler

        cancel_called = []

        handler = HPOKeyboardHandler(
            on_cancel_pause=lambda: cancel_called.append(True),
        )
        handler._debounce_interval = 0

        # First press - pause
        handler._handle_key_input('p')
        assert handler.pause_requested is True

        # Second press - cancel pause
        handler._handle_key_input('p')
        assert handler.pause_requested is False
        assert len(cancel_called) == 1

    def test_handle_key_input_quit(self):
        """Test handling quit key input."""
        from LightningTune.input.keyboard_handler import HPOKeyboardHandler

        quit_called = []

        handler = HPOKeyboardHandler(on_quit=lambda: quit_called.append(True))

        result = handler._handle_key_input('q')
        assert result is True
        assert handler.quit_requested is True
        assert len(quit_called) == 1

    def test_handle_key_input_ctrl_c(self):
        """Test handling Ctrl+C (\\x03) input."""
        from LightningTune.input.keyboard_handler import HPOKeyboardHandler

        pause_called = []

        handler = HPOKeyboardHandler(on_pause=lambda: pause_called.append(True))

        result = handler._handle_key_input('\x03')
        assert result is True
        assert handler.pause_requested is True
        assert len(pause_called) == 1

    def test_handle_key_input_unknown(self):
        """Test handling unknown key input."""
        from LightningTune.input.keyboard_handler import HPOKeyboardHandler

        handler = HPOKeyboardHandler()

        result = handler._handle_key_input('x')
        assert result is False
        assert handler.pause_requested is False
        assert handler.quit_requested is False

    def test_debounce(self):
        """Test key debouncing."""
        from LightningTune.input.keyboard_handler import HPOKeyboardHandler

        handler = HPOKeyboardHandler()
        handler._debounce_interval = 0.1

        # First press
        handler._handle_key_input('p')
        assert handler.pause_requested is True

        # Immediate second press should be debounced (won't toggle)
        handler._handle_key_input('p')
        assert handler.pause_requested is True  # Still paused

        # Wait for debounce
        time.sleep(0.15)

        # Now should work
        handler._handle_key_input('p')
        assert handler.pause_requested is False

    def test_reset_pause(self):
        """Test resetting pause state."""
        from LightningTune.input.keyboard_handler import HPOKeyboardHandler

        handler = HPOKeyboardHandler()
        handler._debounce_interval = 0

        handler._handle_key_input('p')
        assert handler.pause_requested is True

        handler.reset_pause()
        assert handler.pause_requested is False

    def test_is_available_child_process(self):
        """Test availability check for child process."""
        from LightningTune.input.keyboard_handler import HPOKeyboardHandler

        handler = HPOKeyboardHandler()

        # Set environment variable to simulate child process
        with patch.dict(os.environ, {'LT_CHILD': '1'}):
            assert handler.is_available() is False

    def test_is_available_no_tty(self):
        """Test availability check when no TTY."""
        from LightningTune.input.keyboard_handler import HPOKeyboardHandler

        handler = HPOKeyboardHandler()

        with patch('sys.stdin.isatty', return_value=False):
            assert handler.is_available() is False

    def test_stop_without_start(self):
        """Test stopping without starting doesn't error."""
        from LightningTune.input.keyboard_handler import HPOKeyboardHandler

        handler = HPOKeyboardHandler()
        handler.stop()  # Should not raise

    def test_check_pause_convenience_method(self):
        """Test check_pause convenience method."""
        from LightningTune.input.keyboard_handler import HPOKeyboardHandler

        handler = HPOKeyboardHandler()
        handler._debounce_interval = 0

        assert handler.check_pause() is False

        handler._handle_key_input('p')
        assert handler.check_pause() is True

    def test_check_quit_convenience_method(self):
        """Test check_quit convenience method."""
        from LightningTune.input.keyboard_handler import HPOKeyboardHandler

        handler = HPOKeyboardHandler()

        assert handler.check_quit() is False

        handler._handle_key_input('q')
        assert handler.check_quit() is True


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
