"""
Keyboard handler for HPO pause control.

This module encapsulates keyboard monitoring for interactive pause/resume
functionality during HPO runs. It supports multiple backend implementations
and provides a clean, testable interface.
"""

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Ensure Reflow package is importable when used as a submodule
_reflow_path = Path(__file__).parent.parent.parent.parent / "LightningReflow"
if _reflow_path.exists():
    sys.path.insert(0, str(_reflow_path))

# Import keyboard handlers - separate try blocks so one failure doesn't affect the other
_create_improved_keyboard_handler = None
try:
    from lightning_reflow.callbacks.pause.improved_keyboard_handler import (
        create_improved_keyboard_handler as _create_improved_keyboard_handler,
    )
except Exception:
    pass

_KeyboardHandlerService = None
_KeyboardHandlerStrategy = None
_HAS_KEYBOARD_SERVICE = False
try:
    from lightning_reflow.services import KeyboardHandlerService as _KeyboardHandlerService
    from lightning_reflow.services import KeyboardHandlerStrategy as _KeyboardHandlerStrategy
    _HAS_KEYBOARD_SERVICE = True
except Exception:
    pass


class HPOKeyboardHandler:
    """
    Encapsulates keyboard monitoring for HPO pause control.

    This class provides a unified interface for keyboard input handling,
    abstracting away the differences between various backend implementations
    (KeyboardHandlerService, ImprovedKeyboardHandler, etc.).

    Example:
        >>> def on_pause():
        ...     print("Pause requested!")
        >>> def on_quit():
        ...     print("Quit requested!")
        >>> handler = HPOKeyboardHandler(on_pause=on_pause, on_quit=on_quit)
        >>> if handler.start():
        ...     # Keyboard monitoring is active
        ...     pass
        >>> handler.stop()

    Attributes:
        on_pause: Callback invoked when pause is requested.
        on_quit: Callback invoked when quit is requested.
        on_cancel_pause: Callback invoked when pause is cancelled.
    """

    def __init__(
        self,
        on_pause: Optional[Callable[[], None]] = None,
        on_quit: Optional[Callable[[], None]] = None,
        on_cancel_pause: Optional[Callable[[], None]] = None,
        pause_key: str = 'p',
        quit_key: str = 'q',
        debounce_interval: float = 0.3,
        log_file: str = "/tmp/hpo_pause.log",
    ):
        """
        Initialize keyboard handler.

        Args:
            on_pause: Callback when pause is requested (first 'p' press).
            on_quit: Callback when quit is requested ('q' press).
            on_cancel_pause: Callback when pause is cancelled (second 'p' press).
            pause_key: Key to trigger pause toggle (default 'p').
            quit_key: Key to trigger quit (default 'q').
            debounce_interval: Minimum seconds between key presses.
            log_file: Path to log file for pause events.
        """
        self._on_pause = on_pause
        self._on_quit = on_quit
        self._on_cancel_pause = on_cancel_pause
        self._pause_key = pause_key.lower()
        self._quit_key = quit_key.lower()
        self._debounce_interval = debounce_interval
        self._log_file = log_file

        # State
        self._pause_requested = False
        self._quit_requested = False
        self._lock = threading.Lock()
        self._last_key_time = 0.0
        self._is_running = False

        # Backend references
        self._keyboard_service = None
        self._keyboard_handler = None
        self._polling_thread = None
        self._polling_active = False
        self._use_service = False

    @property
    def pause_requested(self) -> bool:
        """Whether pause has been requested (thread-safe)."""
        with self._lock:
            return self._pause_requested

    @pause_requested.setter
    def pause_requested(self, value: bool):
        """Set pause request state (thread-safe)."""
        with self._lock:
            self._pause_requested = value

    @property
    def quit_requested(self) -> bool:
        """Whether quit has been requested (thread-safe)."""
        with self._lock:
            return self._quit_requested

    @property
    def is_running(self) -> bool:
        """Whether keyboard monitoring is active."""
        return self._is_running

    def is_available(self) -> bool:
        """Check if keyboard input is available (e.g., running in TTY)."""
        # Skip if running as child process in subprocess restart mode
        if os.environ.get("LT_CHILD", "0") == "1":
            return False

        # Check for TTY
        try:
            if not sys.stdin.isatty():
                return False
        except Exception:
            return False

        return True

    def start(self) -> bool:
        """
        Start keyboard monitoring.

        Returns:
            True if monitoring started successfully, False otherwise.
        """
        if self._is_running:
            return True

        if not self.is_available():
            logger.debug("Keyboard monitoring not available (no TTY or child process)")
            return False

        # Try KeyboardHandlerService first (preferred, no polling thread needed)
        if _HAS_KEYBOARD_SERVICE and _KeyboardHandlerService is not None:
            try:
                self._keyboard_service = _KeyboardHandlerService.get_instance(
                    strategy=_KeyboardHandlerStrategy.IMPROVED_MODE,
                    debounce_interval=self._debounce_interval,
                )
                if self._keyboard_service.is_available():
                    self._keyboard_service.register_subscriber("hpo_pause", self._on_key_press)
                    self._use_service = True
                    self._is_running = True
                    logger.info("✅ Using KeyboardHandlerService for HPO pause (eliminates duplicate threads)")
                    return True
            except Exception as e:
                logger.debug(f"KeyboardHandlerService failed: {e}")

        # Fall back to ImprovedKeyboardHandler with polling
        if _create_improved_keyboard_handler is not None:
            try:
                self._keyboard_handler = _create_improved_keyboard_handler()
                if hasattr(self._keyboard_handler, 'is_available'):
                    if not self._keyboard_handler.is_available():
                        logger.warning("⚠️  Keyboard monitoring unavailable (no TTY)")
                        return False

                if hasattr(self._keyboard_handler, 'start_monitoring'):
                    self._keyboard_handler.start_monitoring()

                # Start polling thread for legacy handler
                self._start_polling_thread()
                self._is_running = True
                logger.info("⌨️  Using ImprovedKeyboardHandler with polling thread")
                return True
            except Exception as e:
                logger.warning(f"⚠️  Keyboard monitoring failed to start: {e}")
                return False

        logger.warning("⚠️  No keyboard handler available")
        return False

    def stop(self):
        """Stop keyboard monitoring and clean up resources."""
        if not self._is_running:
            return

        self._is_running = False

        # Unregister from service
        if self._use_service and self._keyboard_service:
            try:
                self._keyboard_service.unregister_subscriber("hpo_pause")
                logger.debug("Unregistered from KeyboardHandlerService")
            except Exception:
                pass
            self._keyboard_service = None

        # Stop legacy handler
        if self._keyboard_handler:
            if hasattr(self._keyboard_handler, 'stop_monitoring'):
                try:
                    self._keyboard_handler.stop_monitoring()
                except Exception:
                    pass
            self._keyboard_handler = None

        # Stop polling thread
        self._stop_polling_thread()

        # Reset state
        with self._lock:
            self._pause_requested = False
            self._quit_requested = False

    def reset_pause(self):
        """Reset pause state without stopping monitoring."""
        with self._lock:
            self._pause_requested = False

    def _on_key_press(self, key: str):
        """
        Handle a key press (callback from KeyboardHandlerService).

        This method is thread-safe and handles debouncing.
        """
        self._handle_key_input(key)

    def _handle_key_input(self, key: str) -> bool:
        """
        Core key handling logic with thread-safe state management.

        Args:
            key: The key that was pressed.

        Returns:
            True if the key was handled, False otherwise.
        """
        skey = key.lower() if isinstance(key, str) else str(key).lower()

        if skey == self._pause_key:
            with self._lock:
                # Debounce
                current_time = time.time()
                if current_time - self._last_key_time < self._debounce_interval:
                    return True
                self._last_key_time = current_time

                was_paused = self._pause_requested
                self._pause_requested = not self._pause_requested
                is_paused = self._pause_requested

            # Invoke callbacks outside lock
            if is_paused and not was_paused:
                msg = "\n⏸️  Pause SCHEDULED ('p' pressed)"
                print(msg, flush=True)
                self._log_to_file(msg)
                if self._on_pause:
                    self._on_pause()
            elif not is_paused and was_paused:
                msg = "\n❌ Pause CANCELLED ('p' pressed again)"
                print(msg, flush=True)
                self._log_to_file(msg)
                if self._on_cancel_pause:
                    self._on_cancel_pause()
            return True

        elif skey == self._quit_key:
            with self._lock:
                self._quit_requested = True
            msg = "\n🛑 Quit requested ('q' pressed). Will stop after current trial."
            print(msg, flush=True)
            self._log_to_file(msg)
            if self._on_quit:
                self._on_quit()
            return True

        elif key == "\x03":  # Ctrl+C in cbreak mode
            with self._lock:
                self._pause_requested = True
            msg = "\n⏸️  Ctrl+C detected. Pausing gracefully at trial boundary..."
            print(msg, flush=True)
            self._log_to_file(msg)
            if self._on_pause:
                self._on_pause()
            return True

        return False

    def _log_to_file(self, msg: str):
        """Log message to pause log file for visibility."""
        try:
            with open(self._log_file, "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {msg.strip()}\n")
                f.flush()
        except Exception:
            pass

    def _start_polling_thread(self):
        """Start background polling thread for legacy keyboard handler."""
        if self._polling_thread and self._polling_thread.is_alive():
            return
        self._polling_active = True
        self._polling_thread = threading.Thread(
            target=self._polling_loop,
            daemon=True,
            name="HPOKeyboardPoller",
        )
        self._polling_thread.start()

    def _stop_polling_thread(self):
        """Stop the background polling thread."""
        self._polling_active = False
        if self._polling_thread and self._polling_thread.is_alive():
            try:
                self._polling_thread.join(timeout=1.0)
            except Exception:
                pass
        self._polling_thread = None

    def _polling_loop(self):
        """Continuously poll keyboard handler for input."""
        while self._polling_active:
            try:
                if self._keyboard_handler and hasattr(self._keyboard_handler, 'get_key'):
                    key = self._keyboard_handler.get_key()
                    if key:
                        self._handle_key_input(str(key))
            except Exception:
                pass
            time.sleep(0.05)

    def check_pause(self) -> bool:
        """
        Check if pause is currently requested.

        This is a convenience method that returns the current pause state.
        For legacy code that polls for pause status.

        Returns:
            True if pause is requested.
        """
        return self.pause_requested

    def check_quit(self) -> bool:
        """
        Check if quit is currently requested.

        Returns:
            True if quit is requested.
        """
        return self.quit_requested
