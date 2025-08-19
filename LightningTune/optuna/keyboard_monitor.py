"""
Simple keyboard monitor for 'p' key pause functionality.

This module provides a cross-platform keyboard monitor that watches for 'p' key presses
to trigger pause in optimization loops.
"""

import sys
import threading
import time
from queue import Queue, Empty
import logging

logger = logging.getLogger(__name__)

# Platform-specific imports
try:
    import termios
    import tty
    import select
    HAS_UNIX_TERMINAL = True
except ImportError:
    HAS_UNIX_TERMINAL = False

try:
    import msvcrt
    HAS_WINDOWS_TERMINAL = True
except ImportError:
    HAS_WINDOWS_TERMINAL = False


class KeyboardMonitor:
    """
    Cross-platform keyboard monitor for detecting 'p' key presses.
    
    When keyboard monitoring is not available, pause functionality is disabled.
    Ctrl+C maintains its standard behavior of terminating the program.
    """
    
    def __init__(self, pause_key: str = 'p'):
        """
        Initialize the keyboard monitor.
        
        Args:
            pause_key: Key to trigger pause (default 'p')
        """
        self.pause_key = pause_key.lower()
        self._monitoring = False
        self._monitor_thread = None
        self._pause_requested = False
        self._original_settings = None
        self._stop_event = threading.Event()
    
    def is_available(self) -> bool:
        """Check if keyboard monitoring is available on this platform."""
        if HAS_UNIX_TERMINAL:
            return sys.stdin.isatty()
        elif HAS_WINDOWS_TERMINAL:
            return True
        return False
    
    def start(self) -> bool:
        """
        Start keyboard monitoring.
        
        Returns:
            True if monitoring started successfully, False otherwise
        """
        if not self.is_available() or self._monitoring:
            return False
        
        try:
            # Setup terminal for Unix systems
            if HAS_UNIX_TERMINAL and sys.stdin.isatty():
                self._original_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            
            self._monitoring = True
            self._pause_requested = False
            self._stop_event.clear()
            
            # Start monitor thread
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="KeyboardMonitor"
            )
            self._monitor_thread.start()
            
            logger.info(f"⌨️  Keyboard monitoring started (press '{self.pause_key}' to pause)")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to start keyboard monitoring: {e}")
            self._monitoring = False
            return False
    
    def stop(self):
        """Stop keyboard monitoring and restore terminal settings."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._stop_event.set()
        
        # Wait for thread to finish
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        
        # Restore terminal settings for Unix
        if HAS_UNIX_TERMINAL and self._original_settings:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._original_settings)
            except:
                pass  # Ignore restoration errors
    
    def is_pause_requested(self) -> bool:
        """Check if pause has been requested."""
        return self._pause_requested
    
    def clear_pause(self):
        """Clear the pause request flag."""
        self._pause_requested = False
    
    def _monitor_loop(self):
        """Monitor keyboard input for pause key."""
        while self._monitoring and not self._stop_event.is_set():
            try:
                key = self._read_key(timeout=0.1)
                if key and key.lower() == self.pause_key:
                    self._pause_requested = True
                    logger.info(f"\n⏸️  Pause key ('{self.pause_key}') detected!")
                    logger.info("   Will pause after current trial completes (not interrupting mid-trial)...")
            except Exception:
                # Ignore read errors, just continue
                pass
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.01)
    
    def _read_key(self, timeout: float = 0.1) -> str:
        """
        Read a single key with timeout.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            The key pressed or None if timeout
        """
        if HAS_UNIX_TERMINAL and sys.stdin.isatty():
            # Unix/Linux/Mac
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if ready:
                return sys.stdin.read(1)
        elif HAS_WINDOWS_TERMINAL:
            # Windows
            start_time = time.time()
            while time.time() - start_time < timeout:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    # Handle bytes vs string
                    if isinstance(key, bytes):
                        return key.decode('utf-8', errors='ignore')
                    return key
                time.sleep(0.01)
        
        return None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()