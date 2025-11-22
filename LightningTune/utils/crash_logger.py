"""
Crash-Resistant Circular Buffer Logger

Captures all output in a rolling buffer with frequent flushing to ensure
we can debug crashes that kill the process.

Features:
- Circular buffer (keeps last N lines)
- Frequent flushing (every write)
- Thread-safe
- Captures stdout/stderr
- Timestamps on every line
- Multiple output files (main log + circular buffer)
- No size limits on main log during run
"""

import sys
import os
import threading
import time
import atexit
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Optional, TextIO


class CircularBufferHandler:
    """Thread-safe circular buffer that keeps last N lines."""

    def __init__(self, max_lines: int = 1000):
        self.max_lines = max_lines
        self.buffer = deque(maxlen=max_lines)
        self.lock = threading.Lock()

    def write(self, line: str):
        """Add line to circular buffer."""
        with self.lock:
            self.buffer.append(line)

    def get_lines(self) -> list:
        """Get all lines from buffer."""
        with self.lock:
            return list(self.buffer)

    def save_to_file(self, filepath: str):
        """Save buffer contents to file."""
        lines = self.get_lines()
        with open(filepath, 'w', buffering=1) as f:  # Line buffered
            for line in lines:
                f.write(line)
            f.flush()
            os.fsync(f.fileno())  # Force OS to write to disk


class CrashResistantLogger:
    """
    Captures all output with circular buffer and frequent flushing.

    Creates two files:
    1. Full log: Complete output (can grow large)
    2. Circular log: Last N lines only (fixed size)

    Both are flushed after every write to survive crashes.
    """

    def __init__(
        self,
        log_dir: str = "/tmp/hpo_crash_logs",
        prefix: str = "hpo",
        max_buffer_lines: int = 1000,
        flush_every_line: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{prefix}_{timestamp}_{os.getpid()}"

        # File paths
        self.full_log_path = self.log_dir / f"{self.run_id}_full.log"
        self.circular_log_path = self.log_dir / f"{self.run_id}_last_{max_buffer_lines}.log"
        self.metadata_path = self.log_dir / f"{self.run_id}_metadata.txt"

        # Circular buffer
        self.circular_buffer = CircularBufferHandler(max_buffer_lines)

        # Open files with minimal buffering
        self.full_log_file = open(self.full_log_path, 'w', buffering=1)  # Line buffered
        self.flush_every_line = flush_every_line

        # Original stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Thread safety
        self.lock = threading.Lock()

        # Track if we're active
        self.active = False

        # Write metadata
        self._write_metadata()

        # Register atexit handler for final buffer save
        atexit.register(self._atexit_handler)

        print(f"CrashResistantLogger initialized:")
        print(f"   Full log: {self.full_log_path}")
        print(f"   Circular log: {self.circular_log_path}")
        print(f"   Buffer size: {max_buffer_lines} lines")

    def _write_metadata(self):
        """Write run metadata."""
        with open(self.metadata_path, 'w') as f:
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"PID: {os.getpid()}\n")
            f.write(f"Start Time: {datetime.now().isoformat()}\n")
            f.write(f"Full Log: {self.full_log_path}\n")
            f.write(f"Circular Log: {self.circular_log_path}\n")
            f.write(f"Command: {' '.join(sys.argv)}\n")
            f.write(f"CWD: {os.getcwd()}\n")
            f.flush()
            os.fsync(f.fileno())

    def _timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def write(self, text: str):
        """Write text to all outputs."""
        if not text:
            return

        # Add timestamp to each line
        timestamp = self._timestamp()
        lines = text.splitlines(keepends=True)

        with self.lock:
            for line in lines:
                if line.strip():  # Only timestamp non-empty lines
                    timestamped_line = f"[{timestamp}] {line}"
                else:
                    timestamped_line = line

                # Write to full log
                self.full_log_file.write(timestamped_line)

                # Write to circular buffer
                self.circular_buffer.write(timestamped_line)

                # Write to original stdout
                self.original_stdout.write(timestamped_line)

            # Flush everything
            if self.flush_every_line:
                self.full_log_file.flush()
                os.fsync(self.full_log_file.fileno())  # Force OS to write to disk
                self.original_stdout.flush()

                # Save circular buffer to disk
                self.circular_buffer.save_to_file(self.circular_log_path)

    def flush(self):
        """Explicit flush."""
        with self.lock:
            self.full_log_file.flush()
            os.fsync(self.full_log_file.fileno())
            self.original_stdout.flush()
            self.circular_buffer.save_to_file(self.circular_log_path)

    def isatty(self) -> bool:
        """
        Return False since we're logging to a file, not a terminal.

        This method is called by libraries like tqdm, rich, and wandb
        to check if they should display progress bars.
        """
        return False

    def fileno(self) -> int:
        """Return the file descriptor of the original stdout."""
        return self.original_stdout.fileno()

    @property
    def encoding(self) -> str:
        """Return the encoding of the original stdout."""
        return getattr(self.original_stdout, 'encoding', 'utf-8')

    @property
    def errors(self) -> str:
        """Return the error handling mode of the original stdout."""
        return getattr(self.original_stdout, 'errors', 'strict')

    def start(self):
        """Start capturing stdout/stderr."""
        if self.active:
            return

        self.active = True

        # Redirect stdout/stderr to this logger
        sys.stdout = self
        sys.stderr = self

        self.write(f"\n{'='*80}\n")
        self.write(f"CrashResistantLogger STARTED - {self._timestamp()}\n")
        self.write(f"{'='*80}\n\n")
        self.flush()

    def stop(self):
        """Stop capturing and restore original stdout/stderr."""
        if not self.active:
            return

        self.write(f"\n{'='*80}\n")
        self.write(f"CrashResistantLogger STOPPED - {self._timestamp()}\n")
        self.write(f"{'='*80}\n\n")
        self.flush()

        # Restore original stdout/stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        self.active = False

        # Close full log file
        self.full_log_file.close()

        print(f"Logs saved:")
        print(f"   Full: {self.full_log_path}")
        print(f"   Last {self.circular_buffer.max_lines} lines: {self.circular_log_path}")

    def _atexit_handler(self):
        """Final save of circular buffer on process exit."""
        try:
            if self.active:
                self.circular_buffer.save_to_file(self.circular_log_path)

                # Write exit marker
                with open(self.metadata_path, 'a') as f:
                    f.write(f"Exit Time: {datetime.now().isoformat()}\n")
                    f.write(f"Exit Method: atexit\n")
                    f.flush()
                    os.fsync(f.fileno())
        except Exception:
            # Ignore all errors in atexit handler
            pass

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            # Log exception before stopping
            self.write(f"\n{'!'*80}\n")
            self.write(f"EXCEPTION CAUGHT: {exc_type.__name__}\n")
            self.write(f"{exc_val}\n")
            self.write(f"{'!'*80}\n\n")
            self.flush()

        self.stop()
        return False  # Don't suppress exception


class TeeLogger:
    """
    Simple tee that writes to multiple outputs.
    Useful for capturing specific streams.
    """

    def __init__(self, *outputs: TextIO):
        self.outputs = outputs

    def write(self, text: str):
        for output in self.outputs:
            output.write(text)

    def flush(self):
        for output in self.outputs:
            output.flush()


def setup_crash_resistant_logging(
    log_dir: str = "/tmp/hpo_crash_logs",
    prefix: str = "hpo",
    max_buffer_lines: int = 1000,
    auto_start: bool = True,
) -> CrashResistantLogger:
    """
    Convenience function to set up crash-resistant logging.

    Args:
        log_dir: Directory to store logs
        prefix: Prefix for log filenames
        max_buffer_lines: Number of lines to keep in circular buffer
        auto_start: If True, start capturing immediately

    Returns:
        CrashResistantLogger instance

    Example:
        >>> logger = setup_crash_resistant_logging()
        >>> # Now all print() statements are captured and flushed
        >>> print("This will be captured")
        >>> # When process crashes, last N lines are preserved
    """
    logger = CrashResistantLogger(
        log_dir=log_dir,
        prefix=prefix,
        max_buffer_lines=max_buffer_lines,
        flush_every_line=True,
    )

    if auto_start:
        logger.start()

    return logger
