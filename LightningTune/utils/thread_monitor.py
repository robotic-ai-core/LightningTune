"""
Thread monitoring utility for debugging resource accumulation in HPO.

Tracks thread count over time and logs warnings when threads accumulate.
Useful for detecting thread leaks during long-running optimization experiments.
"""

import threading
import time
import sys
from typing import Optional, Callable
from datetime import datetime


class ThreadMonitor:
    """
    Monitors thread count and logs warnings on accumulation.

    Features:
    - Periodic thread count logging
    - Automatic warnings when thread count grows
    - Thread name tracking
    - Can run as daemon or blocking

    Example:
        # As context manager
        with ThreadMonitor(interval=30, warn_threshold=10) as monitor:
            run_optimization()

        # Manual start/stop
        monitor = ThreadMonitor(interval=30)
        monitor.start(daemon=True)
        try:
            run_optimization()
        finally:
            monitor.stop()
            monitor.print_summary()
    """

    def __init__(
        self,
        interval: int = 30,
        log_callback: Optional[Callable[[str], None]] = None,
        warn_threshold: int = 10,
    ):
        """
        Initialize thread monitor.

        Args:
            interval: Seconds between checks
            log_callback: Optional callback for log messages (default: print)
            warn_threshold: Warn if thread count exceeds this value
        """
        self.interval = interval
        self.log_callback = log_callback or print
        self.warn_threshold = warn_threshold

        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        self.initial_thread_count = threading.active_count()
        self.max_thread_count = self.initial_thread_count
        self.thread_history = []

    def _log(self, message: str):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.log_callback(f"[{timestamp}] ThreadMonitor: {message}")
        sys.stdout.flush()

    def _monitor_loop(self):
        """Main monitoring loop."""
        self._log(f"Started (baseline: {self.initial_thread_count} threads)")

        while not self._stop_event.is_set():
            time.sleep(self.interval)

            if self._stop_event.is_set():
                break

            # Get current thread info
            thread_count = threading.active_count()
            thread_names = [t.name for t in threading.enumerate()]

            # Track history
            self.thread_history.append({
                "timestamp": datetime.now(),
                "count": thread_count,
                "names": thread_names,
            })

            # Update max
            if thread_count > self.max_thread_count:
                self.max_thread_count = thread_count

            # Log current state
            self._log(f"{thread_count} active threads (max: {self.max_thread_count})")
            self._log(f"Thread names: {', '.join(thread_names)}")

            # Warn if threshold exceeded
            if thread_count > self.warn_threshold:
                self._log(f"WARNING: Thread count ({thread_count}) exceeds threshold ({self.warn_threshold})!")
                self._log(f"   Growth: {thread_count - self.initial_thread_count} threads since start")

            # Detect accumulation pattern
            if len(self.thread_history) >= 3:
                recent_counts = [h["count"] for h in self.thread_history[-3:]]
                if all(recent_counts[i] <= recent_counts[i+1] for i in range(len(recent_counts)-1)):
                    self._log(f"WARNING: Thread count growing consistently: {recent_counts}")

        self._log("Stopped")

    def start(self, daemon: bool = True):
        """
        Start monitoring in background thread.

        Args:
            daemon: If True, thread will not block process exit
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._log("Already running")
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=daemon,
            name="ThreadMonitor"
        )
        self._monitor_thread.start()

    def stop(self):
        """Stop monitoring."""
        if not self._monitor_thread or not self._monitor_thread.is_alive():
            return

        self._stop_event.set()
        self._monitor_thread.join(timeout=2.0)

    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            "initial_count": self.initial_thread_count,
            "current_count": threading.active_count(),
            "max_count": self.max_thread_count,
            "growth": threading.active_count() - self.initial_thread_count,
            "history_length": len(self.thread_history),
        }

    def print_summary(self):
        """Print summary to log."""
        summary = self.get_summary()
        self._log("=" * 60)
        self._log("THREAD MONITOR SUMMARY:")
        self._log(f"  Initial threads: {summary['initial_count']}")
        self._log(f"  Current threads: {summary['current_count']}")
        self._log(f"  Max threads:     {summary['max_count']}")
        self._log(f"  Growth:          +{summary['growth']}")
        self._log(f"  Checks:          {summary['history_length']}")
        self._log("=" * 60)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        self.print_summary()
        return False
