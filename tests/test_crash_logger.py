"""
Tests for crash-resistant logging utilities.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path

from LightningTune.utils import (
    CircularBufferHandler,
    CrashResistantLogger,
    TeeLogger,
    setup_crash_resistant_logging,
)


class TestCircularBufferHandler:
    """Tests for CircularBufferHandler."""

    def test_basic_write(self):
        """Test basic write to buffer."""
        buffer = CircularBufferHandler(max_lines=100)
        buffer.write("line 1\n")
        buffer.write("line 2\n")

        lines = buffer.get_lines()
        assert len(lines) == 2
        assert lines[0] == "line 1\n"
        assert lines[1] == "line 2\n"

    def test_circular_behavior(self):
        """Test that buffer is circular and drops old lines."""
        buffer = CircularBufferHandler(max_lines=3)

        for i in range(5):
            buffer.write(f"line {i}\n")

        lines = buffer.get_lines()
        assert len(lines) == 3
        assert lines[0] == "line 2\n"
        assert lines[1] == "line 3\n"
        assert lines[2] == "line 4\n"

    def test_save_to_file(self):
        """Test saving buffer to file."""
        buffer = CircularBufferHandler(max_lines=100)
        buffer.write("line 1\n")
        buffer.write("line 2\n")

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            filepath = f.name

        try:
            buffer.save_to_file(filepath)

            with open(filepath, 'r') as f:
                content = f.read()

            assert "line 1" in content
            assert "line 2" in content
        finally:
            os.unlink(filepath)

    def test_thread_safety(self):
        """Test that buffer is thread-safe."""
        import threading

        buffer = CircularBufferHandler(max_lines=1000)

        def writer(thread_id):
            for i in range(100):
                buffer.write(f"thread {thread_id} line {i}\n")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = buffer.get_lines()
        assert len(lines) == 500


class TestCrashResistantLogger:
    """Tests for CrashResistantLogger."""

    def test_initialization(self):
        """Test logger initialization creates files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CrashResistantLogger(
                log_dir=tmpdir,
                prefix="test",
                max_buffer_lines=100,
            )

            # Check that log directory exists
            assert Path(tmpdir).exists()

            # Check that metadata file was created
            assert logger.metadata_path.exists()

            # Clean up
            logger.full_log_file.close()

    def test_context_manager(self):
        """Test context manager usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with CrashResistantLogger(
                log_dir=tmpdir,
                prefix="test",
                max_buffer_lines=100,
            ) as logger:
                # Check that stdout was redirected
                assert sys.stdout is logger

                # Write some output
                print("test output")

            # Check that stdout was restored
            assert sys.stdout is not logger

    def test_write_creates_logs(self):
        """Test that writes create log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with CrashResistantLogger(
                log_dir=tmpdir,
                prefix="test",
                max_buffer_lines=100,
            ) as logger:
                print("test message")

            # Check that log files were created
            assert logger.full_log_path.exists()
            assert logger.circular_log_path.exists()

            # Check that content was written
            with open(logger.full_log_path, 'r') as f:
                content = f.read()
            assert "test message" in content

    def test_timestamps(self):
        """Test that output is timestamped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with CrashResistantLogger(
                log_dir=tmpdir,
                prefix="test",
                max_buffer_lines=100,
            ) as logger:
                print("timestamped message")

            with open(logger.full_log_path, 'r') as f:
                content = f.read()

            # Check for timestamp format [YYYY-MM-DD HH:MM:SS.mmm]
            import re
            assert re.search(r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\]', content)

    def test_exception_logging(self):
        """Test that exceptions are logged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CrashResistantLogger(
                log_dir=tmpdir,
                prefix="test",
                max_buffer_lines=100,
            )

            try:
                with logger:
                    raise ValueError("test error")
            except ValueError:
                pass

            # Check that exception was logged
            with open(logger.full_log_path, 'r') as f:
                content = f.read()

            assert "EXCEPTION CAUGHT" in content
            assert "ValueError" in content

    def test_isatty_returns_false(self):
        """Test that isatty returns False for compatibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CrashResistantLogger(
                log_dir=tmpdir,
                prefix="test",
                max_buffer_lines=100,
            )

            assert logger.isatty() is False
            logger.full_log_file.close()


class TestTeeLogger:
    """Tests for TeeLogger."""

    def test_tee_writes_to_multiple_outputs(self):
        """Test that TeeLogger writes to multiple outputs."""
        from io import StringIO

        output1 = StringIO()
        output2 = StringIO()

        tee = TeeLogger(output1, output2)
        tee.write("test message")
        tee.flush()

        assert output1.getvalue() == "test message"
        assert output2.getvalue() == "test message"


class TestSetupFunction:
    """Tests for setup_crash_resistant_logging function."""

    def test_setup_returns_logger(self):
        """Test that setup function returns a logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_crash_resistant_logging(
                log_dir=tmpdir,
                prefix="test",
                max_buffer_lines=100,
                auto_start=False,
            )

            assert isinstance(logger, CrashResistantLogger)
            assert not logger.active

            # Clean up
            logger.full_log_file.close()

    def test_setup_auto_start(self):
        """Test that setup function can auto-start."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_crash_resistant_logging(
                log_dir=tmpdir,
                prefix="test",
                max_buffer_lines=100,
                auto_start=True,
            )

            assert logger.active

            # Clean up
            logger.stop()


class TestIntegration:
    """Integration tests for crash logger."""

    def test_full_workflow(self):
        """Test full workflow of crash logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up logging
            logger = setup_crash_resistant_logging(
                log_dir=tmpdir,
                prefix="integration_test",
                max_buffer_lines=50,
                auto_start=True,
            )

            # Write some output
            for i in range(100):
                print(f"Line {i}")

            # Stop logging
            logger.stop()

            # Check circular buffer only has last 50 lines
            with open(logger.circular_log_path, 'r') as f:
                circular_content = f.read()

            # Last line should be present
            assert "Line 99" in circular_content
            # First line should not be in circular (dropped)
            assert "Line 0" not in circular_content

            # Full log should have everything
            with open(logger.full_log_path, 'r') as f:
                full_content = f.read()

            assert "Line 0" in full_content
            assert "Line 99" in full_content
