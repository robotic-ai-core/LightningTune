"""
Tests for the checkpoint_top_k feature that manages trial checkpoint retention.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

from LightningTune.optuna.optimizer import TrialCheckpointManager


class TestTrialCheckpointManager:
    """Tests for TrialCheckpointManager class."""

    def test_checkpoints_disabled_by_default(self):
        """Test that checkpointing is disabled when top_k=0."""
        manager = TrialCheckpointManager(top_k=0)
        assert not manager.checkpoints_enabled

    def test_checkpoints_enabled_when_top_k_positive(self):
        """Test that checkpointing is enabled when top_k > 0."""
        manager = TrialCheckpointManager(top_k=3)
        assert manager.checkpoints_enabled

    def test_register_does_nothing_when_disabled(self):
        """Test that register_trial does nothing when checkpoints are disabled."""
        manager = TrialCheckpointManager(top_k=0)
        manager.register_trial(0, 0.5, Path("/fake/path"))
        assert len(manager.trial_checkpoints) == 0

    def test_register_trial_stores_checkpoint(self, tmp_path):
        """Test that register_trial correctly stores checkpoint info."""
        manager = TrialCheckpointManager(top_k=3, direction="minimize")

        # Create a checkpoint directory
        checkpoint_path = tmp_path / "trial_0"
        checkpoint_path.mkdir()

        manager.register_trial(0, 0.5, checkpoint_path)

        assert 0 in manager.trial_checkpoints
        assert manager.trial_checkpoints[0] == (0.5, checkpoint_path)

    def test_cleanup_keeps_top_k_minimize(self, tmp_path):
        """Test that cleanup keeps only top-k trials when minimizing."""
        manager = TrialCheckpointManager(top_k=2, direction="minimize")

        # Create 4 trial checkpoints with values (lower is better for minimize)
        for i, value in enumerate([0.5, 0.3, 0.8, 0.1]):
            checkpoint_path = tmp_path / f"trial_{i}"
            checkpoint_path.mkdir()
            (checkpoint_path / "checkpoint.ckpt").touch()
            manager.register_trial(i, value, checkpoint_path)

        # Should keep trials 3 (0.1) and 1 (0.3) - the two best (lowest)
        assert len(manager.trial_checkpoints) == 2
        assert 3 in manager.trial_checkpoints  # value 0.1
        assert 1 in manager.trial_checkpoints  # value 0.3

        # Trials 0 and 2 should have been cleaned up
        assert not (tmp_path / "trial_0").exists()
        assert not (tmp_path / "trial_2").exists()

        # Best trials should still exist
        assert (tmp_path / "trial_1").exists()
        assert (tmp_path / "trial_3").exists()

    def test_cleanup_keeps_top_k_maximize(self, tmp_path):
        """Test that cleanup keeps only top-k trials when maximizing."""
        manager = TrialCheckpointManager(top_k=2, direction="maximize")

        # Create 4 trial checkpoints with values (higher is better for maximize)
        for i, value in enumerate([0.5, 0.3, 0.8, 0.1]):
            checkpoint_path = tmp_path / f"trial_{i}"
            checkpoint_path.mkdir()
            (checkpoint_path / "checkpoint.ckpt").touch()
            manager.register_trial(i, value, checkpoint_path)

        # Should keep trials 2 (0.8) and 0 (0.5) - the two best (highest)
        assert len(manager.trial_checkpoints) == 2
        assert 2 in manager.trial_checkpoints  # value 0.8
        assert 0 in manager.trial_checkpoints  # value 0.5

        # Trials 1 and 3 should have been cleaned up
        assert not (tmp_path / "trial_1").exists()
        assert not (tmp_path / "trial_3").exists()

        # Best trials should still exist
        assert (tmp_path / "trial_0").exists()
        assert (tmp_path / "trial_2").exists()

    def test_get_best_checkpoint_path_minimize(self, tmp_path):
        """Test getting the best checkpoint path when minimizing."""
        manager = TrialCheckpointManager(top_k=5, direction="minimize")

        for i, value in enumerate([0.5, 0.3, 0.8]):
            checkpoint_path = tmp_path / f"trial_{i}"
            checkpoint_path.mkdir()
            manager.register_trial(i, value, checkpoint_path)

        best_path = manager.get_best_checkpoint_path()
        assert best_path == tmp_path / "trial_1"  # value 0.3 is best (lowest)

    def test_get_best_checkpoint_path_maximize(self, tmp_path):
        """Test getting the best checkpoint path when maximizing."""
        manager = TrialCheckpointManager(top_k=5, direction="maximize")

        for i, value in enumerate([0.5, 0.3, 0.8]):
            checkpoint_path = tmp_path / f"trial_{i}"
            checkpoint_path.mkdir()
            manager.register_trial(i, value, checkpoint_path)

        best_path = manager.get_best_checkpoint_path()
        assert best_path == tmp_path / "trial_2"  # value 0.8 is best (highest)

    def test_get_best_checkpoint_path_empty(self):
        """Test getting best checkpoint path when no checkpoints exist."""
        manager = TrialCheckpointManager(top_k=3)
        assert manager.get_best_checkpoint_path() is None

    def test_cleanup_handles_missing_directories(self, tmp_path):
        """Test that cleanup handles already-deleted directories gracefully."""
        manager = TrialCheckpointManager(top_k=1, direction="minimize")

        # Register trials but don't create directories
        manager.trial_checkpoints[0] = (0.5, tmp_path / "nonexistent_trial_0")
        manager.trial_checkpoints[1] = (0.3, tmp_path / "nonexistent_trial_1")

        # This should not raise an exception
        manager._cleanup_if_needed()

        # Should still keep only top-1
        assert len(manager.trial_checkpoints) == 1
        assert 1 in manager.trial_checkpoints

    def test_top_k_one_keeps_only_best(self, tmp_path):
        """Test that top_k=1 keeps only the single best trial."""
        manager = TrialCheckpointManager(top_k=1, direction="minimize")

        # Add 5 trials
        for i in range(5):
            checkpoint_path = tmp_path / f"trial_{i}"
            checkpoint_path.mkdir()
            (checkpoint_path / "model.ckpt").touch()
            manager.register_trial(i, i * 0.1, checkpoint_path)  # values: 0.0, 0.1, 0.2, 0.3, 0.4

        # Should only have trial 0 (value 0.0)
        assert len(manager.trial_checkpoints) == 1
        assert 0 in manager.trial_checkpoints

        # Only trial_0 should exist
        assert (tmp_path / "trial_0").exists()
        for i in range(1, 5):
            assert not (tmp_path / f"trial_{i}").exists()


class TestOptunaDrivenOptimizerCheckpointIntegration:
    """Integration tests for checkpoint_top_k in OptunaDrivenOptimizer."""

    def test_checkpoint_top_k_zero_disables_checkpoints(self):
        """Test that checkpoint_top_k=0 disables checkpoint saving."""
        from LightningTune.optuna.optimizer import OptunaDrivenOptimizer

        optimizer = OptunaDrivenOptimizer(
            base_config={"test": "config"},
            search_space=lambda trial: {"x": trial.suggest_float("x", 0, 1)},
            model_class=MagicMock,
            checkpoint_top_k=0,
        )

        assert not optimizer.checkpoint_manager.checkpoints_enabled

    def test_checkpoint_top_k_positive_enables_checkpoints(self):
        """Test that checkpoint_top_k > 0 enables checkpoint saving."""
        from LightningTune.optuna.optimizer import OptunaDrivenOptimizer

        optimizer = OptunaDrivenOptimizer(
            base_config={"test": "config"},
            search_space=lambda trial: {"x": trial.suggest_float("x", 0, 1)},
            model_class=MagicMock,
            checkpoint_top_k=3,
        )

        assert optimizer.checkpoint_manager.checkpoints_enabled
        assert optimizer.checkpoint_manager.top_k == 3


class TestPausibleOptimizerCheckpointIntegration:
    """Integration tests for checkpoint_top_k in PausibleOptunaOptimizer."""

    def test_checkpoint_top_k_passed_through(self):
        """Test that checkpoint_top_k is stored in PausibleOptunaOptimizer."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        optimizer = PausibleOptunaOptimizer(
            base_config={"test": "config"},
            search_space=lambda trial: {"x": trial.suggest_float("x", 0, 1)},
            model_class=MagicMock,
            checkpoint_top_k=5,
        )

        assert optimizer.checkpoint_top_k == 5

    def test_checkpoint_top_k_defaults_to_zero(self):
        """Test that checkpoint_top_k defaults to 0 (disabled)."""
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        optimizer = PausibleOptunaOptimizer(
            base_config={"test": "config"},
            search_space=lambda trial: {"x": trial.suggest_float("x", 0, 1)},
            model_class=MagicMock,
        )

        assert optimizer.checkpoint_top_k == 0
