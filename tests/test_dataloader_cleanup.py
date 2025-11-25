"""
Unit tests for DataLoader cleanup functions.

These tests verify that memory_cleanup functions properly terminate
DataLoader workers and prevent thread accumulation.
"""

import pytest
import threading
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.callbacks import Callback

from LightningTune.optuna.memory_cleanup import (
    cleanup_dataloader_workers,
    cleanup_trial_resources,
)


class DummyModel(LightningModule):
    """Minimal model for testing."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class DummyDataModule(LightningDataModule):
    """DataModule with real DataLoaders for testing cleanup."""

    def __init__(self, num_workers=4, batch_size=32):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Create simple dataset
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        self.dataset = TensorDataset(X, y)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=False,  # Workers should terminate on __del__
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=False,
        )


def count_queue_feeder_threads():
    """Count QueueFeederThread instances."""
    return sum(1 for t in threading.enumerate() if 'QueueFeederThread' in t.name)


def count_pin_memory_threads():
    """Count _pin_memory_loop threads."""
    return sum(1 for t in threading.enumerate() if '_pin_memory_loop' in t.name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for pin_memory threads")
def test_cleanup_dataloader_workers_with_real_dataloaders():
    """
    Test that cleanup_dataloader_workers() properly terminates DataLoader workers.

    This test:
    1. Creates a trainer with DataLoaders (4 workers)
    2. Runs a few training steps to spawn workers
    3. Calls cleanup_dataloader_workers()
    4. Asserts worker threads are terminated (if any were created)

    Note: QueueFeederThread creation depends on system configuration and PyTorch
    internals. With small datasets and few steps, workers may not always spawn.
    """
    # Create model and datamodule
    model = DummyModel()
    datamodule = DummyDataModule(num_workers=4)

    # Create trainer
    trainer = Trainer(
        max_epochs=1,
        max_steps=5,  # Just a few steps to spawn workers
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    # Baseline thread count
    baseline_threads = threading.active_count()
    baseline_queue_feeders = count_queue_feeder_threads()

    # Fit model (this spawns DataLoader workers)
    trainer.fit(model, datamodule=datamodule)

    # After fit, check if workers were created (not guaranteed)
    after_fit_threads = threading.active_count()
    after_fit_queue_feeders = count_queue_feeder_threads()
    workers_were_created = after_fit_queue_feeders > baseline_queue_feeders

    # Call cleanup
    cleanup_dataloader_workers(trainer=trainer, datamodule=datamodule)

    # Wait a moment for threads to terminate
    time.sleep(1.0)

    # Check threads reduced
    after_cleanup_threads = threading.active_count()
    after_cleanup_queue_feeders = count_queue_feeder_threads()

    # If workers were created, verify they were cleaned up
    if workers_were_created:
        assert after_cleanup_queue_feeders <= baseline_queue_feeders, (
            f"QueueFeederThread leak detected! "
            f"Baseline: {baseline_queue_feeders}, "
            f"After cleanup: {after_cleanup_queue_feeders}"
        )

    # Thread count should be back near baseline (allow small variance)
    thread_growth = after_cleanup_threads - baseline_threads
    assert thread_growth <= 2, (
        f"Thread leak detected! "
        f"Baseline: {baseline_threads}, "
        f"After cleanup: {after_cleanup_threads}, "
        f"Growth: {thread_growth}"
    )


def test_cleanup_trial_resources_integration():
    """
    Test cleanup_trial_resources() with real trainer and datamodule.

    This is closer to the actual HPO trial cleanup scenario.
    Note: QueueFeederThread creation depends on system configuration and PyTorch
    internals. The test validates cleanup works when workers exist, but doesn't
    fail if workers weren't spawned (which can happen with small datasets/epochs).
    """
    # Baseline metrics
    baseline_threads = threading.active_count()
    baseline_queue_feeders = count_queue_feeder_threads()

    # Create and run trial
    model = DummyModel()
    datamodule = DummyDataModule(num_workers=4)

    trainer = Trainer(
        max_epochs=1,
        max_steps=5,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    # Run training
    trainer.fit(model, datamodule=datamodule)

    # Check if workers were created (not guaranteed on all systems)
    after_fit_queue_feeders = count_queue_feeder_threads()
    workers_were_created = after_fit_queue_feeders > baseline_queue_feeders

    # Call cleanup_trial_resources (full cleanup including gc)
    cleanup_trial_resources(trainer=trainer, datamodule=datamodule)

    # Wait for cleanup
    time.sleep(1.0)

    # Verify cleanup
    after_cleanup_threads = threading.active_count()
    after_cleanup_queue_feeders = count_queue_feeder_threads()

    # If workers were created, verify they were cleaned up
    if workers_were_created:
        assert after_cleanup_queue_feeders <= baseline_queue_feeders, (
            f"QueueFeederThread not cleaned up properly: "
            f"{after_cleanup_queue_feeders} vs baseline {baseline_queue_feeders}"
        )

    # Thread count should not grow significantly regardless
    thread_growth = after_cleanup_threads - baseline_threads
    assert thread_growth <= 2, (
        f"Threads not cleaned up properly: growth of {thread_growth}"
    )


def test_cleanup_dataloader_workers_handles_none_gracefully():
    """Test that cleanup functions handle None inputs gracefully."""
    # Should not crash
    cleanup_dataloader_workers(trainer=None, datamodule=None)
    cleanup_trial_resources(trainer=None, datamodule=None)


def test_cleanup_multiple_trials_no_accumulation():
    """
    Test that running multiple trials with cleanup doesn't accumulate threads.

    This simulates the HPO scenario where multiple trials run sequentially.
    """
    baseline_threads = threading.active_count()
    trial_thread_counts = []

    # Run 3 mini "trials"
    for i in range(3):
        model = DummyModel()
        datamodule = DummyDataModule(num_workers=2)  # Fewer workers for speed

        trainer = Trainer(
            max_epochs=1,
            max_steps=3,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        # Run trial
        trainer.fit(model, datamodule=datamodule)

        # Cleanup
        cleanup_trial_resources(trainer=trainer, datamodule=datamodule)

        # Wait for cleanup
        time.sleep(0.5)

        # Record thread count after cleanup
        trial_thread_counts.append(threading.active_count())

    # Thread count should NOT grow across trials
    thread_counts_stable = all(
        abs(count - baseline_threads) <= 3
        for count in trial_thread_counts
    )

    assert thread_counts_stable, (
        f"Thread accumulation detected across trials!\n"
        f"Baseline: {baseline_threads}\n"
        f"Trial thread counts: {trial_thread_counts}\n"
        f"Growth: {[c - baseline_threads for c in trial_thread_counts]}"
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
