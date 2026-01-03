"""
Test system memory accumulation during HPO trials.

This test specifically targets system memory issues that can persist
even after Python garbage collection.
"""

import gc
import os
import tempfile
import time
from pathlib import Path
from typing import List

import optuna
import psutil
import pytest
import torch
import torch.nn as nn
import numpy as np

# Mark all tests in this module as slow integration tests
pytestmark = pytest.mark.timeout(120)
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset


def get_system_memory_info():
    """Get detailed system memory information."""
    mem = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    process_mem = process.memory_info()
    
    return {
        'system_total_mb': mem.total / 1024 / 1024,
        'system_available_mb': mem.available / 1024 / 1024,
        'system_used_mb': mem.used / 1024 / 1024,
        'system_percent': mem.percent,
        'process_rss_mb': process_mem.rss / 1024 / 1024,
        'process_vms_mb': process_mem.vms / 1024 / 1024,
    }


class MemoryLeakingModel(LightningModule):
    """Model that intentionally creates memory leaks similar to real scenarios."""
    
    # Class-level cache that could accumulate
    _global_cache = []
    
    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Create a reasonably complex model
        self.encoder = nn.Sequential(
            nn.Linear(100, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )
        
        # Simulate caching behavior that could leak
        self.cache = {}
        self.training_history = []
        
    def forward(self, x):
        encoded = self.encoder(x)
        # Store intermediate results (potential leak)
        self.cache[f'encoded_{len(self.cache)}'] = encoded.detach().cpu()
        
        # Add to global cache (definite leak)
        if len(self._global_cache) < 100:
            self._global_cache.append(encoded.detach().cpu().numpy())
        
        return self.decoder(encoded)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        
        # Store training metrics (potential accumulation)
        self.training_history.append({
            'batch_idx': batch_idx,
            'loss': loss.item(),
            'timestamp': time.time()
        })
        
        # Log the loss for the scheduler
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Create optimizer with potential memory retention
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    @classmethod
    def reset_global_cache(cls):
        """Reset the global cache."""
        cls._global_cache.clear()


def create_large_dataset(size: int = 10000):
    """Create a dataset that uses significant memory."""
    X = torch.randn(size, 100)
    y = torch.randn(size, 10)
    return TensorDataset(X, y)


class TestSystemMemoryAccumulation:
    """Test suite for system memory accumulation during HPO."""
    
    def test_system_memory_leak_detection(self, tmp_path):
        """Test detection of system memory leaks during HPO."""
        
        # Get initial system memory state
        initial_mem = get_system_memory_info()
        print("\nInitial System Memory State:")
        print(f"  System Used: {initial_mem['system_used_mb']:.1f} MB ({initial_mem['system_percent']:.1f}%)")
        print(f"  System Available: {initial_mem['system_available_mb']:.1f} MB")
        print(f"  Process RSS: {initial_mem['process_rss_mb']:.1f} MB")
        
        memory_readings = []
        trial_count = 0
        
        def objective_with_leak(trial: optuna.Trial) -> float:
            """Objective function that simulates realistic memory leaks."""
            nonlocal trial_count, memory_readings
            
            # Record memory before trial
            mem_before = get_system_memory_info()
            
            # Hyperparameters
            hidden_size = trial.suggest_int("hidden_size", 256, 1024)
            batch_size = trial.suggest_int("batch_size", 32, 128)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            
            # Create model and dataset
            model = MemoryLeakingModel(hidden_size=hidden_size)
            dataset = create_large_dataset(size=5000)
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
            
            # Create trainer with potential memory retention
            trainer = Trainer(
                max_epochs=2,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
            )
            
            # Train the model
            trainer.fit(model, dataloader)
            
            # Get some metrics
            val_loss = np.random.random()  # Simulate validation loss
            
            # Record memory after trial
            mem_after = get_system_memory_info()
            
            memory_readings.append({
                'trial': trial_count,
                'system_used_before': mem_before['system_used_mb'],
                'system_used_after': mem_after['system_used_mb'],
                'system_available_before': mem_before['system_available_mb'],
                'system_available_after': mem_after['system_available_mb'],
                'process_rss_before': mem_before['process_rss_mb'],
                'process_rss_after': mem_after['process_rss_mb'],
                'system_delta': mem_after['system_used_mb'] - mem_before['system_used_mb'],
                'process_delta': mem_after['process_rss_mb'] - mem_before['process_rss_mb'],
            })
            trial_count += 1
            
            # Explicitly delete large objects
            del model
            del dataset
            del dataloader
            del trainer
            
            return val_loss
        
        # First run WITHOUT cleanup
        print("\n" + "="*60)
        print("Running trials WITHOUT memory cleanup...")
        
        study_no_cleanup = optuna.create_study(direction="minimize")
        
        # Reset global cache before starting
        MemoryLeakingModel.reset_global_cache()
        memory_readings = []
        trial_count = 0
        
        # Run trials without cleanup
        for _ in range(10):
            study_no_cleanup.optimize(
                objective_with_leak, 
                n_trials=1, 
                gc_after_trial=False
            )
        
        no_cleanup_readings = memory_readings.copy()
        
        # Force cleanup between test runs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1)  # Give OS time to reclaim memory
        
        # Now run WITH cleanup
        print("\n" + "="*60)
        print("Running trials WITH memory cleanup...")
        
        from LightningTune.optuna.memory_cleanup import cleanup_trial_resources
        
        def objective_with_cleanup(trial: optuna.Trial) -> float:
            """Objective with explicit cleanup."""
            result = objective_with_leak(trial)
            cleanup_trial_resources()
            MemoryLeakingModel.reset_global_cache()  # Also reset global cache
            return result
        
        study_with_cleanup = optuna.create_study(direction="minimize")
        
        # Reset for second run
        MemoryLeakingModel.reset_global_cache()
        memory_readings = []
        trial_count = 0
        
        # Run trials with cleanup
        for _ in range(10):
            study_with_cleanup.optimize(
                objective_with_cleanup,
                n_trials=1,
                gc_after_trial=True
            )
        
        with_cleanup_readings = memory_readings.copy()
        
        # Analyze results
        print("\n" + "="*60)
        print("System Memory Analysis:")
        print("\nWITHOUT cleanup:")
        print(f"{'Trial':<7} {'Sys Before':<12} {'Sys After':<12} {'Sys Delta':<12} {'Proc RSS Delta':<15}")
        print("-" * 70)
        
        for r in no_cleanup_readings[:3]:
            print(f"{r['trial']:<7} {r['system_used_before']:<12.1f} {r['system_used_after']:<12.1f} "
                  f"{r['system_delta']:<12.1f} {r['process_delta']:<15.1f}")
        print("...")
        for r in no_cleanup_readings[-2:]:
            print(f"{r['trial']:<7} {r['system_used_before']:<12.1f} {r['system_used_after']:<12.1f} "
                  f"{r['system_delta']:<12.1f} {r['process_delta']:<15.1f}")
        
        print("\nWITH cleanup:")
        print(f"{'Trial':<7} {'Sys Before':<12} {'Sys After':<12} {'Sys Delta':<12} {'Proc RSS Delta':<15}")
        print("-" * 70)
        
        for r in with_cleanup_readings[:3]:
            print(f"{r['trial']:<7} {r['system_used_before']:<12.1f} {r['system_used_after']:<12.1f} "
                  f"{r['system_delta']:<12.1f} {r['process_delta']:<15.1f}")
        print("...")
        for r in with_cleanup_readings[-2:]:
            print(f"{r['trial']:<7} {r['system_used_before']:<12.1f} {r['system_used_after']:<12.1f} "
                  f"{r['system_delta']:<12.1f} {r['process_delta']:<15.1f}")
        
        # Calculate cumulative system memory increase
        def calc_cumulative_system_increase(readings):
            cumulative = 0
            for r in readings:
                cumulative += r['system_delta']
            return cumulative
        
        no_cleanup_cumulative = calc_cumulative_system_increase(no_cleanup_readings)
        with_cleanup_cumulative = calc_cumulative_system_increase(with_cleanup_readings)
        
        print("\n" + "="*60)
        print(f"Cumulative system memory increase WITHOUT cleanup: {no_cleanup_cumulative:.1f} MB")
        print(f"Cumulative system memory increase WITH cleanup: {with_cleanup_cumulative:.1f} MB")
        
        # Check that cleanup reduces system memory accumulation
        improvement = (no_cleanup_cumulative - with_cleanup_cumulative) / max(abs(no_cleanup_cumulative), 1)
        print(f"Improvement with cleanup: {improvement*100:.1f}%")
        
        # The cleanup version should accumulate less system memory
        # We allow for some variance but expect improvement
        if no_cleanup_cumulative > 100:  # Only test if there was significant accumulation
            assert with_cleanup_cumulative < no_cleanup_cumulative * 0.8, (
                f"Cleanup didn't significantly reduce system memory accumulation: "
                f"with_cleanup ({with_cleanup_cumulative:.1f} MB) vs "
                f"no_cleanup ({no_cleanup_cumulative:.1f} MB)"
            )
    
    def test_pytorch_memory_pools(self):
        """Test that PyTorch memory pools are properly managed."""
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        from LightningTune.optuna.memory_cleanup import cleanup_trial_resources
        
        print("\n" + "="*60)
        print("Testing PyTorch Memory Pool Management:")
        
        # Get initial state
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        initial_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        initial_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        
        print(f"\nInitial GPU state:")
        print(f"  Reserved: {initial_reserved:.1f} MB")
        print(f"  Allocated: {initial_allocated:.1f} MB")
        
        # Simulate multiple trials with GPU memory usage
        for trial_num in range(5):
            print(f"\nTrial {trial_num}:")
            
            # Allocate GPU memory
            tensors = []
            for _ in range(10):
                t = torch.randn(1000, 1000).cuda()
                tensors.append(t)
                # Do some operations that might fragment memory
                _ = torch.matmul(t, t.T)
            
            allocated_before_cleanup = torch.cuda.memory_allocated() / 1024 / 1024
            reserved_before_cleanup = torch.cuda.memory_reserved() / 1024 / 1024
            
            print(f"  Before cleanup - Allocated: {allocated_before_cleanup:.1f} MB, "
                  f"Reserved: {reserved_before_cleanup:.1f} MB")
            
            # Delete tensors and run cleanup
            del tensors
            cleanup_trial_resources()
            
            allocated_after_cleanup = torch.cuda.memory_allocated() / 1024 / 1024
            reserved_after_cleanup = torch.cuda.memory_reserved() / 1024 / 1024
            
            print(f"  After cleanup - Allocated: {allocated_after_cleanup:.1f} MB, "
                  f"Reserved: {reserved_after_cleanup:.1f} MB")
        
        # Final state should be close to initial
        final_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        final_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        
        print(f"\nFinal GPU state:")
        print(f"  Reserved: {final_reserved:.1f} MB")
        print(f"  Allocated: {final_allocated:.1f} MB")
        
        # Allocated memory should return to near zero
        # Note: PyTorch may keep some memory allocated for efficiency
        assert final_allocated < 20, (
            f"GPU allocated memory not properly freed: {final_allocated:.1f} MB"
        )
        
        # Reserved memory might stay higher but shouldn't grow unbounded
        assert final_reserved < initial_reserved + 100, (
            f"GPU reserved memory grew too much: "
            f"from {initial_reserved:.1f} MB to {final_reserved:.1f} MB"
        )
    
    def test_multiprocessing_shared_memory_cleanup(self):
        """Test cleanup of shared memory from multiprocessing."""
        
        import multiprocessing as mp
        from multiprocessing import shared_memory
        
        print("\n" + "="*60)
        print("Testing Shared Memory Cleanup:")
        
        initial_mem = get_system_memory_info()
        print(f"\nInitial system memory: {initial_mem['system_used_mb']:.1f} MB")
        
        # Track shared memory blocks
        shm_blocks = []
        
        def create_shared_memory_leak():
            """Simulate shared memory that might not get cleaned up."""
            # Create shared memory blocks
            for i in range(5):
                size = 10 * 1024 * 1024  # 10 MB each
                shm = shared_memory.SharedMemory(create=True, size=size)
                # Write some data
                shm.buf[:100] = b'x' * 100
                shm_blocks.append(shm)
                print(f"  Created shared memory block {i}: {shm.name}")
        
        # Create shared memory blocks
        create_shared_memory_leak()
        
        after_creation = get_system_memory_info()
        print(f"\nAfter creating shared memory: {after_creation['system_used_mb']:.1f} MB")
        print(f"  Delta: {after_creation['system_used_mb'] - initial_mem['system_used_mb']:.1f} MB")
        
        # Now clean up
        for shm in shm_blocks:
            shm.close()
            shm.unlink()  # This is critical for system memory cleanup
        shm_blocks.clear()
        
        # Force garbage collection
        gc.collect()
        time.sleep(0.5)  # Give OS time to reclaim
        
        after_cleanup = get_system_memory_info()
        print(f"\nAfter cleanup: {after_cleanup['system_used_mb']:.1f} MB")
        print(f"  Delta from initial: {after_cleanup['system_used_mb'] - initial_mem['system_used_mb']:.1f} MB")
        
        # Memory should return close to initial levels
        # Use 100 MB threshold to account for system memory variations in CI environments
        memory_increase = after_cleanup['system_used_mb'] - initial_mem['system_used_mb']
        assert memory_increase < 100, (
            f"Shared memory not properly cleaned: {memory_increase:.1f} MB increase"
        )


if __name__ == "__main__":
    # Run tests directly
    import sys
    test = TestSystemMemoryAccumulation()
    tmp_dir = Path(tempfile.mkdtemp())
    
    try:
        print("Running system memory accumulation tests...")
        
        test.test_system_memory_leak_detection(tmp_dir)
        print("\n✓ System memory leak detection test completed!")
        
        if torch.cuda.is_available():
            test.test_pytorch_memory_pools()
            print("\n✓ PyTorch memory pool test passed!")
        
        test.test_multiprocessing_shared_memory_cleanup()
        print("\n✓ Shared memory cleanup test passed!")
        
        print("\n" + "="*60)
        print("All system memory tests completed successfully!")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)