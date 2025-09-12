"""
Test memory accumulation during HPO trials.

This test runs multiple trials with a dummy model and monitors memory usage
to ensure cleanup is working properly.
"""

import gc
import os
import tempfile
from pathlib import Path

import optuna
import psutil
import pytest
import torch
import numpy as np


def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class TestMemoryAccumulation:
    """Test suite for memory accumulation during HPO."""
    
    def test_memory_cleanup_with_optuna_gc(self, tmp_path):
        """Test that Optuna's gc_after_trial prevents memory accumulation."""
        
        # Track memory usage over trials
        memory_readings = []
        trial_count = 0
        
        def objective(trial: optuna.Trial) -> float:
            """Objective function that creates objects that could accumulate."""
            nonlocal trial_count, memory_readings
            
            # Record memory before creating objects
            mem_before = get_memory_usage()
            
            # Hyperparameters
            size = trial.suggest_int("size", 100, 1000)
            n_tensors = trial.suggest_int("n_tensors", 5, 20)
            
            # Create some large objects that could accumulate
            tensors = []
            for i in range(n_tensors):
                # Create tensors of varying sizes
                tensor_size = size * (i + 1)
                tensors.append(torch.randn(tensor_size, tensor_size // 10))
            
            # Do some computation
            result = sum(t.mean().item() for t in tensors)
            
            # Create some regular Python objects too
            large_list = [list(range(size)) for _ in range(100)]
            large_dict = {str(i): list(range(size)) for i in range(50)}
            
            # Record memory after creating objects
            mem_after = get_memory_usage()
            memory_readings.append({
                'trial': trial_count,
                'before': mem_before,
                'after': mem_after,
                'delta': mem_after - mem_before
            })
            trial_count += 1
            
            return result
        
        # Test WITHOUT gc_after_trial first
        study_path = tmp_path / "test_study_no_gc.db"
        study_no_gc = optuna.create_study(
            storage=f"sqlite:///{study_path}",
            study_name="memory_test_no_gc",
            direction="minimize"
        )
        
        memory_readings = []
        trial_count = 0
        
        # Run trials without GC
        study_no_gc.optimize(objective, n_trials=15, gc_after_trial=False)
        
        no_gc_readings = memory_readings.copy()
        
        # Test WITH gc_after_trial
        study_path = tmp_path / "test_study_with_gc.db"
        study_with_gc = optuna.create_study(
            storage=f"sqlite:///{study_path}",
            study_name="memory_test_with_gc",
            direction="minimize"
        )
        
        memory_readings = []
        trial_count = 0
        
        # Run trials with GC
        study_with_gc.optimize(objective, n_trials=15, gc_after_trial=True)
        
        with_gc_readings = memory_readings.copy()
        
        # Analyze results
        print("\n" + "="*60)
        print("Memory usage WITHOUT gc_after_trial:")
        print(f"{'Trial':<8} {'Before (MB)':<12} {'After (MB)':<12} {'Delta (MB)':<12}")
        print("-" * 44)
        
        for reading in no_gc_readings[:5]:  # Show first 5
            print(f"{reading['trial']:<8} {reading['before']:<12.1f} "
                  f"{reading['after']:<12.1f} {reading['delta']:<12.1f}")
        print("...")
        for reading in no_gc_readings[-3:]:  # Show last 3
            print(f"{reading['trial']:<8} {reading['before']:<12.1f} "
                  f"{reading['after']:<12.1f} {reading['delta']:<12.1f}")
        
        print("\n" + "="*60)
        print("Memory usage WITH gc_after_trial:")
        print(f"{'Trial':<8} {'Before (MB)':<12} {'After (MB)':<12} {'Delta (MB)':<12}")
        print("-" * 44)
        
        for reading in with_gc_readings[:5]:  # Show first 5
            print(f"{reading['trial']:<8} {reading['before']:<12.1f} "
                  f"{reading['after']:<12.1f} {reading['delta']:<12.1f}")
        print("...")
        for reading in with_gc_readings[-3:]:  # Show last 3
            print(f"{reading['trial']:<8} {reading['before']:<12.1f} "
                  f"{reading['after']:<12.1f} {reading['delta']:<12.1f}")
        
        # Calculate memory growth
        def calculate_memory_growth(readings):
            after_memories = [r['after'] for r in readings]
            initial = after_memories[0]
            final = after_memories[-1]
            return final / initial
        
        no_gc_growth = calculate_memory_growth(no_gc_readings)
        with_gc_growth = calculate_memory_growth(with_gc_readings)
        
        print("\n" + "="*60)
        print(f"Memory growth WITHOUT gc: {no_gc_growth:.2f}x")
        print(f"Memory growth WITH gc: {with_gc_growth:.2f}x")
        
        # The version with GC should have less memory growth
        assert with_gc_growth <= no_gc_growth + 0.1, (
            f"gc_after_trial didn't help: "
            f"with_gc growth ({with_gc_growth:.2f}x) >= "
            f"no_gc growth ({no_gc_growth:.2f}x)"
        )
    
    def test_memory_cleanup_with_custom_cleanup(self, tmp_path):
        """Test our custom memory cleanup function."""
        
        from LightningTune.optuna.memory_cleanup import cleanup_trial_resources
        
        memory_readings = []
        trial_count = 0
        
        def objective_with_cleanup(trial: optuna.Trial) -> float:
            """Objective that calls our cleanup function."""
            nonlocal trial_count, memory_readings
            
            mem_before = get_memory_usage()
            
            # Create GPU tensors if available
            if torch.cuda.is_available():
                gpu_tensors = [torch.randn(1000, 1000).cuda() for _ in range(5)]
                result = sum(t.mean().item() for t in gpu_tensors)
            else:
                cpu_tensors = [torch.randn(1000, 1000) for _ in range(10)]
                result = sum(t.mean().item() for t in cpu_tensors)
            
            # Large Python objects
            large_data = {
                'arrays': [np.random.rand(1000, 1000) for _ in range(3)],
                'lists': [list(range(10000)) for _ in range(100)],
                'dicts': {str(i): list(range(1000)) for i in range(100)}
            }
            
            # Call our cleanup function
            cleanup_trial_resources()
            
            mem_after = get_memory_usage()
            memory_readings.append({
                'trial': trial_count,
                'before': mem_before,
                'after': mem_after,
                'delta': mem_after - mem_before
            })
            trial_count += 1
            
            return result
        
        study = optuna.create_study(direction="minimize")
        
        # Run trials with our custom cleanup
        study.optimize(objective_with_cleanup, n_trials=20)
        
        # Check memory growth
        after_memories = [r['after'] for r in memory_readings]
        
        # Check that memory is relatively stable
        # Calculate standard deviation of memory usage
        mean_memory = np.mean(after_memories)
        std_memory = np.std(after_memories)
        cv = std_memory / mean_memory  # Coefficient of variation
        
        print("\n" + "="*60)
        print("Memory statistics with custom cleanup:")
        print(f"Mean memory: {mean_memory:.1f} MB")
        print(f"Std deviation: {std_memory:.1f} MB")
        print(f"Coefficient of variation: {cv:.3f}")
        
        # Memory should be relatively stable (low CV)
        assert cv < 0.3, (
            f"Memory usage too variable with custom cleanup: CV={cv:.3f}"
        )
        
        # Check that we don't have runaway growth
        growth_ratio = after_memories[-1] / after_memories[0]
        print(f"Growth ratio (final/initial): {growth_ratio:.2f}x")
        
        assert growth_ratio < 1.5, (
            f"Too much memory growth with custom cleanup: {growth_ratio:.2f}x"
        )
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_cleanup(self):
        """Test that GPU memory is properly cleaned after cleanup."""
        
        from LightningTune.optuna.memory_cleanup import cleanup_trial_resources
        
        # Clear any existing GPU memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        initial_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
        
        print(f"\nInitial GPU memory: {initial_gpu_mem:.1f} MB")
        
        # Allocate some GPU memory
        tensors = []
        for i in range(10):
            tensors.append(torch.randn(1000, 1000).cuda())
        
        allocated_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"GPU memory after allocation: {allocated_gpu_mem:.1f} MB")
        
        # Delete references and call cleanup
        del tensors
        cleanup_trial_resources()
        
        final_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"GPU memory after cleanup: {final_gpu_mem:.1f} MB")
        
        # Memory should be back to near initial levels
        assert final_gpu_mem <= initial_gpu_mem + 10, (
            f"GPU memory not properly cleaned: "
            f"{final_gpu_mem:.1f} MB > {initial_gpu_mem + 10:.1f} MB"
        )


if __name__ == "__main__":
    # Run the test directly for debugging
    import sys
    test = TestMemoryAccumulation()
    tmp_dir = Path(tempfile.mkdtemp())
    
    try:
        print("Running memory cleanup tests...")
        test.test_memory_cleanup_with_optuna_gc(tmp_dir)
        print("\n✓ Optuna GC test passed!")
        
        test.test_memory_cleanup_with_custom_cleanup(tmp_dir)
        print("✓ Custom cleanup test passed!")
        
        if torch.cuda.is_available():
            test.test_gpu_memory_cleanup()
            print("✓ GPU memory cleanup test passed!")
        
        print("\nAll tests passed!")
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)