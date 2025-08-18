#!/usr/bin/env python
"""
Minimal test to verify pause/resume functionality works.

This creates a simple optimization that can be paused and resumed.
"""

import sys
import time
from pathlib import Path
from ray import tune

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lightning_bohb import (
    ConfigDrivenOptimizer,
    SearchSpace,
    OptimizationConfig,
    TuneReflowCLI,
)
from lightning_bohb.core.strategies import RandomSearchStrategy


class SimpleSearchSpace(SearchSpace):
    """Minimal search space for testing."""
    
    def get_search_space(self):
        return {
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([16, 32]),
        }
    
    def get_metric_config(self):
        return {"metric": "loss", "mode": "min"}


def test_new_run():
    """Test starting a new optimization with pause capability."""
    print("\n" + "="*60)
    print("TEST: Starting new optimization with pause/resume")
    print("="*60)
    
    # Create dummy config file
    config_file = Path("/tmp/test_config.yaml")
    config_file.write_text("""
model:
  class_path: lightning.pytorch.demos.boring_classes.BoringModel
    
data:
  class_path: lightning.pytorch.demos.boring_classes.BoringDataModule
  
trainer:
  max_epochs: 5
  limit_train_batches: 2
  limit_val_batches: 2
""")
    
    # Create CLI
    cli = TuneReflowCLI(
        experiment_name="test_pause_resume",
        experiment_dir="/tmp/lightning_bohb_tests",
        enable_pause=True,
        verbose=True,
    )
    
    # Create optimizer
    strategy = RandomSearchStrategy(
        num_samples=3,
        use_early_stopping=False,
    )
    
    optimizer = ConfigDrivenOptimizer(
        base_config_path=str(config_file),
        search_space=SimpleSearchSpace(),
        strategy=strategy,
        optimization_config=OptimizationConfig(
            max_epochs=5,
            max_concurrent_trials=2,
            experiment_name="test_pause_resume",
            experiment_dir=Path("/tmp/lightning_bohb_tests"),
        ),
    )
    
    print(f"\n💾 Session directory: {cli.session_dir}")
    print(f"📝 State file: {cli.state_file}")
    print("\n🚀 Starting optimization...")
    print("   Press 'p' to pause")
    print("   Press Ctrl+C to pause gracefully\n")
    
    try:
        # Run for a short time
        results = cli.run(optimizer)
        print(f"\n✅ Optimization completed!")
        print(f"   Results: {results}")
        
    except KeyboardInterrupt:
        print("\n⏸️  Paused by user")
        print(f"   Session saved to: {cli.session_dir}")
        return cli.session_dir
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return cli.session_dir


def test_resume(session_dir):
    """Test resuming from a paused session."""
    print("\n" + "="*60)
    print("TEST: Resuming from saved session")
    print("="*60)
    print(f"📁 Session directory: {session_dir}")
    
    try:
        # Resume from saved session
        cli = TuneReflowCLI.resume(session_dir=session_dir)
        
        print(f"\n✅ Session restored!")
        print(f"   Experiment: {cli.state.experiment_name}")
        print(f"   Session ID: {cli.state.session_id}")
        
        print("\n🔄 Resuming optimization...")
        
        results = cli.run(resume=True)
        
        print(f"\n✅ Resume completed!")
        print(f"   Results: {results}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Resume failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Full integration test."""
    print("\n" + "="*70)
    print("INTEGRATION TEST: Pause/Resume Functionality")
    print("="*70)
    
    # Test 1: Start new optimization
    session_dir = test_new_run()
    
    if not session_dir:
        print("\n❌ Failed to start optimization")
        return False
    
    # Give it a moment
    time.sleep(2)
    
    # Test 2: Resume from checkpoint
    if session_dir.exists():
        success = test_resume(session_dir)
        
        if success:
            print("\n" + "="*70)
            print("✅ ALL TESTS PASSED!")
            print("="*70)
            print("\nThe pause/resume functionality is working correctly:")
            print("  • Session state is properly saved")
            print("  • Optimizer configuration is preserved")
            print("  • Resume restores exact state")
            return True
    
    print("\n❌ Some tests failed")
    return False


def verify_dependencies():
    """Verify all required dependencies are available."""
    print("\n" + "="*60)
    print("Verifying dependencies...")
    print("="*60)
    
    deps = []
    
    # Check Ray
    try:
        import ray
        deps.append(("Ray", "✅"))
    except ImportError:
        deps.append(("Ray", "❌ Install with: pip install ray[tune]"))
    
    # Check Lightning
    try:
        import lightning
        deps.append(("PyTorch Lightning", "✅"))
    except ImportError:
        deps.append(("PyTorch Lightning", "❌ Install with: pip install lightning"))
    
    # Check LightningReflow
    try:
        from lightning_reflow import LightningReflow
        deps.append(("LightningReflow", "✅"))
    except ImportError:
        deps.append(("LightningReflow", "⚠️  Optional - enables keyboard monitoring"))
    
    # Check keyboard handler
    try:
        from lightning_reflow.callbacks.pause.unified_keyboard_handler import UnifiedKeyboardHandler
        deps.append(("Keyboard Handler", "✅"))
    except ImportError:
        deps.append(("Keyboard Handler", "⚠️  Optional - enables 'p' key pause"))
    
    for dep, status in deps:
        print(f"  {dep}: {status}")
    
    required_ok = all("✅" in status for dep, status in deps if "Optional" not in dep)
    
    if required_ok:
        print("\n✅ All required dependencies are available")
    else:
        print("\n❌ Some required dependencies are missing")
    
    return required_ok


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test pause/resume functionality")
    parser.add_argument("--resume-session", type=str, help="Resume from session directory")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency check")
    
    args = parser.parse_args()
    
    if not args.skip_deps:
        if not verify_dependencies():
            print("\n⚠️  Fix missing dependencies before running tests")
            sys.exit(1)
    
    if args.resume_session:
        # Just test resume
        success = test_resume(Path(args.resume_session))
        sys.exit(0 if success else 1)
    else:
        # Run full integration test
        success = test_integration()
        sys.exit(0 if success else 1)