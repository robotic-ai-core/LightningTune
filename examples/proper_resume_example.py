#!/usr/bin/env python
"""
Proper resume example showing how session state is saved and restored.

This demonstrates the CORRECT way to handle pause/resume with full state preservation.
"""

import sys
import argparse
from pathlib import Path
from ray import tune

from lightning_bohb import ConfigDrivenOptimizer, SearchSpace, OptimizationConfig
from lightning_bohb.core.strategies import BOHBStrategy, OptunaStrategy
from lightning_bohb import TuneReflowCLI


class WorldModelSearchSpace(SearchSpace):
    """Your search space - will be serialized automatically."""
    
    def get_search_space(self):
        return {
            "model.init_args.num_layers": tune.choice([4, 6, 8]),
            "model.init_args.hidden_dim": tune.choice([512, 1024, 2048]),
            "model.init_args.learning_rate": tune.loguniform(1e-5, 1e-2),
            "data.init_args.batch_size": tune.choice([16, 32, 64]),
        }
    
    def get_metric_config(self):
        return {"metric": "val_loss", "mode": "min"}


def main():
    """Main entry point showing proper resume workflow."""
    
    parser = argparse.ArgumentParser(
        description="Optimization with PROPER pause/resume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PROPER RESUME WORKFLOW:

1. START NEW OPTIMIZATION:
   python proper_resume_example.py \\
       --experiment-name my_model \\
       --base-config config.yaml \\
       --strategy bohb

   Output:
   💾 Session state saved to: ./experiments/my_model/session_20240101_120000_abc123/session_state.pkl
   
2. PAUSE (press 'p' or Ctrl+C):
   
   Output:
   ✅ OPTIMIZATION PAUSED
   📊 Session: 20240101_120000_abc123
   💾 State saved to: ./experiments/my_model/session_20240101_120000_abc123/session_state.pkl
   
   📝 To resume this EXACT session with all settings, run:
   python proper_resume_example.py --resume-session ./experiments/my_model/session_20240101_120000_abc123

3. RESUME (using the session directory):
   python proper_resume_example.py \\
       --resume-session ./experiments/my_model/session_20240101_120000_abc123
   
   No need to specify:
   - Search space (restored from pickle)
   - Strategy (restored from pickle)
   - Base config (saved in state)
   - Any other settings (all preserved!)
        """
    )
    
    # Arguments for NEW run
    parser.add_argument("--experiment-name", type=str, help="Name for new experiment")
    parser.add_argument("--base-config", type=str, help="Config file for new run")
    parser.add_argument("--strategy", type=str, choices=["bohb", "optuna"], help="Strategy for new run")
    
    # Argument for RESUME
    parser.add_argument("--resume-session", type=str, help="Session directory to resume from")
    
    # Optional settings
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--max-trials", type=int, default=4)
    
    args = parser.parse_args()
    
    # CASE 1: RESUME FROM SESSION
    if args.resume_session:
        print("\n" + "="*70)
        print("♻️  RESUMING FROM SAVED SESSION")
        print("="*70)
        
        session_dir = Path(args.resume_session)
        if not session_dir.exists():
            print(f"❌ Session directory not found: {session_dir}")
            print("\nAvailable sessions:")
            
            # List available sessions
            exp_dir = Path("./experiments")
            if exp_dir.exists():
                for exp in exp_dir.iterdir():
                    if exp.is_dir():
                        sessions = list(exp.glob("session_*"))
                        if sessions:
                            print(f"\n  {exp.name}:")
                            for session in sessions:
                                state_file = session / "session_state.pkl"
                                if state_file.exists():
                                    print(f"    • {session}")
            sys.exit(1)
        
        # Resume using saved state
        cli = TuneReflowCLI.resume(session_dir=session_dir)
        
        print(f"📁 Session directory: {session_dir}")
        print(f"📊 Experiment: {cli.state.experiment_name}")
        print(f"⏰ Originally started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cli.state.created_at))}")
        
        if cli.state.paused_at:
            import time
            print(f"⏸️  Paused at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cli.state.paused_at))}")
        
        print("\n✅ All configuration restored from session state:")
        print("   • Search space: ✓")
        print("   • Strategy: ✓")
        print("   • Base config: ✓")
        print("   • Optimization config: ✓")
        print("="*70 + "\n")
        
        # Run - no optimizer needed!
        try:
            results = cli.run(resume=True)
            print_results(results)
        except KeyboardInterrupt:
            print("\n⏸️  Paused again. Use the same resume command to continue.")
    
    # CASE 2: START NEW OPTIMIZATION
    elif args.experiment_name and args.base_config:
        print("\n" + "="*70)
        print("🚀 STARTING NEW OPTIMIZATION")
        print("="*70)
        
        # Create strategy
        if args.strategy == "bohb":
            strategy = BOHBStrategy(
                grace_period=10,
                reduction_factor=3,
                max_t=args.max_epochs,
            )
            print(f"🎯 Strategy: BOHB (grace=10, reduction=3)")
        else:
            strategy = OptunaStrategy(
                n_startup_trials=10,
                num_samples=50,
            )
            print(f"🎯 Strategy: Optuna (n_startup=10)")
        
        # Create optimizer with all configuration
        optimizer = ConfigDrivenOptimizer(
            base_config_source=args.base_config,
            search_space=WorldModelSearchSpace(),  # This will be serialized!
            strategy=strategy,  # This will be serialized!
            optimization_config=OptimizationConfig(
                max_epochs=args.max_epochs,
                max_concurrent_trials=args.max_trials,
                experiment_name=args.experiment_name,
            ),
        )
        
        print(f"📊 Experiment: {args.experiment_name}")
        print(f"📁 Base config: {args.base_config}")
        print(f"🔧 Search space: WorldModelSearchSpace")
        print(f"⚙️  Max epochs: {args.max_epochs}")
        print(f"🔄 Max trials: {args.max_trials}")
        
        # Create CLI
        cli = TuneReflowCLI(
            experiment_name=args.experiment_name,
            experiment_dir="./experiments",
        )
        
        print(f"\n💾 Session will be saved to:")
        print(f"   {cli.session_dir}")
        print("="*70 + "\n")
        
        # Run optimization
        try:
            results = cli.run(optimizer)  # State saved automatically!
            print_results(results)
        except KeyboardInterrupt:
            print("\n⏸️  Optimization paused. Resume command printed above.")
    
    else:
        print("❌ Invalid arguments!")
        print("\nFor NEW run, provide:")
        print("  --experiment-name NAME --base-config CONFIG --strategy STRATEGY")
        print("\nFor RESUME, provide:")
        print("  --resume-session SESSION_DIR")
        parser.print_help()
        sys.exit(1)


def print_results(results):
    """Print optimization results."""
    print("\n" + "="*70)
    print("✅ OPTIMIZATION COMPLETED")
    print("="*70)
    
    best = results.get_best_result()
    print(f"\n🏆 Best val_loss: {best.metrics.get('val_loss', 'N/A')}")
    print("\n📊 Best configuration:")
    
    for key, value in best.config.items():
        print(f"   {key}: {value}")


def demo_proper_workflow():
    """
    Demonstrate the PROPER workflow with state preservation.
    """
    print("""
DEMONSTRATION: Proper Pause/Resume with State Preservation
==========================================================

THE PROBLEM with the original design:
--------------------------------------
python optimize.py --resume  # ❌ Where's the search space? Strategy? Config?

The search space and strategy are defined in CODE, not saved anywhere!
When you resume, the script has no idea what search space or strategy was used.

THE SOLUTION in v2:
-------------------
1. When starting, ALL configuration is automatically saved:
   - Search space object (pickled)
   - Strategy object with all parameters (pickled)
   - Base config path
   - Optimization config
   
2. Resume needs only the session directory:
   python optimize.py --resume-session ./experiments/my_opt/session_20240101_120000_abc123
   
3. Everything is restored from the saved state:
   - No need to redefine search space
   - No need to recreate strategy
   - No need to specify base config
   - Exact same configuration guaranteed!

EXAMPLE WORKFLOW:
-----------------

STEP 1 - Start new optimization:
$ python optimize.py --experiment-name my_model --base-config config.yaml --strategy bohb

Output:
> 💾 Session state saved to: ./experiments/my_model/session_20240101_120000_abc123/session_state.pkl
> Running optimization...
> [Press 'p' to pause]

STEP 2 - Pause (press 'p'):
> ✅ OPTIMIZATION PAUSED
> 📝 To resume this EXACT session with all settings, run:
> python optimize.py --resume-session ./experiments/my_model/session_20240101_120000_abc123

STEP 3 - Resume later (even on different machine):
$ python optimize.py --resume-session ./experiments/my_model/session_20240101_120000_abc123

Output:
> ♻️ Resuming session: 20240101_120000_abc123
> ✅ All configuration restored from session state:
>    • Search space: ✓
>    • Strategy: ✓ 
>    • Base config: ✓
>    • Optimization config: ✓
> Continuing optimization...

KEY INSIGHT:
------------
The session directory contains EVERYTHING needed to resume:
./experiments/my_model/session_20240101_120000_abc123/
├── session_state.pkl           # Complete pickled state
├── session_state.summary.yaml  # Human-readable summary
└── [Ray Tune files]            # Trial checkpoints

This is similar to how LightningReflow saves the complete training state,
but extended to handle the entire hyperparameter optimization session!
    """)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_proper_workflow()
    else:
        import time  # Needed for time formatting
        main()