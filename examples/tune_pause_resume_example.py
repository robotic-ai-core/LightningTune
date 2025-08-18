#!/usr/bin/env python
"""
Example: Interactive pause/resume for Ray Tune optimization.

This example demonstrates how to use TuneReflowCLI to add LightningReflow-style
pause/resume capabilities to Ray Tune hyperparameter optimization.

Usage:
    # Start optimization
    python tune_pause_resume_example.py \
        --experiment-name world_model_opt \
        --base-config config.yaml
    
    # While running, press 'p' to pause at next validation
    # The script will print the resume command
    
    # Resume optimization
    python tune_pause_resume_example.py \
        --experiment-name world_model_opt \
        --base-config config.yaml \
        --resume
"""

import argparse
from pathlib import Path
from ray import tune

from lightning_bohb import ConfigDrivenOptimizer, SearchSpace, OptimizationConfig
from lightning_bohb.core.strategies import (
    BOHBStrategy,
    OptunaStrategy,
    RandomSearchStrategy,
)
from lightning_bohb import TuneReflowCLI


class WorldModelSearchSpace(SearchSpace):
    """Example search space for world model optimization."""
    
    def get_search_space(self):
        return {
            # Architecture
            "model.init_args.num_layers": tune.choice([4, 6, 8]),
            "model.init_args.hidden_dim": tune.choice([512, 1024, 2048]),
            "model.init_args.num_heads": tune.choice([4, 8, 16]),
            
            # Optimization
            "model.init_args.learning_rate": tune.loguniform(1e-5, 1e-2),
            "model.init_args.warmup_steps": tune.choice([500, 1000, 2000]),
            "model.init_args.weight_decay": tune.loguniform(1e-6, 1e-2),
            
            # Regularization
            "model.init_args.dropout": tune.uniform(0.0, 0.3),
            "model.init_args.label_smoothing": tune.uniform(0.0, 0.2),
            
            # Training
            "data.init_args.batch_size": tune.choice([16, 32, 64]),
            "trainer.accumulate_grad_batches": tune.choice([1, 2, 4]),
        }
    
    def get_metric_config(self):
        return {"metric": "val_loss", "mode": "min"}


def main():
    """Main entry point with interactive pause/resume."""
    parser = argparse.ArgumentParser(
        description="Interactive Ray Tune optimization with pause/resume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start new optimization
  %(prog)s --experiment-name my_exp --base-config config.yaml
  
  # Resume paused optimization
  %(prog)s --experiment-name my_exp --base-config config.yaml --resume
  
  # Use different strategy
  %(prog)s --experiment-name my_exp --base-config config.yaml --strategy optuna
  
Interactive controls:
  Press 'p' - Pause at next validation boundary
  Press 'q' - Quit immediately (saves checkpoint)
  Press Ctrl+C - Graceful pause (same as 'p')
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Name of the experiment (used for saving/resuming)"
    )
    parser.add_argument(
        "--base-config",
        type=str,
        required=True,
        help="Path to base Lightning configuration file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["bohb", "optuna", "random"],
        default="bohb",
        help="Optimization strategy to use"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="./experiments",
        help="Directory for experiment results"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum epochs per trial"
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=4,
        help="Maximum concurrent trials"
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        help="Time budget in hours (optional)"
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Disable interactive pause functionality"
    )
    parser.add_argument(
        "--resume",
        type=str,
        nargs='?',
        const='auto',
        help="Resume from previous run (optionally specify path)"
    )
    
    args = parser.parse_args()
    
    # Create TuneReflowCLI for interactive control
    cli = TuneReflowCLI(
        experiment_name=args.experiment_name,
        experiment_dir=args.experiment_dir,
        enable_pause=not args.no_pause,
        verbose=True,
    )
    
    print("\n" + "="*70)
    print("🚀 RAY TUNE OPTIMIZATION WITH INTERACTIVE PAUSE/RESUME")
    print("="*70)
    print(f"📊 Experiment: {args.experiment_name}")
    print(f"📁 Directory: {args.experiment_dir}")
    print(f"🎯 Strategy: {args.strategy.upper()}")
    print(f"⚙️  Max epochs: {args.max_epochs}")
    print(f"🔄 Max concurrent trials: {args.max_trials}")
    
    if args.resume:
        if args.resume == 'auto':
            resume_path = Path(args.experiment_dir) / args.experiment_name
        else:
            resume_path = Path(args.resume)
        print(f"♻️  Resuming from: {resume_path}")
    else:
        print("🆕 Starting fresh optimization")
    
    print("="*70 + "\n")
    
    # Create strategy based on selection
    if args.strategy == "bohb":
        strategy = BOHBStrategy(
            grace_period=10,
            reduction_factor=3,
            max_t=args.max_epochs,
        )
        print("Using BOHB: Bayesian Optimization with aggressive early stopping")
        
    elif args.strategy == "optuna":
        strategy = OptunaStrategy(
            n_startup_trials=10,
            use_pruner=True,
            pruner_type="median",
            num_samples=50,
        )
        print("Using Optuna: Tree-structured Parzen Estimator with median pruning")
        
    else:  # random
        strategy = RandomSearchStrategy(
            num_samples=30,
            use_early_stopping=True,
        )
        print("Using Random Search: Fast exploration with early stopping")
    
    # Create optimization config
    optimization_config = OptimizationConfig(
        max_epochs=args.max_epochs,
        max_concurrent_trials=args.max_trials,
        experiment_name=args.experiment_name,
        experiment_dir=Path(args.experiment_dir),
        time_budget_hrs=args.time_budget,
        resources_per_trial={"cpu": 4, "gpu": 0.5},
    )
    
    # Create optimizer
    optimizer = ConfigDrivenOptimizer(
        base_config_source=args.base_config,
        search_space=WorldModelSearchSpace(),
        strategy=strategy,
        optimization_config=optimization_config,
    )
    
    # Run optimization with CLI (handles pause/resume)
    try:
        results = cli.run(
            optimizer,
            resume=(args.resume is not None),
            resume_path=args.resume if args.resume != 'auto' else None,
        )
        
        # Print results
        print("\n" + "="*70)
        print("✅ OPTIMIZATION COMPLETED")
        print("="*70)
        
        best_result = results.get_best_result()
        print(f"\n🏆 Best {optimization_config.metric}: {best_result.metrics[optimization_config.metric]:.4f}")
        print("\n📊 Best hyperparameters:")
        for key, value in best_result.config.items():
            print(f"   {key}: {value}")
        
        # Save final summary
        summary_path = Path(args.experiment_dir) / args.experiment_name / "final_summary.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write(f"Experiment: {args.experiment_name}\n")
            f.write(f"Strategy: {args.strategy}\n")
            f.write(f"Best {optimization_config.metric}: {best_result.metrics[optimization_config.metric]:.4f}\n")
            f.write("\nBest hyperparameters:\n")
            for key, value in best_result.config.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"\n📝 Summary saved to: {summary_path}")
        
    except KeyboardInterrupt:
        print("\n⏸️  Optimization paused by user")
        # CLI will handle printing resume instructions
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        raise


def demo_mode():
    """
    Demo mode to show the pause/resume workflow without actual training.
    """
    print("\n" + "="*70)
    print("🎮 DEMO: Interactive Pause/Resume Workflow")
    print("="*70)
    
    print("\n1️⃣  Starting optimization:")
    print("   $ python tune_pause_resume_example.py \\")
    print("       --experiment-name world_model \\")
    print("       --base-config config.yaml")
    
    print("\n2️⃣  While running, you'll see:")
    print("   🎮 Interactive mode enabled:")
    print("   Press 'p' to pause at next validation")
    print("   Press 'q' to quit immediately")
    print("   Press Ctrl+C to pause gracefully")
    
    print("\n3️⃣  After pressing 'p', the system will:")
    print("   - Wait for all trials to reach validation boundaries")
    print("   - Save checkpoints for each trial")
    print("   - Print the resume command")
    
    print("\n4️⃣  Example output after pause:")
    print("   " + "="*60)
    print("   ✅ OPTIMIZATION PAUSED SUCCESSFULLY")
    print("   " + "="*60)
    print("   📊 Experiment: world_model")
    print("   📁 Saved to: ./experiments/world_model")
    print("   🔄 Trials paused: 4/4")
    print("")
    print("   📝 To resume this optimization session, run:")
    print("")
    print("   python tune_pause_resume_example.py \\")
    print("       --experiment-name world_model \\")
    print("       --base-config config.yaml \\")
    print("       --resume")
    
    print("\n5️⃣  Resume will:")
    print("   - Load all trial checkpoints")
    print("   - Continue from exact pause points")
    print("   - Preserve all optimization history")
    
    print("\n" + "="*70)
    print("💡 This provides the same experience as LightningReflow's pause/resume")
    print("   but for entire Ray Tune optimization sessions!")
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    # Check for demo mode
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_mode()
    else:
        main()