"""
Example: Comparing different optimization strategies on the same problem.

This example shows how to use different optimization algorithms 
(BOHB, Optuna, Random Search) with the same search space and config.
"""

from pathlib import Path
from ray import tune

from lightning_bohb import (
    ConfigDrivenOptimizer,
    OptimizationConfig,
    SearchSpace,
)


class ExampleSearchSpace(SearchSpace):
    """Example search space for comparison."""
    
    def get_search_space(self):
        return {
            "model.init_args.learning_rate": tune.loguniform(1e-5, 1e-2),
            "model.init_args.hidden_dim": tune.choice([256, 512, 1024, 2048]),
            "model.init_args.num_layers": tune.choice([2, 4, 6, 8]),
            "model.init_args.dropout": tune.uniform(0.0, 0.5),
            "model.init_args.weight_decay": tune.loguniform(1e-6, 1e-2),
            "data.init_args.batch_size": tune.choice([16, 32, 64]),
            "trainer.accumulate_grad_batches": tune.choice([1, 2, 4]),
        }
    
    def get_metric_config(self):
        return {"metric": "val_loss", "mode": "min"}


def run_strategy_comparison(base_config_source: str, strategies: list = None):
    """
    Run optimization with different strategies and compare results.
    
    Args:
        base_config_source: Path to base configuration file
        strategies: List of strategies to compare (default: all)
    """
    if strategies is None:
        strategies = ["bohb", "optuna", "random"]
    
    # Common configuration for all strategies
    optimization_config = OptimizationConfig(
        max_epochs=50,
        max_concurrent_trials=4,
        experiment_dir=Path("./strategy_comparison"),
        resources_per_trial={"cpu": 4, "gpu": 0.5},
        seed=42,
    )
    
    # Same search space for all
    search_space = ExampleSearchSpace()
    
    results = {}
    
    for strategy_name in strategies:
        print(f"\n{'='*60}")
        print(f"Running optimization with {strategy_name.upper()}")
        print('='*60)
        
        # Strategy-specific configuration
        strategy_config = {}
        
        if strategy_name == "bohb":
            strategy_config = {
                "grace_period": 5,
                "reduction_factor": 3,
            }
        elif strategy_name == "optuna":
            strategy_config = {
                "n_startup_trials": 10,
                "use_pruner": True,
                "pruner_type": "median",
            }
        elif strategy_name == "random":
            strategy_config = {
                "use_early_stopping": True,
            }
        
        # Update experiment name
        optimization_config.experiment_name = f"comparison_{strategy_name}"
        
        # Create optimizer with the strategy
        optimizer = ConfigDrivenOptimizer(
            base_config_source=base_config_source,
            search_space=search_space,
            strategy=strategy_name,
            optimization_config=optimization_config,
            strategy_config=strategy_config,
        )
        
        # Run optimization
        try:
            result = optimizer.run(time_budget_hrs=1.0)  # 1 hour per strategy
            analysis = optimizer.analyze_results()
            results[strategy_name] = analysis
            
            print(f"\n{strategy_name.upper()} Results:")
            print(f"  Best metric: {analysis['best_metric_value']:.4f}")
            print(f"  Total trials: {analysis['total_trials']}")
            
            if 'time_statistics' in analysis:
                print(f"  Time: {analysis['time_statistics']['total_hours']:.2f} hours")
            
        except Exception as e:
            print(f"Error with {strategy_name}: {e}")
            results[strategy_name] = None
    
    # Compare results
    print("\n" + "="*60)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*60)
    
    for strategy_name, analysis in results.items():
        if analysis:
            print(f"\n{strategy_name.upper()}:")
            print(f"  Best loss: {analysis['best_metric_value']:.4f}")
            print(f"  Trials: {analysis['total_trials']}")
            
            if 'metric_statistics' in analysis:
                stats = analysis['metric_statistics']
                print(f"  Mean loss: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Find winner
    best_strategy = None
    best_value = float('inf')
    
    for strategy_name, analysis in results.items():
        if analysis and analysis['best_metric_value'] < best_value:
            best_value = analysis['best_metric_value']
            best_strategy = strategy_name
    
    if best_strategy:
        print(f"\n🏆 Best strategy: {best_strategy.upper()} with loss {best_value:.4f}")
    
    return results


def main():
    """Run strategy comparison example."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare HPO Strategies")
    parser.add_argument(
        "--base-config",
        type=str,
        required=True,
        help="Path to base configuration file"
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["bohb", "optuna", "random"],
        help="Strategies to compare"
    )
    
    args = parser.parse_args()
    
    # Run comparison
    results = run_strategy_comparison(
        base_config_source=args.base_config,
        strategies=args.strategies
    )
    
    # Save comparison results
    import yaml
    comparison_path = Path("./strategy_comparison/comparison_summary.yaml")
    comparison_path.parent.mkdir(exist_ok=True)
    
    with open(comparison_path, 'w') as f:
        # Convert to serializable format
        summary = {}
        for strategy, analysis in results.items():
            if analysis:
                summary[strategy] = {
                    "best_value": float(analysis['best_metric_value']),
                    "total_trials": int(analysis['total_trials']),
                    "best_config": analysis['best_overrides'],
                }
        yaml.dump(summary, f)
    
    print(f"\nComparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()