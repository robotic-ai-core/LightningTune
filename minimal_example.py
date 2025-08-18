#!/usr/bin/env python
"""
Minimal working example of LightningTune hyperparameter sweeping.

This example demonstrates:
1. Creating a simple Lightning model
2. Defining a search space
3. Running hyperparameter optimization with BOHB
4. Getting the best configuration
"""

import tempfile
import yaml
from pathlib import Path

# Import LightningTune components
from LightningTune import (
    ConfigDrivenOptimizer,
    SearchSpace,
    BOHBStrategy,
    OptimizationConfig,
)

# For this example, we'll use the test fixtures
import sys
sys.path.insert(0, 'tests')
from fixtures.dummy_model import DummyModel, DummyDataModule


class MinimalSearchSpace(SearchSpace):
    """Define the hyperparameter search space."""
    
    def get_search_space(self):
        """Define which hyperparameters to sweep."""
        from ray import tune
        return {
            # Sweep learning rate
            "model.init_args.learning_rate": tune.loguniform(1e-4, 1e-2),
            # Sweep hidden dimension
            "model.init_args.hidden_dim": tune.choice([16, 32, 64]),
            # Sweep batch size
            "data.init_args.batch_size": tune.choice([16, 32]),
            # Sweep dropout
            "model.init_args.dropout": tune.uniform(0.1, 0.5),
        }
    
    def get_metric_config(self):
        """Define what metric to optimize."""
        return {
            "metric": "val_loss",
            "mode": "min"  # minimize validation loss
        }


def create_base_config():
    """Create the base configuration file."""
    config = {
        "model": {
            "class_path": "fixtures.dummy_model.DummyModel",
            "init_args": {
                "input_dim": 10,
                "hidden_dim": 32,  # Will be overridden by search
                "output_dim": 2,
                "learning_rate": 0.001,  # Will be overridden by search
                "dropout": 0.1,  # Will be overridden by search
            }
        },
        "data": {
            "class_path": "fixtures.dummy_model.DummyDataModule",
            "init_args": {
                "batch_size": 32,  # Will be overridden by search
                "num_samples": 200,
                "input_dim": 10,
                "num_classes": 2,
            }
        },
        "trainer": {
            "max_epochs": 5,  # Short for demo
            "accelerator": "cpu",
            "devices": 1,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
        }
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        return Path(f.name)


def main():
    """Run the hyperparameter sweep."""
    print("🚀 LightningTune Minimal Hyperparameter Sweep Example")
    print("=" * 60)
    
    # Create base configuration
    config_path = create_base_config()
    print(f"✅ Created base config: {config_path}")
    
    # Define search space
    search_space = MinimalSearchSpace()
    print("✅ Defined search space for: learning_rate, hidden_dim, batch_size, dropout")
    
    # Choose optimization strategy
    # Using RandomSearchStrategy for simplicity
    from LightningTune import RandomSearchStrategy
    strategy = RandomSearchStrategy(
        num_samples=4,  # Number of trials to run
        use_early_stopping=True,
        grace_period=2,
    )
    print("✅ Using Random Search strategy with early stopping")
    
    # Configure optimization
    optimization_config = OptimizationConfig(
        max_epochs=5,
        max_concurrent_trials=2,  # Run 2 trials in parallel
        experiment_name="minimal_example",
        experiment_dir=Path(tempfile.gettempdir()) / "lightning_raytune_demo",
        verbose=1,
    )
    print(f"✅ Will run trials with max {optimization_config.max_concurrent_trials} concurrent trials")
    
    # Create optimizer
    optimizer = ConfigDrivenOptimizer(
        base_config_source=config_path,
        search_space=search_space,
        strategy=strategy,
        optimization_config=optimization_config,
    )
    print("✅ Created optimizer")
    
    print("\n" + "=" * 60)
    print("🔍 Starting hyperparameter optimization...")
    print("=" * 60 + "\n")
    
    # Run optimization
    results = optimizer.run()
    
    print("\n" + "=" * 60)
    print("📊 Optimization Results")
    print("=" * 60)
    
    # Get best configuration
    best_config = optimizer.get_best_config()
    best_result = results.get_best_result(metric="val_loss", mode="min")
    
    print(f"\n✨ Best validation loss: {best_result.metrics['val_loss']:.4f}")
    print("\n🎯 Best hyperparameters found:")
    for key, value in best_config.items():
        param_name = key.split('.')[-1]  # Get just the parameter name
        if isinstance(value, float):
            print(f"  - {param_name}: {value:.6f}")
        else:
            print(f"  - {param_name}: {value}")
    
    # Analyze results
    analysis = optimizer.analyze_results()
    print(f"\n📈 Trials Summary:")
    print(f"  - Total trials run: {analysis['total_trials']}")
    print(f"  - Completed trials: {analysis['completed_trials']}")
    
    if 'metric_statistics' in analysis:
        stats = analysis['metric_statistics']
        print(f"\n📊 Validation Loss Statistics:")
        print(f"  - Best:   {stats['min']:.4f}")
        print(f"  - Mean:   {stats['mean']:.4f}")
        print(f"  - Worst:  {stats['max']:.4f}")
        print(f"  - StdDev: {stats['std']:.4f}")
    
    # Save best configuration for production
    production_config_path = optimizer.create_production_config()
    print(f"\n💾 Saved production config to: {production_config_path}")
    
    # Cleanup
    config_path.unlink()
    
    print("\n✅ Hyperparameter sweep complete!")
    
    # Shutdown Ray
    import ray
    if ray.is_initialized():
        ray.shutdown()
    
    return results


if __name__ == "__main__":
    main()