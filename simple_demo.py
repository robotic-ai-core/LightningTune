#!/usr/bin/env python
"""
Simple demonstration of LightningTune hyperparameter optimization.

This shows the most basic usage for hyperparameter sweeping using a dictionary configuration.
"""

import tempfile
from pathlib import Path
import sys

# Add tests to path for fixtures
sys.path.insert(0, 'tests')

# Import LightningTune
from LightningTune import (
    ConfigDrivenOptimizer,
    SearchSpace,
    RandomSearchStrategy,
    OptimizationConfig,
)


class SimpleSearchSpace(SearchSpace):
    """Simple search space sweeping just 2 hyperparameters."""
    
    def get_search_space(self):
        from ray import tune
        return {
            # Just sweep learning rate and hidden size
            "model.init_args.learning_rate": tune.loguniform(1e-4, 1e-2),
            "model.init_args.hidden_dim": tune.choice([16, 32, 64]),
        }
    
    def get_metric_config(self):
        return {"metric": "val_loss", "mode": "min"}


def main():
    print("\n" + "="*60)
    print("🚀 LightningTune Hyperparameter Optimization Demo")
    print("     (Using Dictionary Configuration)")
    print("="*60 + "\n")
    
    # 1. Create base configuration as a dictionary (no file needed!)
    config_dict = {
        "model": {
            "class_path": "fixtures.dummy_model.DummyModel",
            "init_args": {
                "input_dim": 10,
                "hidden_dim": 32,
                "output_dim": 2,
                "learning_rate": 0.001,
            }
        },
        "data": {
            "class_path": "fixtures.dummy_model.DummyDataModule",
            "init_args": {
                "batch_size": 32,
                "num_samples": 100,
            }
        },
        "trainer": {
            "max_epochs": 3,
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "logger": False,
        }
    }
    
    # 2. Setup hyperparameter optimization with dict config
    optimizer = ConfigDrivenOptimizer(
        base_config_source=config_dict,  # Pass dict directly - no file needed!
        search_space=SimpleSearchSpace(),
        strategy=RandomSearchStrategy(num_samples=3),  # Just 3 trials for demo
        optimization_config=OptimizationConfig(
            max_epochs=3,
            max_concurrent_trials=2,
            experiment_name="simple_demo",
            experiment_dir=Path(tempfile.gettempdir()) / "raytune_demo",
        ),
    )
    
    print("📋 Configuration:")
    print(f"  - Config source: Python dictionary (no file needed!)")
    print(f"  - Search space: learning_rate, hidden_dim")
    print(f"  - Strategy: Random Search")
    print(f"  - Trials: 3")
    print(f"  - Max epochs per trial: 3")
    
    print("\n🔍 Running optimization...\n")
    
    # 3. Run optimization
    results = optimizer.run()
    
    # 4. Show results
    print("\n" + "="*60)
    print("✨ Optimization Complete!")
    print("="*60 + "\n")
    
    best_result = results.get_best_result(metric="val_loss", mode="min")
    best_config = optimizer.get_best_config()
    
    print(f"🏆 Best validation loss: {best_result.metrics['val_loss']:.4f}\n")
    print("🎯 Best hyperparameters:")
    print(f"  - learning_rate: {best_config['model.init_args.learning_rate']:.6f}")
    print(f"  - hidden_dim: {best_config['model.init_args.hidden_dim']}")
    
    print("\n📊 All trials:")
    for i, result in enumerate(results):
        lr = result.config['model.init_args.learning_rate']
        hidden = result.config['model.init_args.hidden_dim']
        loss = result.metrics.get('val_loss', float('inf'))
        print(f"  Trial {i+1}: lr={lr:.6f}, hidden={hidden}, val_loss={loss:.4f}")
    
    # Cleanup - no file to delete since we're using dict!
    import ray
    if ray.is_initialized():
        ray.shutdown()
    
    print("\n✅ Demo complete! (No temporary files were created!)")


if __name__ == "__main__":
    main()