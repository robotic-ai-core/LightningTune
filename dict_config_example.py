#!/usr/bin/env python
"""
Example demonstrating dict configuration support in LightningTune.

This shows how you can use a Python dict directly instead of a YAML/JSON file
for the base configuration, making it easier to programmatically generate configs.
"""

import sys
sys.path.insert(0, 'tests')

from LightningTune import (
    ConfigDrivenOptimizer,
    SearchSpace,
    RandomSearchStrategy,
    OptimizationConfig,
)
from pathlib import Path
import tempfile


class SimpleSearchSpace(SearchSpace):
    """Define hyperparameter search space."""
    
    def get_search_space(self):
        from ray import tune
        return {
            "model.init_args.learning_rate": tune.loguniform(1e-4, 1e-2),
            "model.init_args.hidden_dim": tune.choice([16, 32, 64]),
            "model.init_args.dropout": tune.uniform(0.1, 0.5),
        }
    
    def get_metric_config(self):
        return {"metric": "val_loss", "mode": "min"}


def run_with_dict_config():
    """Run optimization using a dictionary configuration."""
    print("\n" + "="*60)
    print("🔧 Using Dictionary Configuration (No File Required!)")
    print("="*60 + "\n")
    
    # Define configuration as a Python dictionary
    # No need to create a temporary file!
    config_dict = {
        "model": {
            "class_path": "fixtures.dummy_model.DummyModel",
            "init_args": {
                "input_dim": 10,
                "hidden_dim": 32,
                "output_dim": 2,
                "learning_rate": 0.001,
                "dropout": 0.1,
            }
        },
        "data": {
            "class_path": "fixtures.dummy_model.DummyDataModule", 
            "init_args": {
                "batch_size": 32,
                "num_samples": 100,
                "input_dim": 10,
                "num_classes": 2,
            }
        },
        "trainer": {
            "max_epochs": 2,
            "accelerator": "cpu",
            "devices": 1,
            "enable_progress_bar": False,
            "logger": False,
        }
    }
    
    print("✅ Created configuration dictionary directly in Python")
    print(f"   Config has {len(config_dict)} top-level keys: {list(config_dict.keys())}")
    
    # Create optimizer with dict config
    optimizer = ConfigDrivenOptimizer(
        base_config_source=config_dict,  # Pass dict directly!
        search_space=SimpleSearchSpace(),
        strategy=RandomSearchStrategy(num_samples=3),
        optimization_config=OptimizationConfig(
            max_epochs=2,
            max_concurrent_trials=2,
            experiment_name="dict_config_demo",
            experiment_dir=Path(tempfile.gettempdir()) / "raytune_dict_demo",
        ),
    )
    
    print("✅ Created optimizer with dictionary configuration")
    print("\n🔍 Running optimization...\n")
    
    # Run optimization
    results = optimizer.run()
    
    # Show results
    print("\n" + "="*60)
    print("✨ Results")
    print("="*60 + "\n")
    
    best_result = results.get_best_result(metric="val_loss", mode="min")
    best_config = optimizer.get_best_config()
    
    print(f"🏆 Best validation loss: {best_result.metrics['val_loss']:.4f}")
    print("\n🎯 Best hyperparameters:")
    for key, value in best_config.items():
        param_name = key.split('.')[-1]
        if isinstance(value, float):
            print(f"  - {param_name}: {value:.6f}")
        else:
            print(f"  - {param_name}: {value}")
    
    # Cleanup
    import ray
    if ray.is_initialized():
        ray.shutdown()


def run_with_file_config():
    """Run optimization using a file configuration (traditional way)."""
    print("\n" + "="*60)
    print("📄 Using File Configuration (Traditional Way)")
    print("="*60 + "\n")
    
    import yaml
    
    # Create config file
    config_dict = {
        "model": {
            "class_path": "fixtures.dummy_model.DummyModel",
            "init_args": {
                "input_dim": 10,
                "hidden_dim": 32,
                "output_dim": 2,
                "learning_rate": 0.001,
                "dropout": 0.1,
            }
        },
        "data": {
            "class_path": "fixtures.dummy_model.DummyDataModule",
            "init_args": {
                "batch_size": 32,
                "num_samples": 100,
                "input_dim": 10,
                "num_classes": 2,
            }
        },
        "trainer": {
            "max_epochs": 2,
            "accelerator": "cpu",
            "devices": 1,
            "enable_progress_bar": False,
            "logger": False,
        }
    }
    
    # Save to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = Path(f.name)
    
    print(f"✅ Created configuration file: {config_path}")
    
    try:
        # Create optimizer with file path
        optimizer = ConfigDrivenOptimizer(
            base_config_source=config_path,  # Pass file path
            search_space=SimpleSearchSpace(),
            strategy=RandomSearchStrategy(num_samples=3),
            optimization_config=OptimizationConfig(
                max_epochs=2,
                max_concurrent_trials=2,
                experiment_name="file_config_demo",
                experiment_dir=Path(tempfile.gettempdir()) / "raytune_file_demo",
            ),
        )
        
        print("✅ Created optimizer with file configuration")
        print("\n🔍 Running optimization...\n")
        
        # Run optimization
        results = optimizer.run()
        
        # Show results
        print("\n" + "="*60)
        print("✨ Results")
        print("="*60 + "\n")
        
        best_result = results.get_best_result(metric="val_loss", mode="min")
        best_config = optimizer.get_best_config()
        
        print(f"🏆 Best validation loss: {best_result.metrics['val_loss']:.4f}")
        print("\n🎯 Best hyperparameters:")
        for key, value in best_config.items():
            param_name = key.split('.')[-1]
            if isinstance(value, float):
                print(f"  - {param_name}: {value:.6f}")
            else:
                print(f"  - {param_name}: {value}")
        
    finally:
        # Cleanup
        config_path.unlink()
        import ray
        if ray.is_initialized():
            ray.shutdown()


def main():
    """Run both examples to show the difference."""
    print("\n" + "="*70)
    print("🚀 LightningTune Dict Configuration Support Demo")
    print("="*70)
    print("\nThis demo shows two ways to provide base configuration:")
    print("1. Using a Python dictionary directly (NEW!)")
    print("2. Using a YAML/JSON file path (traditional)")
    
    # Run with dict config (new way)
    run_with_dict_config()
    
    print("\n" + "="*70)
    print("Now let's compare with the traditional file-based approach...")
    print("="*70)
    
    # Run with file config (traditional way)
    run_with_file_config()
    
    print("\n" + "="*70)
    print("🎉 Demo Complete!")
    print("="*70)
    print("\n✅ Both methods work identically!")
    print("✅ Dict config is more convenient for programmatic use")
    print("✅ File config is better for persistent configurations")
    print("✅ Choose based on your use case!")


if __name__ == "__main__":
    main()