"""
Example: Optimizing World Model with Lightning BOHB.

This example shows how to use the config-driven BOHB optimizer
with your World Model training pipeline.
"""

from pathlib import Path
from ray import tune

# Import the config-driven optimizer
from lightning_bohb import ConfigDrivenBOHBOptimizer, BOHBConfig, SearchSpace


class WorldModelSearchSpace(SearchSpace):
    """
    Define the hyperparameter search space for the World Model.
    
    All parameters use dot notation to override nested config values.
    """
    
    def get_search_space(self):
        return {
            # Model architecture
            "model.init_args.transformer_hparams.num_layers": tune.choice([4, 6, 8]),
            "model.init_args.transformer_hparams.num_heads": tune.choice([4, 8, 16]),
            "model.init_args.transformer_hparams.feedforward_dim": tune.choice([512, 1024, 2048]),
            
            # Learning rate and optimization
            "model.init_args.learning_rate": tune.loguniform(1e-5, 1e-2),
            "model.init_args.warmup_steps": tune.choice([500, 1000, 2000]),
            "model.init_args.use_one_cycle_lr": tune.choice([True, False]),
            
            # Regularization
            "model.init_args.no_op_regularizer_weight": tune.loguniform(0.01, 2.0),
            "model.init_args.temporal_consistency_weight": tune.loguniform(0.01, 2.0),
            "model.init_args.regularizer_rollout_length": tune.choice([2, 4, 8]),
            
            # Scheduled sampling
            "model.init_args.use_scheduled_sampling": tune.choice([True, False]),
            "model.init_args.scheduled_sampling_decay_steps": tune.choice([0.3, 0.5, 0.7]),
            "model.init_args.scheduled_sampling_decay_strategy": tune.choice(["linear", "exponential", "cosine"]),
            
            # Loss configuration
            "model.init_args.teacher_forced_loss_coefficient": tune.uniform(0.1, 1.0),
            "model.init_args.use_masked_prediction": tune.choice([True, False]),
            "model.init_args.masking_ratio": tune.uniform(0.1, 0.5),
            
            # Data configuration
            "data.init_args.batch_size": tune.choice([8, 16, 32]),
            
            # Trainer configuration
            "trainer.gradient_clip_val": tune.loguniform(0.1, 10.0),
            "trainer.accumulate_grad_batches": tune.choice([1, 2, 4]),
        }
    
    def get_metric_config(self):
        """Define the metric to optimize."""
        return {
            "metric": "val_total_loss",  # Or any metric your model logs
            "mode": "min"
        }
    
    def validate_config(self, config):
        """
        Optional: Validate that sampled configurations are sensible.
        """
        # Example: Ensure hidden_dim is divisible by num_heads
        if "model.init_args.transformer_hparams.num_heads" in config:
            # Add validation logic if needed
            pass
        return True
    
    def transform_config(self, config):
        """
        Optional: Apply transformations to the sampled config.
        
        For example, you might want to compute derived parameters.
        """
        # Example: Set feedforward_dim based on hidden_dim if not specified
        # This is where you could add complex parameter relationships
        return config


def main():
    """Run World Model hyperparameter optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="World Model BOHB Optimization")
    parser.add_argument(
        "--base-config",
        type=str,
        default="scripts/world_model_config.yaml",
        help="Path to base configuration file"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum epochs per trial"
    )
    parser.add_argument(
        "--grace-period",
        type=int,
        default=10,
        help="Minimum epochs before pruning"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum concurrent trials"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="world_model_bohb",
        help="Experiment name"
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="Time budget in hours"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous experiment"
    )
    
    args = parser.parse_args()
    
    # Configure BOHB
    bohb_config = BOHBConfig(
        max_epochs=args.max_epochs,
        grace_period=args.grace_period,
        reduction_factor=3,
        max_concurrent_trials=args.max_concurrent,
        experiment_name=args.experiment_name,
        experiment_dir=Path("./bohb_experiments"),
        resources_per_trial={
            "cpu": 4,
            "gpu": 1.0  # Adjust based on your hardware
        },
        verbose=1,
        log_to_file=True,
    )
    
    # Create optimizer
    optimizer = ConfigDrivenBOHBOptimizer(
        base_config_source=args.base_config,
        search_space=WorldModelSearchSpace(),
        bohb_config=bohb_config,
        # Optional: Additional LightningReflow kwargs
        lightning_reflow_kwargs={
            "seed_everything": 42,
        }
    )
    
    # Run optimization
    print("="*60)
    print("Starting World Model BOHB Optimization")
    print("="*60)
    print(f"Base config: {args.base_config}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Max epochs per trial: {args.max_epochs}")
    print(f"Max concurrent trials: {args.max_concurrent}")
    
    if args.time_budget:
        print(f"Time budget: {args.time_budget} hours")
    
    results = optimizer.run(
        resume=args.resume,
        time_budget_hrs=args.time_budget
    )
    
    # Analyze results
    print("\n" + "="*60)
    print("Optimization Complete!")
    print("="*60)
    
    analysis = optimizer.analyze_results()
    
    print(f"Total trials: {analysis['total_trials']}")
    print(f"Completed trials: {analysis['completed_trials']}")
    print(f"Early stopped: {analysis['early_stopped_trials']}")
    print(f"Best validation loss: {analysis['best_metric_value']:.4f}")
    
    if 'time_statistics' in analysis:
        print(f"Total compute time: {analysis['time_statistics']['total_hours']:.2f} hours")
    
    # Show best hyperparameters
    print("\nBest hyperparameter overrides:")
    best_config = optimizer.get_best_config()
    for param, value in best_config.items():
        print(f"  {param}: {value}")
    
    # Create production config
    production_config_path = optimizer.create_production_config()
    print(f"\nProduction config saved to: {production_config_path}")
    
    # Show parameter importance
    if 'parameter_importance' in analysis:
        print("\nParameter importance (correlation with metric):")
        for param, importance in list(analysis['parameter_importance'].items())[:10]:
            print(f"  {param}: {importance:.3f}")


if __name__ == "__main__":
    main()