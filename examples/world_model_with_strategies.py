"""
Example: World Model optimization with different strategies.

This shows how to optimize your World Model using different algorithms
based on your needs and constraints.
"""

from pathlib import Path
from ray import tune

from lightning_bohb import (
    ConfigDrivenOptimizer,
    OptimizationConfig,
    SearchSpace,
    StrategyFactory,
)


class WorldModelSearchSpace(SearchSpace):
    """Search space for World Model hyperparameters."""
    
    def get_search_space(self):
        return {
            # Architecture
            "model.init_args.transformer_hparams.num_layers": tune.choice([4, 6, 8]),
            "model.init_args.transformer_hparams.num_heads": tune.choice([4, 8, 16]),
            "model.init_args.transformer_hparams.feedforward_dim": tune.choice([512, 1024, 2048]),
            
            # Optimization
            "model.init_args.learning_rate": tune.loguniform(1e-5, 1e-2),
            "model.init_args.warmup_steps": tune.choice([500, 1000, 2000]),
            
            # Regularization
            "model.init_args.no_op_regularizer_weight": tune.loguniform(0.01, 2.0),
            "model.init_args.temporal_consistency_weight": tune.loguniform(0.01, 2.0),
            
            # Scheduled sampling
            "model.init_args.use_scheduled_sampling": tune.choice([True, False]),
            "model.init_args.scheduled_sampling_decay_steps": tune.choice([0.3, 0.5, 0.7]),
            
            # Training
            "data.init_args.batch_size": tune.choice([8, 16, 32]),
            "trainer.accumulate_grad_batches": tune.choice([1, 2, 4]),
        }
    
    def get_metric_config(self):
        return {"metric": "val_total_loss", "mode": "min"}


def optimize_with_bohb(base_config_source: str):
    """
    Use BOHB for thorough exploration with Bayesian optimization.
    
    Best when:
    - You have moderate compute resources
    - Training is expensive (>30 min per trial)
    - You want the best possible results
    - Parameter landscape is smooth
    """
    print("Using BOHB Strategy - Best for expensive evaluations")
    
    optimizer = ConfigDrivenOptimizer(
        base_config_source=base_config_source,
        search_space=WorldModelSearchSpace(),
        strategy="bohb",
        optimization_config=OptimizationConfig(
            max_epochs=100,
            max_concurrent_trials=4,
            experiment_name="world_model_bohb",
        ),
        strategy_config={
            "grace_period": 10,  # Minimum epochs before pruning
            "reduction_factor": 3,  # Aggressive pruning
        }
    )
    
    return optimizer.run()


def optimize_with_optuna(base_config_source: str):
    """
    Use Optuna for balanced exploration/exploitation.
    
    Best when:
    - You want good results quickly
    - You have categorical parameters
    - Training is moderately expensive (5-30 min per trial)
    - You want interpretable parameter importance
    """
    print("Using Optuna Strategy - Good balance and interpretability")
    
    optimizer = ConfigDrivenOptimizer(
        base_config_source=base_config_source,
        search_space=WorldModelSearchSpace(),
        strategy="optuna",
        optimization_config=OptimizationConfig(
            max_epochs=50,
            max_concurrent_trials=8,
            experiment_name="world_model_optuna",
            num_samples=100,  # Optuna needs explicit samples
        ),
        strategy_config={
            "n_startup_trials": 20,  # Random exploration first
            "use_pruner": True,
            "pruner_type": "median",  # Prune below median
        }
    )
    
    return optimizer.run()


def optimize_with_random_search(base_config_source: str):
    """
    Use Random Search for initial exploration.
    
    Best when:
    - You're just starting and want a baseline
    - You have lots of parallel resources
    - Training is fast (<5 min per trial)
    - You want to understand parameter sensitivity
    """
    print("Using Random Search - Fast initial exploration")
    
    optimizer = ConfigDrivenOptimizer(
        base_config_source=base_config_source,
        search_space=WorldModelSearchSpace(),
        strategy="random",
        optimization_config=OptimizationConfig(
            max_epochs=20,  # Shorter trials
            max_concurrent_trials=16,  # More parallel trials
            experiment_name="world_model_random",
            num_samples=50,  # Fixed number of trials
        ),
        strategy_config={
            "use_early_stopping": True,  # Stop bad trials early
        }
    )
    
    return optimizer.run()


def optimize_with_pbt(base_config_source: str):
    """
    Use Population Based Training for adaptive schedules.
    
    Best when:
    - You want to tune learning rate schedules
    - Training is very long (>2 hours per trial)
    - You want parameters to adapt during training
    - You have stable training that can handle perturbations
    """
    print("Using PBT Strategy - Adaptive hyperparameter schedules")
    
    optimizer = ConfigDrivenOptimizer(
        base_config_source=base_config_source,
        search_space=WorldModelSearchSpace(),
        strategy="pbt",
        optimization_config=OptimizationConfig(
            max_epochs=200,  # Long training for adaptation
            max_concurrent_trials=8,
            experiment_name="world_model_pbt",
        ),
        strategy_config={
            "perturbation_interval": 10,  # Perturb every 10 epochs
            "population_size": 8,
            "hyperparam_mutations": {
                # Parameters that can change during training
                "model.init_args.learning_rate": tune.loguniform(1e-5, 1e-2),
                "model.init_args.no_op_regularizer_weight": tune.uniform(0.1, 2.0),
            }
        }
    )
    
    return optimizer.run()


def optimize_progressive(base_config_source: str):
    """
    Progressive optimization: Random → Optuna → BOHB.
    
    A practical workflow that balances exploration and exploitation.
    """
    print("Progressive Optimization Strategy")
    
    search_space = WorldModelSearchSpace()
    
    # Stage 1: Quick random exploration (1 hour)
    print("\n📊 Stage 1: Random Exploration")
    optimizer = ConfigDrivenOptimizer(
        base_config_source=base_config_source,
        search_space=search_space,
        strategy="random",
        optimization_config=OptimizationConfig(
            max_epochs=10,
            max_concurrent_trials=16,
            experiment_name="world_model_stage1_random",
            time_budget_hrs=1.0,
        ),
    )
    
    results_random = optimizer.run()
    best_random = optimizer.get_best_config()
    
    # Stage 2: Refine with Optuna (2 hours)
    print("\n🎯 Stage 2: Optuna Refinement")
    
    # Narrow search space based on random results
    refined_search = {
        **search_space.get_search_space(),
        # Could narrow ranges based on best_random insights
    }
    
    optimizer = ConfigDrivenOptimizer(
        base_config_source=base_config_source,
        search_space=refined_search,
        strategy="optuna",
        optimization_config=OptimizationConfig(
            max_epochs=30,
            max_concurrent_trials=8,
            experiment_name="world_model_stage2_optuna",
            time_budget_hrs=2.0,
        ),
        strategy_config={"use_pruner": True}
    )
    
    results_optuna = optimizer.run()
    best_optuna = optimizer.get_best_config()
    
    # Stage 3: Final optimization with BOHB (4 hours)
    print("\n🚀 Stage 3: BOHB Final Optimization")
    
    # Further refined search space
    final_search = {
        **search_space.get_search_space(),
        # Could use even narrower ranges
    }
    
    optimizer = ConfigDrivenOptimizer(
        base_config_source=base_config_source,
        search_space=final_search,
        strategy="bohb",
        optimization_config=OptimizationConfig(
            max_epochs=100,
            max_concurrent_trials=4,
            experiment_name="world_model_stage3_bohb",
            time_budget_hrs=4.0,
        ),
        strategy_config={
            "grace_period": 10,
            "reduction_factor": 3,
        }
    )
    
    results_final = optimizer.run()
    
    # Create final production config
    production_config = optimizer.create_production_config()
    print(f"\n✅ Production config created: {production_config}")
    
    return results_final


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="World Model Optimization with Strategies")
    parser.add_argument(
        "--base-config",
        type=str,
        default="scripts/world_model_config.yaml",
        help="Path to base configuration"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["bohb", "optuna", "random", "pbt", "progressive"],
        default="bohb",
        help="Optimization strategy to use"
    )
    
    args = parser.parse_args()
    
    # Run selected strategy
    if args.strategy == "bohb":
        results = optimize_with_bohb(args.base_config)
    elif args.strategy == "optuna":
        results = optimize_with_optuna(args.base_config)
    elif args.strategy == "random":
        results = optimize_with_random_search(args.base_config)
    elif args.strategy == "pbt":
        results = optimize_with_pbt(args.base_config)
    elif args.strategy == "progressive":
        results = optimize_progressive(args.base_config)
    
    print("\n" + "="*60)
    print("Optimization Complete!")
    print("="*60)


if __name__ == "__main__":
    main()