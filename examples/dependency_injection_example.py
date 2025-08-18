"""
Example: Clean dependency injection pattern for optimization strategies.

This example shows how strategies are configured independently and then
injected into the optimizer, providing better separation of concerns
and more flexible configuration.
"""

from pathlib import Path
from ray import tune

from lightning_bohb import (
    ConfigDrivenOptimizer,
    OptimizationConfig,
    SearchSpace,
)

# Import strategies directly for dependency injection
from lightning_bohb.core.strategies import (
    BOHBStrategy,
    OptunaStrategy,
    RandomSearchStrategy,
    PBTStrategy,
    GridSearchStrategy,
    BayesianOptimizationStrategy,
)


class WorldModelSearchSpace(SearchSpace):
    """Search space for World Model hyperparameters."""
    
    def get_search_space(self):
        return {
            # Architecture
            "model.init_args.num_layers": tune.choice([4, 6, 8]),
            "model.init_args.hidden_dim": tune.choice([512, 1024, 2048]),
            
            # Optimization
            "model.init_args.learning_rate": tune.loguniform(1e-5, 1e-2),
            "model.init_args.weight_decay": tune.loguniform(1e-6, 1e-2),
            
            # Training
            "data.init_args.batch_size": tune.choice([16, 32, 64]),
        }
    
    def get_metric_config(self):
        return {"metric": "val_loss", "mode": "min"}


def example_1_bohb_injection():
    """
    Example 1: BOHB with dependency injection.
    
    The strategy is configured independently and then injected.
    """
    print("=" * 60)
    print("Example 1: BOHB with Dependency Injection")
    print("=" * 60)
    
    # 1. Configure strategy independently
    strategy = BOHBStrategy(
        grace_period=10,        # Min epochs before pruning
        reduction_factor=3,     # Aggressive pruning
        max_t=100,             # Max epochs
        metric="val_loss",      # What to optimize
        mode="min",            # Minimize the metric
        seed=42,               # For reproducibility
    )
    
    # 2. Configure optimization settings
    optimization_config = OptimizationConfig(
        max_epochs=100,
        max_concurrent_trials=4,
        experiment_name="world_model_bohb",
        resources_per_trial={"cpu": 4, "gpu": 0.5},
    )
    
    # 3. Inject strategy into optimizer
    optimizer = ConfigDrivenOptimizer(
        base_config_source="config.yaml",
        search_space=WorldModelSearchSpace(),
        strategy=strategy,  # Direct injection, no string names!
        optimization_config=optimization_config,
    )
    
    print(f"Strategy: {strategy.describe()}")
    print(f"Search space: {list(optimizer.search_space_dict.keys())}")
    
    # Run optimization
    # results = optimizer.run()
    print("(Skipping actual run for example)")


def example_2_optuna_injection():
    """
    Example 2: Optuna with dependency injection.
    
    Shows how different strategies can be swapped easily.
    """
    print("\n" + "=" * 60)
    print("Example 2: Optuna with Dependency Injection")
    print("=" * 60)
    
    # Configure Optuna strategy
    strategy = OptunaStrategy(
        n_startup_trials=20,    # Random exploration first
        use_pruner=True,        # Enable pruning
        pruner_type="median",   # Prune below median
        num_samples=100,        # Total trials
        metric="val_loss",
        mode="min",
    )
    
    # Same optimizer interface, different strategy
    optimizer = ConfigDrivenOptimizer(
        base_config_source="config.yaml",
        search_space=WorldModelSearchSpace(),
        strategy=strategy,  # Just swap the strategy!
    )
    
    print(f"Strategy: {strategy.describe()}")
    print(f"Summary: {strategy.get_summary()}")


def example_3_strategy_comparison():
    """
    Example 3: Compare multiple strategies on the same problem.
    
    Shows the power of dependency injection for experimentation.
    """
    print("\n" + "=" * 60)
    print("Example 3: Strategy Comparison with DI")
    print("=" * 60)
    
    # Define multiple strategies
    strategies = [
        RandomSearchStrategy(
            num_samples=20,
            use_early_stopping=True,
        ),
        OptunaStrategy(
            n_startup_trials=10,
            num_samples=20,
        ),
        BOHBStrategy(
            grace_period=5,
            reduction_factor=2,
        ),
    ]
    
    # Same search space for all
    search_space = WorldModelSearchSpace()
    
    # Run each strategy
    for strategy in strategies:
        print(f"\nRunning {strategy.get_strategy_name()}...")
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source="config.yaml",
            search_space=search_space,
            strategy=strategy,  # Inject each strategy
            optimization_config=OptimizationConfig(
                max_epochs=20,
                experiment_name=f"compare_{strategy.get_strategy_name()}",
            ),
        )
        
        print(f"  Config: {strategy.describe()}")
        # results = optimizer.run(time_budget_hrs=0.5)
        # print(f"  Best: {results.get_best_result().metrics}")


def example_4_pbt_with_mutations():
    """
    Example 4: PBT with adaptive hyperparameters.
    
    Shows how complex strategy configurations are cleaner with DI.
    """
    print("\n" + "=" * 60)
    print("Example 4: PBT with Dependency Injection")
    print("=" * 60)
    
    # Configure PBT with mutations
    strategy = PBTStrategy(
        perturbation_interval=10,
        population_size=8,
        hyperparam_mutations={
            # These can change during training
            "model.init_args.learning_rate": tune.loguniform(1e-5, 1e-2),
            "model.init_args.weight_decay": tune.uniform(0.0, 0.1),
        },
        metric="val_loss",
        mode="min",
    )
    
    optimizer = ConfigDrivenOptimizer(
        base_config_source="config.yaml",
        search_space=WorldModelSearchSpace(),
        strategy=strategy,
    )
    
    print(f"Strategy: {strategy.describe()}")
    print(f"Mutations: {list(strategy.hyperparam_mutations.keys())}")


def example_5_custom_strategy():
    """
    Example 5: Custom strategy with dependency injection.
    
    Shows how to create and inject custom strategies.
    """
    print("\n" + "=" * 60)
    print("Example 5: Custom Strategy with DI")
    print("=" * 60)
    
    from lightning_bohb.core.strategies import OptimizationStrategy
    from typing import Optional
    from ray.tune.schedulers import ASHAScheduler
    
    class MyCustomStrategy(OptimizationStrategy):
        """Custom strategy combining random search with aggressive pruning."""
        
        def __init__(self, num_samples: int = 100, metric: str = "val_loss"):
            self.num_samples = num_samples
            self.metric = metric
            self.mode = "min"
        
        def get_search_algorithm(self) -> Optional[Any]:
            return None  # Random search
        
        def get_scheduler(self):
            # Very aggressive ASHA scheduler
            return ASHAScheduler(
                time_attr="training_iteration",
                metric=self.metric,
                mode=self.mode,
                max_t=100,
                grace_period=3,  # Very early pruning
                reduction_factor=4,  # Aggressive reduction
            )
        
        def get_num_samples(self) -> int:
            return self.num_samples
        
        def describe(self) -> str:
            return f"MyCustom(aggressive pruning, n={self.num_samples})"
    
    # Use custom strategy
    strategy = MyCustomStrategy(num_samples=50)
    
    optimizer = ConfigDrivenOptimizer(
        base_config_source="config.yaml",
        search_space=WorldModelSearchSpace(),
        strategy=strategy,  # Inject custom strategy!
    )
    
    print(f"Strategy: {strategy.describe()}")


def example_6_dynamic_strategy_selection():
    """
    Example 6: Dynamic strategy selection based on resources.
    
    Shows how DI enables runtime strategy selection.
    """
    print("\n" + "=" * 60)
    print("Example 6: Dynamic Strategy Selection")
    print("=" * 60)
    
    def select_strategy(time_budget_hrs: float, n_gpus: int):
        """Select best strategy based on available resources."""
        
        if time_budget_hrs < 1.0:
            # Quick exploration
            return RandomSearchStrategy(
                num_samples=20,
                use_early_stopping=True,
            )
        elif time_budget_hrs < 4.0:
            # Balanced approach
            return OptunaStrategy(
                n_startup_trials=10,
                num_samples=50,
                use_pruner=True,
            )
        elif n_gpus >= 4:
            # Many GPUs, use PBT
            return PBTStrategy(
                population_size=n_gpus * 2,
                perturbation_interval=10,
            )
        else:
            # Thorough optimization
            return BOHBStrategy(
                grace_period=10,
                reduction_factor=3,
            )
    
    # Select strategy based on resources
    time_budget = 2.5  # hours
    n_gpus = 2
    
    strategy = select_strategy(time_budget, n_gpus)
    print(f"Selected: {strategy.get_strategy_name()}")
    print(f"Reason: {time_budget}hrs, {n_gpus} GPUs")
    
    # Use selected strategy
    optimizer = ConfigDrivenOptimizer(
        base_config_source="config.yaml",
        search_space=WorldModelSearchSpace(),
        strategy=strategy,
    )
    
    print(f"Strategy config: {strategy.describe()}")


def main():
    """Run all examples."""
    examples = [
        example_1_bohb_injection,
        example_2_optuna_injection,
        example_3_strategy_comparison,
        example_4_pbt_with_mutations,
        example_5_custom_strategy,
        example_6_dynamic_strategy_selection,
    ]
    
    for example_fn in examples:
        example_fn()
    
    print("\n" + "=" * 60)
    print("Dependency Injection Benefits:")
    print("=" * 60)
    print("✓ Strategies are self-contained and testable")
    print("✓ No string-based selection or separate configs")
    print("✓ Easy to swap strategies for experimentation")
    print("✓ Custom strategies integrate seamlessly")
    print("✓ Better type safety and IDE support")
    print("✓ Dynamic strategy selection based on context")


if __name__ == "__main__":
    main()