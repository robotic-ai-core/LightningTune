"""
Minimal example of using Lightning BOHB with a config-driven pipeline.

This shows the simplest possible usage where your entire training pipeline
is defined by a YAML config file (following Lightning CLI pattern).
"""

from lightning_bohb import ConfigDrivenBOHBOptimizer, BOHBConfig
from ray import tune


# Step 1: Define what hyperparameters to tune
search_space = {
    "model.init_args.learning_rate": tune.loguniform(1e-5, 1e-2),
    "model.init_args.hidden_dim": tune.choice([256, 512, 1024]),
    "model.init_args.dropout": tune.uniform(0.0, 0.5),
    "data.init_args.batch_size": tune.choice([16, 32, 64]),
}

# Step 2: Run optimization
optimizer = ConfigDrivenBOHBOptimizer(
    base_config_source="configs/train_config.yaml",  # Your existing config
    search_space=search_space,
    bohb_config=BOHBConfig(
        max_epochs=50,
        max_concurrent_trials=4,
        experiment_name="my_optimization"
    )
)

# Step 3: Run and get results
results = optimizer.run()

# Step 4: Get the best config for production
production_config = optimizer.create_production_config("configs/best_config.yaml")
print(f"Best config saved to: {production_config}")