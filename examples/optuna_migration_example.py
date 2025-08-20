"""
Example demonstrating migration from Ray Tune to Optuna.

This example shows how to migrate from the old Ray Tune-based
LightningTune to the new Optuna-based version.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from LightningTune import OptunaDrivenOptimizer, WandBOptunaOptimizer
from LightningTune.optuna.search_space import SimpleSearchSpace
from LightningTune.optuna.strategies import BOHBStrategy, TPEStrategy, RandomStrategy


# Example Lightning Module
class ExampleModel(pl.LightningModule):
    """Simple neural network for demonstration."""
    
    def __init__(self, learning_rate=0.001, hidden_size=64, dropout=0.1, optimizer_type="adam"):
        super().__init__()
        self.save_hyperparameters()
        
        self.net = nn.Sequential(
            nn.Linear(10, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        if self.hparams.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        else:
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


# Example DataModule
class ExampleDataModule(LightningDataModule):
    """Simple data module for demonstration."""
    
    def __init__(self, batch_size=32, num_samples=1000):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
    
    def setup(self, stage=None):
        # Generate some dummy regression data
        X = torch.randn(self.num_samples, 10)
        y = X.sum(dim=1) + 0.1 * torch.randn(self.num_samples)
        
        # Split into train/val
        train_size = int(0.8 * self.num_samples)
        self.train_dataset = TensorDataset(X[:train_size], y[:train_size])
        self.val_dataset = TensorDataset(X[train_size:], y[train_size:])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


def example_basic_optuna_optimization():
    """
    Basic Optuna optimization example.
    """
    print("=" * 60)
    print("BASIC OPTUNA OPTIMIZATION EXAMPLE")
    print("=" * 60)
    
    # Base configuration
    base_config = {
        "model": {
            "learning_rate": 0.001,
            "hidden_size": 64,
            "dropout": 0.1,
            "optimizer_type": "adam"
        },
        "data": {
            "batch_size": 32,
            "num_samples": 1000
        },
        "trainer": {
            "max_epochs": 5,
            "enable_progress_bar": False,
            "logger": False,
            "enable_checkpointing": False
        }
    }
    
    # Define search space
    search_space = SimpleSearchSpace({
        "model.learning_rate": ("loguniform", 1e-4, 1e-2),
        "model.hidden_size": ("int", 32, 128, 16),
        "model.dropout": ("uniform", 0.0, 0.5),
        "model.optimizer_type": ("categorical", ["adam", "sgd", "rmsprop"])
    })
    
    # Create optimizer
    optimizer = OptunaDrivenOptimizer(
        base_config=base_config,
        search_space=search_space,
        model_class=ExampleModel,
        datamodule_class=ExampleDataModule,
        strategy=TPEStrategy(),
        n_trials=10,
        direction="minimize",
        metric="val_loss",
        verbose=True
    )
    
    # Run optimization
    print("Starting optimization...")
    study = optimizer.run()
    
    # Print results
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best val_loss: {study.best_value:.6f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best configuration
    best_config_path = optimizer.save_best_config()
    print(f"Best configuration saved to: {best_config_path}")
    
    return study


def example_strategy_comparison():
    """
    Example comparing different optimization strategies.
    """
    print("=" * 60) 
    print("STRATEGY COMPARISON EXAMPLE")
    print("=" * 60)
    
    base_config = {
        "model": {"learning_rate": 0.001, "hidden_size": 64, "dropout": 0.1},
        "data": {"batch_size": 32, "num_samples": 500},  # Smaller for faster comparison
        "trainer": {"max_epochs": 3, "enable_progress_bar": False, "logger": False}
    }
    
    search_space = SimpleSearchSpace({
        "model.learning_rate": ("loguniform", 1e-4, 1e-2),
        "model.hidden_size": ("int", 32, 128, 32)
    })
    
    strategies = {
        "TPE": TPEStrategy(),
        "Random": RandomStrategy(seed=42),
        "BOHB": BOHBStrategy(min_resource=1, max_resource=3)
    }
    
    results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"\nTesting {strategy_name} strategy...")
        
        optimizer = OptunaDrivenOptimizer(
            base_config=base_config,
            search_space=search_space,
            model_class=ExampleModel,
            datamodule_class=ExampleDataModule,
            strategy=strategy,
            n_trials=8,
            direction="minimize",
            verbose=False
        )
        
        study = optimizer.run()
        results[strategy_name] = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials)
        }
        
        print(f"  Best val_loss: {study.best_value:.6f}")
    
    # Compare results
    print("\n" + "=" * 40)
    print("STRATEGY COMPARISON RESULTS")
    print("=" * 40)
    
    for strategy_name, result in results.items():
        print(f"{strategy_name:10s}: {result['best_value']:.6f} "
              f"(trials: {result['n_trials']})")
    
    # Find best strategy
    best_strategy = min(results.keys(), key=lambda k: results[k]['best_value'])
    print(f"\nBest strategy: {best_strategy}")
    
    return results


def example_wandb_integration():
    """
    Example with WandB integration (mocked for demo).
    """
    print("=" * 60)
    print("WANDB INTEGRATION EXAMPLE")
    print("=" * 60)
    
    # Mock objective function for WandB example
    def objective(trial, base_config=None, **kwargs):
        """Mock objective that doesn't require actual training."""
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        hidden_size = trial.suggest_int("hidden_size", 32, 128, step=16)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        
        # Simulate some loss based on hyperparameters
        # In practice, this would train your actual model
        loss = (
            abs(learning_rate - 0.001) * 100 +  # Prefer lr around 0.001
            abs(hidden_size - 64) * 0.01 +      # Prefer hidden_size around 64
            dropout * 2                         # Prefer lower dropout
        )
        
        return loss
    
    try:
        # This would require actual WandB setup
        optimizer = WandBOptunaOptimizer(
            objective=objective,
            project_name="lightningtune-demo",
            study_name="optuna-migration-example",
            n_trials=10,
            direction="minimize",
            save_every_n_trials=5,
            fast_dev_run=True,  # Enable fast dev mode
            log_to_wandb=False  # Disable actual WandB logging for demo
        )
        
        print("Starting WandB-integrated optimization...")
        study = optimizer.run()
        
        print(f"Completed {len(study.trials)} trials")
        print(f"Best value: {study.best_value:.6f}")
        print("Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
            
        return study
        
    except Exception as e:
        print(f"WandB integration demo failed (expected): {e}")
        print("This would work with proper WandB setup.")
        return None


def migration_comparison():
    """
    Show the difference between old Ray Tune and new Optuna APIs.
    """
    print("=" * 60)
    print("MIGRATION COMPARISON")
    print("=" * 60)
    
    print("\nOLD WAY (Ray Tune - DEPRECATED):")
    print("-" * 40)
    print("""
    from LightningTune.core.optimizer import ConfigDrivenOptimizer
    from LightningTune.core.config import SearchSpace
    
    # Define search space
    search_space = SearchSpace()
    search_space.add_float("learning_rate", 1e-4, 1e-2, log=True)
    search_space.add_int("hidden_size", 32, 128)
    
    # Create optimizer
    optimizer = ConfigDrivenOptimizer(
        base_config_path="config.yaml",
        search_space=search_space,
        strategy="bohb",
        max_epochs=10,
        num_samples=50
    )
    
    # Run optimization
    results = optimizer.run()
    """)
    
    print("\nNEW WAY (Optuna - RECOMMENDED):")
    print("-" * 40)
    print("""
    from LightningTune import OptunaDrivenOptimizer
    from LightningTune.optuna.search_space import OptunaSearchSpace
    from LightningTune.optuna.strategies import BOHBStrategy
    
    # Define search space
    search_space = OptunaSearchSpace()
    search_space.add_float("model.learning_rate", 1e-4, 1e-2, log=True)
    search_space.add_int("model.hidden_size", 32, 128)
    
    # Create optimizer
    optimizer = OptunaDrivenOptimizer(
        base_config="config.yaml",
        search_space=search_space,
        model_class=MyLightningModule,
        datamodule_class=MyDataModule,
        strategy=# TPESampler with HyperbandPruner,
        n_trials=50,
        direction="minimize"
    )
    
    # Run optimization
    study = optimizer.run()
    """)
    
    print("\nBACKWARD COMPATIBLE WAY:")
    print("-" * 40)
    print("""
    from LightningTune import ConfigDrivenOptimizer  # Now Optuna-based!
    
    # ConfigDrivenOptimizer is now an alias to OptunaDrivenOptimizer
    # Your existing code will need minor adjustments for the new API
    optimizer = ConfigDrivenOptimizer(...)
    """)


def main():
    """Run all examples."""
    print("LightningTune Optuna Migration Examples")
    print("=" * 60)
    
    try:
        # Show migration comparison
        migration_comparison()
        
        # Run basic optimization
        study1 = example_basic_optuna_optimization()
        
        # Compare strategies
        results = example_strategy_comparison()
        
        # Show WandB integration
        study2 = example_wandb_integration()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("- Optuna provides better optimization algorithms")
        print("- More reliable pruning and early stopping")
        print("- Native WandB integration with pause/resume")
        print("- Cleaner API with better error handling")
        print("- Extensive visualization capabilities")
        
    except Exception as e:
        print(f"\nExample failed: {e}")
        print("This might be due to missing dependencies or environment issues.")


if __name__ == "__main__":
    main()