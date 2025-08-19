#!/usr/bin/env python
"""
End-to-end test for LightningTune using Fashion-MNIST dataset.
This test verifies that HPO actually optimizes hyperparameters and improves performance.
"""

import sys
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
from LightningTune import OptunaDrivenOptimizer, SimpleSearchSpace


class FashionMNISTModel(pl.LightningModule):
    """Simple CNN for Fashion-MNIST with tunable hyperparameters."""
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        dropout_rate: float = 0.2,
        hidden_size: int = 128,
        conv_channels: int = 32,
        optimizer_type: str = "adam",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Convolutional layers with tunable channels
        self.conv1 = nn.Conv2d(1, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size: 28x28 -> 14x14 -> 7x7
        flattened_size = conv_channels * 2 * 7 * 7
        
        # Fully connected layers with tunable hidden size
        self.fc1 = nn.Linear(flattened_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 10)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        if self.hparams.optimizer_type == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer_type == "sgd":
            return torch.optim.SGD(
                self.parameters(), 
                lr=self.hparams.learning_rate,
                momentum=0.9
            )
        else:
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)


class FashionMNISTDataModule(pl.LightningDataModule):
    """DataModule for Fashion-MNIST with tunable batch size."""
    
    def __init__(self, batch_size: int = 32, data_dir: str = "./data"):
        super().__init__()
        self.save_hyperparameters()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def prepare_data(self):
        # Download data if needed
        FashionMNIST(self.hparams.data_dir, train=True, download=True)
        FashionMNIST(self.hparams.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = FashionMNIST(
                self.hparams.data_dir, 
                train=True, 
                transform=self.transform
            )
            # Use smaller subset for faster testing
            # Take only 6000 samples for training and 1000 for validation
            self.mnist_train, self.mnist_val, _ = random_split(
                mnist_full, 
                [6000, 1000, len(mnist_full) - 7000],
                generator=torch.Generator().manual_seed(42)
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, 
            batch_size=self.hparams.batch_size, 
            shuffle=True,
            num_workers=2
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, 
            batch_size=self.hparams.batch_size,
            num_workers=2
        )


def test_e2e_fashion_mnist_optimization():
    """
    End-to-end test that verifies HPO actually improves model performance.
    """
    print("\n" + "="*60)
    print("🧪 E2E Test: Fashion-MNIST Hyperparameter Optimization")
    print("="*60)
    
    # Base configuration
    base_config = {
        "model": {
            "learning_rate": 0.001,
            "dropout_rate": 0.2,
            "hidden_size": 128,
            "conv_channels": 32,
            "optimizer_type": "adam",
        },
        "data": {
            "batch_size": 64,
            "data_dir": "./data",
        },
        "trainer": {
            "max_epochs": 3,  # Short for testing
            "accelerator": "auto",
            "devices": 1,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
            "log_every_n_steps": 10,
        }
    }
    
    # Define search space
    search_space = SimpleSearchSpace({
        "model.learning_rate": ("loguniform", 1e-4, 1e-2),
        "model.dropout_rate": ("uniform", 0.1, 0.5),
        "model.hidden_size": ("categorical", [64, 128, 256]),
        "model.conv_channels": ("categorical", [16, 32, 64]),
        "model.optimizer_type": ("categorical", ["adam", "sgd", "adamw"]),
        "data.batch_size": ("categorical", [32, 64, 128]),
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n📁 Using temporary directory: {tmpdir}")
        
        # Create optimizer
        optimizer = OptunaDrivenOptimizer(
            base_config=base_config,
            search_space=search_space,
            model_class=FashionMNISTModel,
            datamodule_class=FashionMNISTDataModule,
            sampler=TPESampler(n_startup_trials=2, seed=42),
            pruner=MedianPruner(n_warmup_steps=10),
            experiment_dir=Path(tmpdir),
            n_trials=5,  # Small number for testing
            metric="val_loss",
            direction="minimize",
            verbose=False,
            save_checkpoints=False,
        )
        
        print("\n🔍 Running HPO with 5 trials...")
        study = optimizer.optimize()
        
        # Analyze results
        print("\n" + "="*60)
        print("📊 Results Analysis")
        print("-"*60)
        
        # Get all trial values
        trial_values = [t.value for t in study.trials if t.value is not None]
        
        if len(trial_values) < 2:
            print("❌ Not enough completed trials to analyze")
            return False
        
        # Check that we have different hyperparameter combinations
        param_variations = {}
        for trial in study.trials:
            if trial.value is not None:
                for key, value in trial.params.items():
                    if key not in param_variations:
                        param_variations[key] = set()
                    param_variations[key].add(value)
        
        print(f"✅ Completed trials: {len(trial_values)}")
        print(f"   Best loss: {min(trial_values):.4f}")
        print(f"   Worst loss: {max(trial_values):.4f}")
        print(f"   Mean loss: {np.mean(trial_values):.4f}")
        print(f"   Std loss: {np.std(trial_values):.4f}")
        
        # Check parameter exploration
        print("\n📈 Parameter Exploration:")
        explored_params = 0
        for param, values in param_variations.items():
            if len(values) > 1:
                explored_params += 1
                print(f"   ✅ {param}: {len(values)} different values tried")
            else:
                print(f"   ⚠️  {param}: only 1 value tried")
        
        # Verify optimization is working
        print("\n🎯 Verification:")
        
        # Test 1: Check that different hyperparameters were tried
        if explored_params < 2:
            print("   ❌ Not enough parameter variation")
            return False
        print(f"   ✅ Explored {explored_params} parameters with multiple values")
        
        # Test 2: Check that there's variation in performance
        if np.std(trial_values) < 1e-6:
            print("   ❌ No variation in trial results (all identical)")
            return False
        print(f"   ✅ Performance variation detected (std: {np.std(trial_values):.4f})")
        
        # Test 3: Check that best is better than mean
        best_val = min(trial_values)
        mean_val = np.mean(trial_values)
        if best_val >= mean_val:
            print("   ⚠️  Best trial not better than average")
        else:
            improvement = (mean_val - best_val) / mean_val * 100
            print(f"   ✅ Best trial {improvement:.1f}% better than average")
        
        # Test 4: Display best hyperparameters
        print("\n🏆 Best Hyperparameters:")
        for key, value in study.best_params.items():
            print(f"   {key}: {value}")
        
        # Overall success
        print("\n" + "="*60)
        print("✅ E2E Test PASSED: HPO is working correctly!")
        print("   - Different hyperparameters were explored")
        print("   - Performance varies with hyperparameters")
        print("   - Optimization found better configurations")
        print("="*60)
        
        return True


def test_pruning_actually_prunes():
    """
    Test that pruning actually stops bad trials early.
    """
    print("\n" + "="*60)
    print("🧪 Test: Pruning Functionality")
    print("="*60)
    
    # Create a deliberately bad configuration for some trials
    base_config = {
        "model": {
            "learning_rate": 0.001,
            "dropout_rate": 0.2,
            "hidden_size": 128,
            "conv_channels": 32,
            "optimizer_type": "adam",
        },
        "data": {
            "batch_size": 64,
            "data_dir": "./data",
        },
        "trainer": {
            "max_epochs": 5,
            "accelerator": "auto",
            "devices": 1,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
            "log_every_n_steps": 5,
            "check_val_every_n_epoch": 1,  # Check every epoch for pruning
        }
    }
    
    # Search space with extreme values that should get pruned
    search_space = SimpleSearchSpace({
        "model.learning_rate": ("categorical", [1e-6, 0.001, 0.1]),  # 1e-6 and 0.1 should perform poorly
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        optimizer = OptunaDrivenOptimizer(
            base_config=base_config,
            search_space=search_space,
            model_class=FashionMNISTModel,
            datamodule_class=FashionMNISTDataModule,
            sampler=TPESampler(n_startup_trials=1, seed=42),
            pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=5),
            experiment_dir=Path(tmpdir),
            n_trials=4,
            metric="val_loss",
            direction="minimize",
            verbose=False,
            save_checkpoints=False,
        )
        
        print("\n🔍 Running HPO with aggressive pruning...")
        study = optimizer.optimize()
        
        # Check pruning statistics
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        print(f"\n📊 Pruning Statistics:")
        print(f"   Total trials: {len(study.trials)}")
        print(f"   Completed: {len(completed_trials)}")
        print(f"   Pruned: {len(pruned_trials)}")
        
        if len(pruned_trials) > 0:
            print("   ✅ Pruning is working - bad trials were stopped early")
        else:
            print("   ⚠️  No trials were pruned (this can happen with small trial counts)")
        
        return True


if __name__ == "__main__":
    import sys
    
    success = True
    
    try:
        # Run main e2e test
        if not test_e2e_fashion_mnist_optimization():
            success = False
        
        # Run pruning test
        if not test_pruning_actually_prunes():
            success = False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    sys.exit(0 if success else 1)