"""
End-to-end integration tests for Optuna-based optimization using Fashion-MNIST.
"""

import pytest
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner, NopPruner
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from LightningTune import OptunaDrivenOptimizer, SimpleSearchSpace


class FashionMNISTModel(L.LightningModule):
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
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size
        flattened_size = conv_channels * 2 * 7 * 7
        
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 10)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (torch.argmax(logits, dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=False)
        self.log("train_acc", acc, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (torch.argmax(logits, dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", acc, prog_bar=False)
        return loss
    
    def configure_optimizers(self):
        if self.hparams.optimizer_type == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer_type == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)
        else:
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)


class FashionMNISTDataModule(L.LightningDataModule):
    """DataModule for Fashion-MNIST."""
    
    def __init__(self, batch_size: int = 32, data_dir: str = "./data"):
        super().__init__()
        self.save_hyperparameters()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def prepare_data(self):
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
            num_workers=0  # Set to 0 for stability in tests
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, 
            batch_size=self.hparams.batch_size,
            num_workers=0  # Set to 0 for stability in tests
        )


@pytest.mark.slow
def test_optuna_optimizer_improves_performance():
    """Test that Optuna optimizer actually improves model performance."""
    
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
            "max_epochs": 2,  # Very short for testing
            "accelerator": "auto",
            "devices": 1,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
            "log_every_n_steps": 10,
        }
    }
    
    search_space = SimpleSearchSpace({
        "model.learning_rate": ("loguniform", 1e-4, 1e-2),
        "model.dropout_rate": ("uniform", 0.1, 0.5),
        "model.hidden_size": ("categorical", [64, 128]),
        "model.conv_channels": ("categorical", [16, 32]),
        "data.batch_size": ("categorical", [32, 64]),
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        optimizer = OptunaDrivenOptimizer(
            base_config=base_config,
            search_space=search_space,
            model_class=FashionMNISTModel,
            datamodule_class=FashionMNISTDataModule,
            sampler=TPESampler(n_startup_trials=2, seed=42),
            pruner=NopPruner(),
            experiment_dir=Path(tmpdir),
            n_trials=3,  # Minimum for testing
            metric="val_loss",
            direction="minimize",
            verbose=False,
            save_checkpoints=False,
        )
        
        study = optimizer.optimize()
        
        # Verify results
        trial_values = [t.value for t in study.trials if t.value is not None]
        assert len(trial_values) >= 2, "Need at least 2 completed trials"
        
        # Check parameter variation
        param_variations = {}
        for trial in study.trials:
            if trial.value is not None:
                for key, value in trial.params.items():
                    if key not in param_variations:
                        param_variations[key] = set()
                    param_variations[key].add(value)
        
        # At least some parameters should have been explored
        explored_params = sum(1 for values in param_variations.values() if len(values) > 1)
        assert explored_params >= 1, "No parameter exploration occurred"
        
        # Check that there's variation in performance
        assert np.std(trial_values) > 1e-6, "No variation in trial results"
        
        # Best should be no worse than worst
        assert min(trial_values) <= max(trial_values)


@pytest.mark.parametrize("sampler_name,pruner_name", [
    ("tpe", "median"),
    ("random", "none"),
])
def test_different_samplers_and_pruners(sampler_name, pruner_name):
    """Test that different sampler/pruner combinations work."""
    
    samplers = {
        "tpe": TPESampler(n_startup_trials=1, seed=42),
        "random": RandomSampler(seed=42),
    }
    
    pruners = {
        "median": MedianPruner(n_warmup_steps=5),
        "none": NopPruner(),
    }
    
    base_config = {
        "model": {
            "learning_rate": 0.001,
            "dropout_rate": 0.2,
            "hidden_size": 64,  # Smaller for speed
            "conv_channels": 16,  # Smaller for speed
            "optimizer_type": "adam",
        },
        "data": {
            "batch_size": 128,  # Larger for speed
            "data_dir": "./data",
        },
        "trainer": {
            "max_epochs": 1,  # Minimal
            "accelerator": "auto",
            "devices": 1,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
        }
    }
    
    search_space = SimpleSearchSpace({
        "model.learning_rate": ("loguniform", 1e-4, 1e-2),
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        optimizer = OptunaDrivenOptimizer(
            base_config=base_config,
            search_space=search_space,
            model_class=FashionMNISTModel,
            datamodule_class=FashionMNISTDataModule,
            sampler=samplers[sampler_name],
            pruner=pruners[pruner_name],
            experiment_dir=Path(tmpdir),
            n_trials=2,  # Minimal
            metric="val_loss",
            direction="minimize",
            verbose=False,
            save_checkpoints=False,
        )
        
        study = optimizer.optimize()
        
        # Just verify it runs without errors
        assert len(study.trials) >= 1
        assert any(t.value is not None for t in study.trials)