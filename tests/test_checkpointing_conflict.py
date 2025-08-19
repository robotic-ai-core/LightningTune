"""
Test for checkpoint configuration conflicts.

This test ensures that the optimizer handles the conflict between
enable_checkpointing=False in config and save_checkpoints=True parameter.
"""

import pytest
import tempfile
from pathlib import Path
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from LightningTune import OptunaDrivenOptimizer, SimpleSearchSpace
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import lightning as L
from lightning.pytorch import LightningModule
import torch
import torch.nn as nn


class DummyModel(LightningModule):
    """Simple model for testing."""
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.layer = nn.Linear(10, 1)
        self.learning_rate = learning_rate
        
    def forward(self, x):
        return self.layer(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class DummyDataModule(L.LightningDataModule):
    """Simple data module for testing."""
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        # Create dummy data
        self.train_data = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randn(100, 1)
        )
        self.val_data = torch.utils.data.TensorDataset(
            torch.randn(20, 10),
            torch.randn(20, 1)
        )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size)


def test_checkpointing_conflict():
    """
    Test that optimizer handles conflict between enable_checkpointing=False 
    and save_checkpoints=True.
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config with enable_checkpointing=False
        config = {
            "model": {
                "init_args": {
                    "learning_rate": 1e-3
                }
            },
            "data": {
                "init_args": {
                    "batch_size": 32
                }
            },
            "trainer": {
                "max_epochs": 1,
                "enable_checkpointing": False,  # This conflicts with save_checkpoints=True
                "enable_progress_bar": False,
                "logger": False
            }
        }
        
        config_path = Path(tmpdir) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Create search space
        search_space = SimpleSearchSpace({
            "model.init_args.learning_rate": ("loguniform", 1e-4, 1e-2)
        })
        
        # This should handle the conflict gracefully
        optimizer = OptunaDrivenOptimizer(
            base_config=config_path,
            search_space=search_space,
            model_class=DummyModel,
            datamodule_class=DummyDataModule,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(),
            save_checkpoints=True,  # This conflicts with enable_checkpointing=False
            n_trials=2,
            experiment_dir=tmpdir,
            verbose=False
        )
        
        # Should not raise MisconfigurationException
        study = optimizer.optimize()
        
        # Verify optimization ran
        assert len(study.trials) == 2
        assert study.best_value is not None


def test_checkpointing_with_config_override():
    """
    Test that config_overrides can fix checkpointing conflicts.
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config with enable_checkpointing=False
        config = {
            "model": {
                "init_args": {
                    "learning_rate": 1e-3
                }
            },
            "data": {
                "init_args": {
                    "batch_size": 32
                }
            },
            "trainer": {
                "max_epochs": 1,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "logger": False
            }
        }
        
        config_path = Path(tmpdir) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Create search space
        search_space = SimpleSearchSpace({
            "model.init_args.learning_rate": ("loguniform", 1e-4, 1e-2)
        })
        
        # Use config_overrides to enable checkpointing
        config_overrides = {
            "trainer.enable_checkpointing": True
        }
        
        optimizer = OptunaDrivenOptimizer(
            base_config=config_path,
            search_space=search_space,
            config_overrides=config_overrides,
            model_class=DummyModel,
            datamodule_class=DummyDataModule,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(),
            save_checkpoints=True,
            n_trials=2,
            experiment_dir=tmpdir,
            verbose=False
        )
        
        study = optimizer.optimize()
        
        # Verify optimization ran
        assert len(study.trials) == 2
        assert study.best_value is not None


def test_no_checkpointing():
    """
    Test that optimizer works with both checkpointing disabled.
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "model": {
                "init_args": {
                    "learning_rate": 1e-3
                }
            },
            "data": {
                "init_args": {
                    "batch_size": 32
                }
            },
            "trainer": {
                "max_epochs": 1,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "logger": False
            }
        }
        
        config_path = Path(tmpdir) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        search_space = SimpleSearchSpace({
            "model.init_args.learning_rate": ("loguniform", 1e-4, 1e-2)
        })
        
        optimizer = OptunaDrivenOptimizer(
            base_config=config_path,
            search_space=search_space,
            model_class=DummyModel,
            datamodule_class=DummyDataModule,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(),
            save_checkpoints=False,  # Both are False, no conflict
            n_trials=2,
            experiment_dir=tmpdir,
            verbose=False
        )
        
        study = optimizer.optimize()
        
        # Verify optimization ran
        assert len(study.trials) == 2
        assert study.best_value is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])