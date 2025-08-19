#!/usr/bin/env python
"""
Test cases for the simplified Optuna optimizer with direct dependency injection.
"""

import pytest
import tempfile
from pathlib import Path
import json
import yaml

import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner, SuccessiveHalvingPruner, NopPruner

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from LightningTune.optuna import OptunaDrivenOptimizer, SimpleSearchSpace


class SimpleModel(LightningModule):
    """Simple model for testing."""
    
    def __init__(self, learning_rate=0.001, hidden_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.layer = nn.Linear(10, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x):
        x = torch.relu(self.layer(x))
        return self.output(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def train_dataloader(self):
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=32)
    
    def val_dataloader(self):
        X = torch.randn(50, 10)
        y = torch.randn(50, 1)
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=32)


class TestOptunaDrivenOptimizer:
    """Test the simplified Optuna optimizer."""
    
    def test_default_sampler_pruner(self):
        """Test that defaults to TPESampler and MedianPruner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "model": {"learning_rate": 0.001},
                "trainer": {"max_epochs": 1}
            }
            
            search_space = SimpleSearchSpace({
                "model.learning_rate": ("loguniform", 0.0001, 0.1),
                "model.hidden_size": ("categorical", [16, 32, 64]),
            })
            
            optimizer = OptunaDrivenOptimizer(
                base_config=config,
                search_space=search_space,
                model_class=SimpleModel,
                experiment_dir=Path(tmpdir),
                n_trials=2,
                verbose=False
            )
            
            # Check defaults
            assert isinstance(optimizer.sampler, TPESampler)
            assert isinstance(optimizer.pruner, MedianPruner)
            
            # Run optimization
            study = optimizer.optimize()
            assert len(study.trials) == 2
    
    def test_custom_sampler_pruner(self):
        """Test with custom sampler and pruner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "model": {"learning_rate": 0.001},
                "trainer": {"max_epochs": 1}
            }
            
            search_space = SimpleSearchSpace({
                "model.learning_rate": ("loguniform", 0.0001, 0.1),
            })
            
            # Use custom sampler and pruner
            sampler = RandomSampler(seed=42)
            pruner = HyperbandPruner(min_resource=1, max_resource=3)
            
            optimizer = OptunaDrivenOptimizer(
                base_config=config,
                search_space=search_space,
                model_class=SimpleModel,
                sampler=sampler,
                pruner=pruner,
                experiment_dir=Path(tmpdir),
                n_trials=3,
                verbose=False
            )
            
            # Check custom components
            assert optimizer.sampler is sampler
            assert optimizer.pruner is pruner
            
            # Run optimization
            study = optimizer.optimize()
            assert len(study.trials) == 3
    
    def test_no_pruning(self):
        """Test with NopPruner (no pruning)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "model": {"learning_rate": 0.001},
                "trainer": {"max_epochs": 1}
            }
            
            search_space = SimpleSearchSpace({
                "model.hidden_size": ("categorical", [16, 32, 64]),
            })
            
            optimizer = OptunaDrivenOptimizer(
                base_config=config,
                search_space=search_space,
                model_class=SimpleModel,
                sampler=TPESampler(),
                pruner=NopPruner(),  # No pruning
                experiment_dir=Path(tmpdir),
                n_trials=2,
                verbose=False
            )
            
            # Check no pruning
            assert isinstance(optimizer.pruner, NopPruner)
            
            # Run optimization
            study = optimizer.optimize()
            # With no pruning, all trials should complete
            assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    
    def test_different_samplers(self):
        """Test with different Optuna samplers."""
        samplers_to_test = [
            TPESampler(n_startup_trials=1),
            RandomSampler(seed=42),
            # CmaEsSampler() - Skip as it requires all continuous parameters
        ]
        
        for sampler in samplers_to_test:
            with tempfile.TemporaryDirectory() as tmpdir:
                config = {
                    "model": {"learning_rate": 0.001},
                    "trainer": {"max_epochs": 1}
                }
                
                search_space = SimpleSearchSpace({
                    "model.learning_rate": ("loguniform", 0.0001, 0.1),
                })
                
                optimizer = OptunaDrivenOptimizer(
                    base_config=config,
                    search_space=search_space,
                    model_class=SimpleModel,
                    sampler=sampler,
                    pruner=NopPruner(),
                    experiment_dir=Path(tmpdir),
                    n_trials=1,
                    verbose=False
                )
                
                study = optimizer.optimize()
                assert len(study.trials) == 1
    
    def test_different_pruners(self):
        """Test with different Optuna pruners."""
        pruners_to_test = [
            MedianPruner(n_warmup_steps=0),
            HyperbandPruner(min_resource=1, max_resource=2),
            SuccessiveHalvingPruner(min_resource=1, reduction_factor=2),
            NopPruner(),
        ]
        
        for pruner in pruners_to_test:
            with tempfile.TemporaryDirectory() as tmpdir:
                config = {
                    "model": {"learning_rate": 0.001},
                    "trainer": {"max_epochs": 2}
                }
                
                search_space = SimpleSearchSpace({
                    "model.learning_rate": ("loguniform", 0.0001, 0.1),
                })
                
                optimizer = OptunaDrivenOptimizer(
                    base_config=config,
                    search_space=search_space,
                    model_class=SimpleModel,
                    sampler=RandomSampler(seed=42),
                    pruner=pruner,
                    experiment_dir=Path(tmpdir),
                    n_trials=1,
                    verbose=False
                )
                
                study = optimizer.optimize()
                assert len(study.trials) >= 1
    
    def test_config_loading(self):
        """Test loading config from YAML and JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Test YAML config
            yaml_config = {
                "model": {"learning_rate": 0.001},
                "trainer": {"max_epochs": 1}
            }
            yaml_path = tmpdir / "config.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_config, f)
            
            search_space = SimpleSearchSpace({
                "model.learning_rate": ("loguniform", 0.0001, 0.1),
            })
            
            optimizer = OptunaDrivenOptimizer(
                base_config=yaml_path,
                search_space=search_space,
                model_class=SimpleModel,
                experiment_dir=tmpdir,
                n_trials=1,
                verbose=False
            )
            
            study = optimizer.optimize()
            assert len(study.trials) == 1
            
            # Test JSON config
            json_config = {
                "model": {"learning_rate": 0.001},
                "trainer": {"max_epochs": 1}
            }
            json_path = tmpdir / "config.json"
            with open(json_path, 'w') as f:
                json.dump(json_config, f)
            
            optimizer = OptunaDrivenOptimizer(
                base_config=json_path,
                search_space=search_space,
                model_class=SimpleModel,
                experiment_dir=tmpdir,
                n_trials=1,
                verbose=False
            )
            
            study = optimizer.optimize()
            assert len(study.trials) == 1
    
    def test_best_config(self):
        """Test getting the best configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "model": {"learning_rate": 0.001, "hidden_size": 32},
                "trainer": {"max_epochs": 1}
            }
            
            search_space = SimpleSearchSpace({
                "model.learning_rate": ("loguniform", 0.0001, 0.1),
                "model.hidden_size": ("categorical", [16, 32, 64]),
            })
            
            optimizer = OptunaDrivenOptimizer(
                base_config=config,
                search_space=search_space,
                model_class=SimpleModel,
                sampler=RandomSampler(seed=42),
                pruner=NopPruner(),
                experiment_dir=Path(tmpdir),
                n_trials=3,
                verbose=False
            )
            
            study = optimizer.optimize()
            best_config = optimizer.get_best_config()
            
            # Check that best config has the optimized parameters
            assert "model" in best_config
            assert "learning_rate" in best_config["model"]
            assert "hidden_size" in best_config["model"]
            
            # Check that best params match study's best params
            for key, value in study.best_params.items():
                if "learning_rate" in key:
                    assert best_config["model"]["learning_rate"] == value
                elif "hidden_size" in key:
                    assert best_config["model"]["hidden_size"] == value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])