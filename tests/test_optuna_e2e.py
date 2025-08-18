"""
Comprehensive end-to-end tests for Optuna migration.

This module tests all major functionality of the Optuna-based LightningTune
including basic optimization, different strategies, WandB integration,
pause/resume functionality, and backward compatibility.
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import yaml
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Import the new Optuna classes
from LightningTune.optuna.optimizer import OptunaDrivenOptimizer
from LightningTune.optuna.search_space import OptunaSearchSpace, SimpleSearchSpace
from LightningTune.optuna.strategies import (
    TPEStrategy, BOHBStrategy, RandomStrategy, GridStrategy, ASHAStrategy
)
from LightningTune.optuna.wandb_integration import WandBOptunaOptimizer

# Test if we can import from main package (backward compatibility)
try:
    from LightningTune import ConfigDrivenOptimizer, OptunaDrivenOptimizer as MainOptunaDrivenOptimizer
    MAIN_IMPORTS_WORK = True
except ImportError:
    MAIN_IMPORTS_WORK = False


class DummyDataset(Dataset):
    """Simple dataset for testing."""
    
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 10)
        self.targets = torch.randint(0, 2, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class DummyDataModule(LightningDataModule):
    """Simple data module for testing."""
    
    def __init__(self, batch_size=16, num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        self.train_dataset = DummyDataset(100)
        self.val_dataset = DummyDataset(50)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )


class DummyLightningModule(pl.LightningModule):
    """Simple Lightning module for testing."""
    
    def __init__(self, learning_rate=0.001, hidden_size=64, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        
        self.net = nn.Sequential(
            nn.Linear(10, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        return {'val_loss': loss, 'val_accuracy': accuracy}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def base_config():
    """Basic configuration for testing."""
    return {
        "model": {
            "learning_rate": 0.001,
            "hidden_size": 64,
            "dropout": 0.1
        },
        "data": {
            "batch_size": 16,
            "num_workers": 0
        },
        "trainer": {
            "max_epochs": 2,
            "enable_progress_bar": False,
            "enable_checkpointing": False,
            "logger": False
        }
    }


@pytest.fixture
def search_space():
    """Basic search space for testing."""
    space = SimpleSearchSpace({
        "model.learning_rate": ("loguniform", 1e-4, 1e-2),
        "model.hidden_size": ("int", 32, 128),
        "model.dropout": ("uniform", 0.0, 0.5)
    })
    return space


class TestOptunaBasicFunctionality:
    """Test basic Optuna optimizer functionality."""
    
    def test_optimizer_initialization(self, base_config, search_space, temp_dir):
        """Test that the optimizer initializes correctly."""
        optimizer = OptunaDrivenOptimizer(
            base_config=base_config,
            search_space=search_space,
            model_class=DummyLightningModule,
            datamodule_class=DummyDataModule,
            n_trials=2,
            experiment_dir=temp_dir,
            verbose=False
        )
        
        assert optimizer.base_config == base_config
        assert optimizer.search_space == search_space
        assert optimizer.model_class == DummyLightningModule
        assert optimizer.datamodule_class == DummyDataModule
        assert optimizer.n_trials == 2
        assert not optimizer.verbose
    
    def test_config_merging(self, base_config, search_space, temp_dir):
        """Test that configuration merging works correctly."""
        optimizer = OptunaDrivenOptimizer(
            base_config=base_config,
            search_space=search_space,
            model_class=DummyLightningModule,
            datamodule_class=DummyDataModule,
            n_trials=1,
            experiment_dir=temp_dir,
            verbose=False
        )
        
        # Test nested key merging
        updates = {
            "model.learning_rate": 0.01,
            "model.hidden_size": 128,
            "new_param": "test"
        }
        
        merged = optimizer._merge_configs(base_config, updates)
        
        assert merged["model"]["learning_rate"] == 0.01
        assert merged["model"]["hidden_size"] == 128
        assert merged["model"]["dropout"] == 0.1  # Original value preserved
        assert merged["new_param"] == "test"
    
    def test_basic_optimization(self, base_config, search_space, temp_dir):
        """Test basic optimization run."""
        optimizer = OptunaDrivenOptimizer(
            base_config=base_config,
            search_space=search_space,
            model_class=DummyLightningModule,
            datamodule_class=DummyDataModule,
            n_trials=2,
            experiment_dir=temp_dir,
            verbose=False,
            timeout=30  # Short timeout for testing
        )
        
        study = optimizer.run()
        
        assert study is not None
        assert len(study.trials) == 2
        assert optimizer.best_trial is not None
        assert hasattr(optimizer.best_trial, 'value')
        assert hasattr(optimizer.best_trial, 'params')
    
    def test_config_file_loading(self, base_config, search_space, temp_dir):
        """Test loading configuration from file."""
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(base_config, f)
        
        optimizer = OptunaDrivenOptimizer(
            base_config=str(config_path),
            search_space=search_space,
            model_class=DummyLightningModule,
            datamodule_class=DummyDataModule,
            n_trials=1,
            experiment_dir=temp_dir,
            verbose=False
        )
        
        assert optimizer.base_config == base_config


class TestOptunaStrategies:
    """Test different Optuna strategies."""
    
    @pytest.mark.parametrize("strategy_class", [
        TPEStrategy,
        BOHBStrategy,
        RandomStrategy,
        ASHAStrategy
    ])
    def test_strategy_initialization(self, strategy_class):
        """Test that all strategies initialize correctly."""
        if strategy_class == GridStrategy:
            # GridStrategy requires explicit search space
            strategy = strategy_class({"param1": [1, 2, 3]})
        else:
            strategy = strategy_class()
        
        assert strategy.name is not None
        assert hasattr(strategy, 'create_sampler')
        assert hasattr(strategy, 'create_pruner')
        assert hasattr(strategy, 'create_study')
    
    def test_tpe_strategy(self, base_config, search_space, temp_dir):
        """Test TPE strategy specifically."""
        strategy = TPEStrategy(n_startup_trials=1)
        
        optimizer = OptunaDrivenOptimizer(
            base_config=base_config,
            search_space=search_space,
            model_class=DummyLightningModule,
            datamodule_class=DummyDataModule,
            strategy=strategy,
            n_trials=2,
            experiment_dir=temp_dir,
            verbose=False,
            timeout=30
        )
        
        study = optimizer.run()
        assert len(study.trials) == 2
    
    def test_random_strategy(self, base_config, search_space, temp_dir):
        """Test Random strategy specifically."""
        strategy = RandomStrategy(seed=42)
        
        optimizer = OptunaDrivenOptimizer(
            base_config=base_config,
            search_space=search_space,
            model_class=DummyLightningModule,
            datamodule_class=DummyDataModule,
            strategy=strategy,
            n_trials=2,
            experiment_dir=temp_dir,
            verbose=False,
            timeout=30
        )
        
        study = optimizer.run()
        assert len(study.trials) == 2


class TestWandBIntegration:
    """Test WandB integration with mocking."""
    
    @patch('wandb.init')
    @patch('wandb.log')
    @patch('wandb.finish')
    def test_wandb_optimizer_basic(self, mock_finish, mock_log, mock_init, 
                                   base_config, search_space, temp_dir):
        """Test WandB optimizer basic functionality."""
        # Mock WandB run
        mock_run = MagicMock()
        mock_init.return_value = mock_run
        
        def objective(trial, **kwargs):
            # Simulate a simple objective
            x = trial.suggest_float('x', -10, 10)
            return (x - 2) ** 2
        
        optimizer = WandBOptunaOptimizer(
            objective=objective,
            project_name="test-project",
            study_name="test-study",
            n_trials=3,
            fast_dev_run=True,
            log_to_wandb=True
        )
        
        study = optimizer.run()
        
        assert study is not None
        assert len(study.trials) == 4  # fast_dev_run sets to 4
        assert mock_init.called
        assert mock_finish.called
    
    @patch('LightningTune.optuna.wandb_integration.save_optuna_session')
    @patch('LightningTune.optuna.wandb_integration.load_optuna_session')
    @patch('wandb.init')
    def test_wandb_pause_resume(self, mock_init, mock_load, mock_save, temp_dir):
        """Test WandB pause/resume functionality."""
        # Mock existing session
        mock_study = MagicMock()
        mock_study.best_value = 1.5
        mock_study.trials = [MagicMock() for _ in range(2)]
        
        mock_load.return_value = {
            "study": mock_study,
            "total_trials_completed": 2,
            "best_params": {"x": 1.0},
            "best_value": 1.5,
        }
        
        mock_run = MagicMock()
        mock_init.return_value = mock_run
        
        def objective(trial, **kwargs):
            x = trial.suggest_float('x', -10, 10)
            return (x - 2) ** 2
        
        optimizer = WandBOptunaOptimizer(
            objective=objective,
            project_name="test-project",
            study_name="test-study",
            n_trials=4,  # Should resume from trial 2
            save_every_n_trials=2,
            fast_dev_run=False,
            log_to_wandb=False  # Disable logging for this test
        )
        
        # This would normally resume from existing study
        optimizer._create_or_load_study()
        
        assert mock_load.called
        assert optimizer.total_trials_completed == 2


class TestBackwardCompatibility:
    """Test backward compatibility with existing interfaces."""
    
    @pytest.mark.skipif(not MAIN_IMPORTS_WORK, reason="Main imports not working")
    def test_main_package_imports(self):
        """Test that main package imports work."""
        # Test that we can import the main classes
        assert ConfigDrivenOptimizer is not None
        assert MainOptunaDrivenOptimizer is not None
    
    @pytest.mark.skipif(not MAIN_IMPORTS_WORK, reason="Main imports not working")
    def test_config_driven_optimizer_alias(self, base_config, search_space, temp_dir):
        """Test that ConfigDrivenOptimizer works as an alias."""
        # ConfigDrivenOptimizer should now be the Optuna-based optimizer
        from LightningTune.optuna.optimizer import OptunaDrivenOptimizer
        from LightningTune import ConfigDrivenOptimizer
        
        # They should be the same class
        assert ConfigDrivenOptimizer is OptunaDrivenOptimizer


class TestSearchSpace:
    """Test OptunaSearchSpace functionality."""
    
    def test_search_space_creation(self):
        """Test creating and configuring search space."""
        space = SimpleSearchSpace({
            "learning_rate": ("loguniform", 1e-4, 1e-2),
            "hidden_size": ("int", 32, 128, 16),
            "optimizer": ("categorical", ["adam", "sgd", "rmsprop"])
        })
        
        # Test the space has the parameters
        assert len(space.param_names) == 3
        assert "learning_rate" in space.param_names
        assert "hidden_size" in space.param_names
        assert "optimizer" in space.param_names
    
    def test_search_space_with_trial(self, search_space):
        """Test search space parameter suggestion."""
        # Mock trial
        mock_trial = MagicMock()
        mock_trial.suggest_float.side_effect = [0.001, 0.2]
        mock_trial.suggest_int.return_value = 64
        
        params = search_space.suggest_params(mock_trial)
        
        assert "model.learning_rate" in params
        assert "model.hidden_size" in params
        assert "model.dropout" in params


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_config_file(self, search_space, temp_dir):
        """Test handling of invalid config file."""
        with pytest.raises(FileNotFoundError):
            OptunaDrivenOptimizer(
                base_config="nonexistent.yaml",
                search_space=search_space,
                model_class=DummyLightningModule,
                n_trials=1,
                experiment_dir=temp_dir
            )
    
    def test_empty_search_space(self, base_config, temp_dir):
        """Test handling of empty search space."""
        empty_space = SimpleSearchSpace({})
        
        optimizer = OptunaDrivenOptimizer(
            base_config=base_config,
            search_space=empty_space,
            model_class=DummyLightningModule,
            datamodule_class=DummyDataModule,
            n_trials=1,
            experiment_dir=temp_dir,
            verbose=False
        )
        
        # Should still work, just won't optimize anything
        study = optimizer.run()
        assert study is not None
    
    def test_optimization_without_run(self, base_config, search_space, temp_dir):
        """Test getting results before running optimization."""
        optimizer = OptunaDrivenOptimizer(
            base_config=base_config,
            search_space=search_space,
            model_class=DummyLightningModule,
            n_trials=1,
            experiment_dir=temp_dir
        )
        
        with pytest.raises(ValueError):
            optimizer.get_best_config()


class TestUtilityFunctions:
    """Test utility functions and helper methods."""
    
    def test_config_saving(self, base_config, search_space, temp_dir):
        """Test saving and loading best configuration."""
        optimizer = OptunaDrivenOptimizer(
            base_config=base_config,
            search_space=search_space,
            model_class=DummyLightningModule,
            datamodule_class=DummyDataModule,
            n_trials=2,
            experiment_dir=temp_dir,
            verbose=False,
            timeout=30
        )
        
        study = optimizer.run()
        
        # Save best config
        config_path = optimizer.save_best_config()
        assert config_path.exists()
        
        # Load and verify
        with open(config_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        best_config = optimizer.get_best_config()
        assert saved_config == best_config
    
    def test_results_export(self, base_config, search_space, temp_dir):
        """Test exporting optimization results."""
        optimizer = OptunaDrivenOptimizer(
            base_config=base_config,
            search_space=search_space,
            model_class=DummyLightningModule,
            datamodule_class=DummyDataModule,
            n_trials=3,
            experiment_dir=temp_dir,
            verbose=False,
            timeout=30
        )
        
        study = optimizer.run()
        
        # Export results
        results_path = optimizer.export_results()
        assert results_path.exists()
        
        # Verify CSV format
        import pandas as pd
        df = pd.read_csv(results_path)
        assert len(df) == 3  # Should have 3 trials
        assert 'value' in df.columns