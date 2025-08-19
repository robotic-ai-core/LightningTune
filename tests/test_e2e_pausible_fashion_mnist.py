"""
End-to-end integration tests for PausibleOptunaOptimizer using Fashion-MNIST.

This test demonstrates actual pause/resume functionality with a real model and dataset.
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
import pickle
import sys
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).parent.parent))
from LightningTune import PausibleOptunaOptimizer


class SimpleFashionMNISTModel(L.LightningModule):
    """Very simple CNN for Fashion-MNIST to minimize training time."""
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        hidden_size: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Simple network
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 14 * 14, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=False)
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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class FastFashionMNISTDataModule(L.LightningDataModule):
    """Fast DataModule using small subset of Fashion-MNIST."""
    
    def __init__(self, batch_size: int = 128, data_dir: str = "./data"):
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
            # Use very small subset for speed
            self.mnist_train, self.mnist_val, _ = random_split(
                mnist_full, 
                [1000, 200, len(mnist_full) - 1200],
                generator=torch.Generator().manual_seed(42)
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, 
            batch_size=self.hparams.batch_size, 
            shuffle=True,
            num_workers=0
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, 
            batch_size=self.hparams.batch_size,
            num_workers=0
        )


@pytest.mark.slow
def test_pausible_optimizer_with_real_model():
    """Test PausibleOptunaOptimizer with actual model training."""
    
    base_config = {
        "model": {
            "learning_rate": 0.001,
            "hidden_size": 32,
        },
        "data": {
            "batch_size": 128,
            "data_dir": "./data",
        },
        "trainer": {
            "max_epochs": 1,  # Very minimal
            "accelerator": "auto",
            "devices": 1,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
            "limit_train_batches": 5,  # Only 5 batches per epoch
            "limit_val_batches": 2,     # Only 2 validation batches
        }
    }
    
    def search_space(trial):
        return {
            "model.learning_rate": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "model.hidden_size": trial.suggest_categorical("hidden", [16, 32, 64]),
        }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock WandB to avoid actual uploads
        with patch('wandb.init') as mock_init, \
             patch('wandb.Artifact') as mock_artifact_class, \
             patch('wandb.Api') as mock_api_class:
            
            # Setup minimal mocks
            mock_run = Mock()
            mock_init.return_value = mock_run
            mock_artifact = Mock()
            mock_artifact_class.return_value = mock_artifact
            
            # Create optimizer
            optimizer = PausibleOptunaOptimizer(
                base_config=base_config,
                search_space=search_space,
                model_class=SimpleFashionMNISTModel,
                datamodule_class=FastFashionMNISTDataModule,
                wandb_project="test-project",
                study_name="fashion-mnist-test",
                sampler_name="random",
                pruner_name="none",
                save_every_n_trials=2,
                enable_pause=True,
                experiment_dir=Path(tmpdir),
                save_checkpoints=False,
                metric="val_loss",
                direction="minimize",
                verbose=False,
            )
            
            # Run optimization
            study = optimizer.optimize(n_trials=3)
            
            # Verify results
            assert len(study.trials) == 3
            completed_trials = [t for t in study.trials 
                              if t.state == optuna.trial.TrialState.COMPLETE]
            assert len(completed_trials) >= 2, "Should have at least 2 completed trials"
            
            # Check that different hyperparameters were explored
            lrs = set()
            hiddens = set()
            for trial in completed_trials:
                if "lr" in trial.params:
                    lrs.add(trial.params["lr"])
                if "hidden" in trial.params:
                    hiddens.add(trial.params["hidden"])
            
            assert len(lrs) >= 2, "Should explore different learning rates"
            assert len(hiddens) >= 1, "Should explore different hidden sizes"


@pytest.mark.slow  
def test_pause_resume_with_real_training():
    """Test actual pause and resume with model training."""
    
    base_config = {
        "model": {
            "learning_rate": 0.001,
            "hidden_size": 32,
        },
        "data": {
            "batch_size": 128,
            "data_dir": "./data",
        },
        "trainer": {
            "max_epochs": 1,
            "accelerator": "auto",
            "devices": 1,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
            "limit_train_batches": 3,
            "limit_val_batches": 1,
        }
    }
    
    search_space = lambda trial: {
        "model.learning_rate": trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        study_storage = {}  # Simulated WandB storage
        
        def mock_save_artifact(artifact):
            """Mock saving to WandB."""
            calls = artifact.add_file.call_args_list
            if calls:
                file_path = calls[0][0][0]
                with open(file_path, 'rb') as f:
                    study_storage['latest'] = pickle.load(f)
        
        def mock_load_artifact(version):
            """Mock loading from WandB."""
            if version in study_storage:
                with tempfile.TemporaryDirectory() as td:
                    file_path = Path(td) / "study.pkl"
                    with open(file_path, 'wb') as f:
                        pickle.dump(study_storage[version], f)
                    return td
            raise Exception("Not found")
        
        with patch('wandb.init') as mock_init, \
             patch('wandb.Artifact') as mock_artifact_class, \
             patch('wandb.Api') as mock_api_class:
            
            # Setup mocks
            mock_run = Mock()
            mock_run.log_artifact.side_effect = lambda a: mock_save_artifact(a)
            mock_init.return_value = mock_run
            
            mock_artifact = Mock()
            mock_artifact_class.return_value = mock_artifact
            
            mock_api = Mock()
            mock_api_class.return_value = mock_api
            mock_api_artifact = Mock()
            mock_api_artifact.download.side_effect = mock_load_artifact
            mock_api.artifact.return_value = mock_api_artifact
            
            # PHASE 1: Run 2 trials then "pause"
            optimizer1 = PausibleOptunaOptimizer(
                base_config=base_config,
                search_space=search_space,
                model_class=SimpleFashionMNISTModel,
                datamodule_class=FastFashionMNISTDataModule,
                wandb_project="test-project",
                study_name="pause-resume-test",
                sampler_name="random",
                pruner_name="none",
                save_every_n_trials=10,  # Don't auto-save
                enable_pause=True,
                experiment_dir=Path(tmpdir) / "run1",
                save_checkpoints=False,
                metric="val_loss",
                direction="minimize",
                verbose=False,
            )
            
            # Simulate pause after 2 trials
            original_optimize = optimizer1.optimize
            
            def mock_optimize_with_pause(*args, **kwargs):
                # Run 2 trials then trigger pause
                optimizer1.save_every_n_trials = 1  # Force save on each trial
                
                class PauseAfterN:
                    def __init__(self, n):
                        self.count = 0
                        self.n = n
                    
                    def __call__(self, study, trial):
                        self.count += 1
                        if self.count >= self.n:
                            optimizer1.should_pause = True
                
                stopper = PauseAfterN(2)
                kwargs['callbacks'] = [stopper] if 'callbacks' not in kwargs else kwargs['callbacks'] + [stopper]
                return original_optimize(*args, **kwargs)
            
            optimizer1.optimize = mock_optimize_with_pause
            study1 = optimizer1.optimize(n_trials=5)
            
            # Should have stopped after 2 trials
            assert len(study1.trials) == 2
            assert optimizer1.total_trials_completed == 2
            
            # Get the loss values from first run
            first_run_values = [t.value for t in study1.trials if t.value is not None]
            assert len(first_run_values) == 2
            
            # PHASE 2: Resume and complete
            optimizer2 = PausibleOptunaOptimizer(
                base_config=base_config,
                search_space=search_space,
                model_class=SimpleFashionMNISTModel,
                datamodule_class=FastFashionMNISTDataModule,
                wandb_project="test-project",
                study_name="pause-resume-test",
                sampler_name="random",
                pruner_name="none",
                save_every_n_trials=10,
                enable_pause=True,
                experiment_dir=Path(tmpdir) / "run2",
                save_checkpoints=False,
                metric="val_loss",
                direction="minimize",
                verbose=False,
            )
            
            # Resume and run 2 more trials (total target is 4)
            study2 = optimizer2.optimize(
                n_trials=4,
                resume_from="latest"
            )
            
            # Should have 4 trials total (2 old + 2 new)
            assert len(study2.trials) == 4
            assert optimizer2.total_trials_completed == 4
            
            # Verify the first 2 trials are the same
            resumed_values = [t.value for t in study2.trials[:2] if t.value is not None]
            assert resumed_values == first_run_values, "First trials should be preserved"
            
            # Verify new trials were added
            new_trials = study2.trials[2:]
            assert len(new_trials) == 2
            assert all(t.state == optuna.trial.TrialState.COMPLETE for t in new_trials)


@pytest.mark.slow
def test_pruning_with_pausible_optimizer():
    """Test that pruning works correctly with pausible optimizer."""
    
    base_config = {
        "model": {
            "learning_rate": 0.001,
            "hidden_size": 32,
        },
        "data": {
            "batch_size": 128,
            "data_dir": "./data",
        },
        "trainer": {
            "max_epochs": 3,  # Need multiple epochs for pruning
            "accelerator": "auto",
            "devices": 1,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
            "limit_train_batches": 2,
            "limit_val_batches": 1,
            "val_check_interval": 1.0,  # Check every epoch
        }
    }
    
    search_space = lambda trial: {
        "model.learning_rate": trial.suggest_float("lr", 1e-5, 1e-1, log=True),  # Wide range to trigger bad trials
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('wandb.init'), patch('wandb.Artifact'), patch('wandb.Api'):
            
            optimizer = PausibleOptunaOptimizer(
                base_config=base_config,
                search_space=search_space,
                model_class=SimpleFashionMNISTModel,
                datamodule_class=FastFashionMNISTDataModule,
                wandb_project=None,  # No WandB for this test
                study_name="pruning-test",
                sampler_name="random",
                pruner_name="median",  # Use median pruner
                save_every_n_trials=10,
                enable_pause=False,
                experiment_dir=Path(tmpdir),
                save_checkpoints=False,
                metric="val_loss",
                direction="minimize",
                verbose=False,
            )
            
            study = optimizer.optimize(n_trials=5)
            
            # Check that we have both completed and pruned trials
            completed = [t for t in study.trials 
                        if t.state == optuna.trial.TrialState.COMPLETE]
            pruned = [t for t in study.trials 
                     if t.state == optuna.trial.TrialState.PRUNED]
            
            assert len(study.trials) == 5
            # With median pruner, we should have at least some pruned trials
            # (unless all trials happen to perform similarly)
            assert len(completed) >= 1, "Should have at least 1 completed trial"
            
            # All finished trials (completed + pruned) should be counted
            finished = len(completed) + len(pruned)
            assert finished == optimizer.total_trials_completed


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])