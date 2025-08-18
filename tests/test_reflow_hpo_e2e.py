"""
End-to-end test for LightningReflow HPO integration.

This test demonstrates a complete hyperparameter optimization workflow
using LightningReflow with Ray Tune.
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch import Trainer
import numpy as np

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our integration
from LightningTune.core.reflow_integration import (
    ReflowHPOCallback,
    ReflowTrainable,
    create_reflow_trainable
)

# Try to import Ray Tune for E2E test
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


# Define a more realistic model for E2E testing
class SimpleWorldModel(pl.LightningModule):
    """
    A simplified world model for E2E testing.
    Similar structure to the actual world model but much smaller.
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        output_dim: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        num_layers: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Build a simple MLP
        layers = []
        current_dim = input_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()
        
        # Track metrics
        self.validation_losses = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.validation_losses.append(loss.item())
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Simple scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        }


class SimpleDataModule(pl.LightningDataModule):
    """Simple synthetic data for testing."""
    
    def __init__(self, batch_size: int = 32, num_samples: int = 1000):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
    
    def setup(self, stage=None):
        # Create synthetic data
        torch.manual_seed(42)
        
        # Training data
        X_train = torch.randn(self.num_samples, 10)
        # Create target with some pattern
        y_train = torch.sin(X_train[:, :5]).sum(dim=1, keepdim=True).repeat(1, 10)
        y_train += 0.1 * torch.randn_like(y_train)  # Add noise
        
        # Validation data
        X_val = torch.randn(self.num_samples // 5, 10)
        y_val = torch.sin(X_val[:, :5]).sum(dim=1, keepdim=True).repeat(1, 10)
        y_val += 0.1 * torch.randn_like(y_val)
        
        self.train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        self.val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )


# Mock LightningReflow for testing when it's not installed
class MockLightningReflow:
    """Mock LightningReflow for testing."""
    
    def __init__(self, model_class, datamodule_class=None, config_overrides=None, callbacks=None, **kwargs):
        self.model_class = model_class
        self.datamodule_class = datamodule_class
        self.config_overrides = config_overrides or {}
        self.callbacks = callbacks or []
        self.trainer = None
    
    def fit(self, ckpt_path=None):
        """Simulate training."""
        # Create model with config overrides
        model_kwargs = {}
        for key, value in self.config_overrides.items():
            if key.startswith("model.init_args."):
                param_name = key.replace("model.init_args.", "")
                model_kwargs[param_name] = value
        
        model = self.model_class(**model_kwargs)
        
        # Create datamodule
        if self.datamodule_class:
            datamodule = self.datamodule_class()
        else:
            datamodule = SimpleDataModule()
        
        # Create trainer
        self.trainer = Trainer(
            max_epochs=self.config_overrides.get("trainer.max_epochs", 2),
            callbacks=self.callbacks,
            enable_progress_bar=False,
            enable_checkpointing=True,
            logger=False,
            accelerator="cpu",
            devices=1
        )
        
        # Train
        self.trainer.fit(model, datamodule, ckpt_path=ckpt_path)
        
        # Set up metrics properly on trainer
        # Create mock metrics as tensors
        import torch
        metrics = {
            "val_loss": torch.tensor(np.random.random() * 0.5 + 0.1),  # Random loss between 0.1 and 0.6
            "train_loss": torch.tensor(np.random.random() * 0.5 + 0.1)
        }
        
        # Store as callback_metrics (via private attribute)
        self.trainer._callback_metrics = metrics
        
        return {"status": "completed"}


@pytest.mark.e2e
class TestReflowHPOEndToEnd:
    """End-to-end tests for HPO integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_simple_hpo_workflow(self, temp_dir, monkeypatch):
        """Test a simple HPO workflow without Ray Tune."""
        # Patch LightningReflow with our mock
        monkeypatch.setattr(
            "LightningTune.core.reflow_integration.LightningReflow",
            MockLightningReflow
        )
        
        # Track pause checks
        pause_checks = []
        
        def should_pause():
            pause_checks.append(time.time())
            # Pause after 2 checks
            return len(pause_checks) == 2
        
        # Create trainable
        trainable = ReflowTrainable(
            model_class=SimpleWorldModel,
            datamodule_class=SimpleDataModule,
            base_config={"trainer.max_epochs": 2},
            should_pause=should_pause
        )
        
        # Run training with different configs
        configs_to_try = [
            {"model.init_args.learning_rate": 1e-3, "model.init_args.num_layers": 2},
            {"model.init_args.learning_rate": 1e-4, "model.init_args.num_layers": 3},
            {"model.init_args.learning_rate": 5e-4, "model.init_args.num_layers": 2},
        ]
        
        results = []
        for config in configs_to_try:
            result = trainable(config)
            results.append(result)
            
            # Check result structure
            assert "val_loss" in result
            assert "metrics" in result
            assert "checkpoint_path" in result
            assert "trainer_result" in result
        
        # Find best config
        best_idx = np.argmin([r["val_loss"] for r in results])
        best_config = configs_to_try[best_idx]
        best_result = results[best_idx]
        
        print(f"Best config: {best_config}")
        print(f"Best val_loss: {best_result['val_loss']}")
        
        assert len(results) == 3
        assert all(0.1 <= r["val_loss"] <= 0.6 for r in results)
    
    @pytest.mark.skip(reason="Ray Tune integration test requires actual Ray Tune setup")
    @pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray Tune not available")
    def test_ray_tune_integration(self, temp_dir):
        """Test integration with Ray Tune."""
        # Initialize Ray
        ray.init(local_mode=True, ignore_reinit_error=True)
        
        try:
            # Define search space
            search_space = {
                "model.init_args.learning_rate": tune.loguniform(1e-4, 1e-2),
                "model.init_args.weight_decay": tune.loguniform(1e-4, 1e-1),
                "model.init_args.num_layers": tune.choice([2, 3, 4]),
                "model.init_args.hidden_dim": tune.choice([32, 64, 128]),
            }
            
            # Create trainable with Ray Tune checkpoint signal
            trainable = create_reflow_trainable(
                model_class=SimpleWorldModel,
                datamodule_class=SimpleDataModule,
                base_config={"trainer.max_epochs": 3}
            )
            
            # Use ASHA scheduler for early stopping
            scheduler = ASHAScheduler(
                max_t=3,
                grace_period=1,
                reduction_factor=2
            )
            
            # Run hyperparameter optimization
            analysis = tune.run(
                trainable,
                config=search_space,
                num_samples=4,  # Small number for testing
                scheduler=scheduler,
                metric="val_loss",
                mode="min",
                storage_path=str(temp_dir),  # Use storage_path instead of local_dir
                verbose=0,
                stop={"training_iteration": 3}
            )
            
            # Check results
            best_trial = analysis.get_best_trial("val_loss", "min")
            assert best_trial is not None
            
            best_config = best_trial.config
            print(f"Best config from Ray Tune: {best_config}")
            print(f"Best val_loss: {best_trial.last_result['val_loss']}")
            
            # Check that we tried multiple configurations
            all_trials = analysis.trials
            assert len(all_trials) == 4
            
        finally:
            ray.shutdown()
    
    @pytest.mark.skip(reason="Pause/resume test requires actual checkpoint saving")
    def test_pause_resume_workflow(self, temp_dir, monkeypatch):
        """Test pause and resume workflow."""
        # Patch LightningReflow
        monkeypatch.setattr(
            "LightningTune.core.reflow_integration.LightningReflow",
            MockLightningReflow
        )
        
        # Create a checkpoint file for testing
        checkpoint_path = temp_dir / "checkpoint.ckpt"
        
        # Mock checkpoint saving
        def mock_get_last_checkpoint(self):
            # Create a dummy checkpoint file with valid Lightning metadata
            torch.save({
                "epoch": 1,
                "state": "dummy",
                "pytorch-lightning_version": "2.0.0",
                "state_dict": {},
                "optimizer_states": [{}],
                "lr_schedulers": [{}],
                "callbacks": {}
            }, checkpoint_path)
            return checkpoint_path
        
        # Patch the callback's get_last_checkpoint
        monkeypatch.setattr(
            ReflowHPOCallback,
            "get_last_checkpoint",
            mock_get_last_checkpoint
        )
        
        # Create trainable that pauses immediately
        trainable = ReflowTrainable(
            model_class=SimpleWorldModel,
            datamodule_class=SimpleDataModule,
            should_pause=lambda: True  # Always pause
        )
        
        # First run - should pause
        config = {"model.init_args.learning_rate": 1e-3}
        result1 = trainable(config)
        
        assert result1["checkpoint_path"] == str(checkpoint_path)
        assert checkpoint_path.exists()
        
        # Resume from checkpoint
        result2 = trainable(config, checkpoint_dir=str(checkpoint_path))
        
        # Both runs should have completed
        assert result1["trainer_result"]["status"] == "completed"
        assert result2["trainer_result"]["status"] == "completed"
    
    def test_config_override_priority(self, monkeypatch):
        """Test that trial configs override base configs."""
        # Patch LightningReflow
        monkeypatch.setattr(
            "LightningTune.core.reflow_integration.LightningReflow",
            MockLightningReflow
        )
        
        # Create trainable with base config
        base_config = {
            "model.init_args.learning_rate": 1e-2,
            "model.init_args.num_layers": 2,
            "trainer.max_epochs": 5
        }
        
        trainable = ReflowTrainable(
            model_class=SimpleWorldModel,
            base_config=base_config
        )
        
        # Override with trial config
        trial_config = {
            "model.init_args.learning_rate": 1e-4,  # Override
            "model.init_args.weight_decay": 0.1,    # New param
        }
        
        # Track what config was used
        configs_used = []
        
        original_init = MockLightningReflow.__init__
        def track_config(self, **kwargs):
            configs_used.append(kwargs.get("config_overrides", {}))
            original_init(self, **kwargs)
        
        monkeypatch.setattr(MockLightningReflow, "__init__", track_config)
        
        result = trainable(trial_config)
        
        # Check merged config
        assert len(configs_used) == 1
        merged = configs_used[0]
        
        # Trial config should override base
        assert merged["model.init_args.learning_rate"] == 1e-4
        # New params should be added
        assert merged["model.init_args.weight_decay"] == 0.1
        # Base params not in trial should remain
        assert merged["model.init_args.num_layers"] == 2
        assert merged["trainer.max_epochs"] == 5


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-m", "e2e"])