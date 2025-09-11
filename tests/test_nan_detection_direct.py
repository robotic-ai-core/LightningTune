#!/usr/bin/env python
"""
Direct test for NaN detection that ensures we actually produce NaN values.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import lightning as L
from unittest.mock import Mock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external" / "LightningTune"))

from LightningTune.optuna.nan_detection_callback import NaNDetectionCallback
import optuna


class NaNProducingModel(L.LightningModule):
    """Model that deliberately produces NaN after a few steps."""
    
    def __init__(self, produce_nan_at_step=5):
        super().__init__()
        self.layer = nn.Linear(10, 1)
        self.produce_nan_at_step = produce_nan_at_step
        self.step_count = 0
        
    def training_step(self, batch, batch_idx):
        self.step_count += 1
        x, y = batch
        
        # Always do forward pass to maintain gradients
        y_hat = self.layer(x)
        loss = nn.functional.mse_loss(y_hat, y)
        
        if self.step_count >= self.produce_nan_at_step:
            # Deliberately produce NaN by multiplying by inf
            loss = loss * float('inf')
        
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def test_nan_detection_at_step_10():
    """Test that NaN is detected at step 10 with check_every_n_steps=10."""
    
    # Create mock trial
    trial = Mock(spec=optuna.Trial)
    trial.number = 0
    trial.set_user_attr = Mock()
    
    # Create model that produces NaN at step 5
    model = NaNProducingModel(produce_nan_at_step=5)
    
    # Create callback that checks every 10 steps
    callback = NaNDetectionCallback(
        trial=trial,
        monitor='train_loss',
        check_train_loss=True,
        check_every_n_steps=10,  # Check every 10 steps
        verbose=True
    )
    
    # Create simple dataset
    dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 10),
        torch.randn(100, 1)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    # Create trainer
    trainer = L.Trainer(
        max_steps=20,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )
    
    # Train - should detect NaN at step 10 (first check after NaN appears at step 5)
    try:
        trainer.fit(model, dataloader)
        assert False, "Should have raised TrialPruned"
    except optuna.TrialPruned as e:
        print(f"✅ NaN detected and pruned: {e}")
        # Check that it was detected at step 10 (first check after step 5)
        assert "step 10" in str(e).lower() or model.step_count <= 10, \
            f"NaN should be detected by step 10, but was at step {model.step_count}"
        # Verify trial was marked as failed
        trial.set_user_attr.assert_called_with('failed_reason', 'nan_or_inf_loss')


def test_nan_detection_immediate():
    """Test that NaN is detected immediately with check_every_n_steps=1."""
    
    # Create mock trial
    trial = Mock(spec=optuna.Trial)
    trial.number = 1
    trial.set_user_attr = Mock()
    
    # Create model that produces NaN at step 3
    model = NaNProducingModel(produce_nan_at_step=3)
    
    # Create callback that checks every step
    callback = NaNDetectionCallback(
        trial=trial,
        monitor='train_loss',
        check_train_loss=True,
        check_every_n_steps=1,  # Check EVERY step
        verbose=True
    )
    
    # Create simple dataset
    dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 10),
        torch.randn(100, 1)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    # Create trainer
    trainer = L.Trainer(
        max_steps=20,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )
    
    # Train - should detect NaN at step 3 immediately
    try:
        trainer.fit(model, dataloader)
        assert False, "Should have raised TrialPruned"
    except optuna.TrialPruned as e:
        print(f"✅ NaN detected immediately: {e}")
        # Should be detected at step 3 or very close to it
        assert model.step_count <= 4, \
            f"NaN should be detected immediately at step 3, but was at step {model.step_count}"
        trial.set_user_attr.assert_called_with('failed_reason', 'nan_or_inf_loss')


def test_inf_detection():
    """Test that Inf values are also detected."""
    
    class InfProducingModel(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 1)
            
        def training_step(self, batch, batch_idx):
            x, y = batch
            # Do forward pass
            y_hat = self.layer(x)
            loss = nn.functional.mse_loss(y_hat, y)
            # Make it Inf
            loss = loss * float('inf')
            self.log('train_loss', loss)
            return loss
        
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    # Create mock trial
    trial = Mock(spec=optuna.Trial)
    trial.number = 2
    trial.set_user_attr = Mock()
    
    model = InfProducingModel()
    
    callback = NaNDetectionCallback(
        trial=trial,
        monitor='train_loss',
        check_train_loss=True,
        check_every_n_steps=1,
        verbose=True
    )
    
    dataset = torch.utils.data.TensorDataset(
        torch.randn(10, 10),
        torch.randn(10, 1)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    trainer = L.Trainer(
        max_steps=5,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )
    
    try:
        trainer.fit(model, dataloader)
        assert False, "Should have raised TrialPruned for Inf"
    except optuna.TrialPruned as e:
        print(f"✅ Inf detected and pruned: {e}")
        assert "inf" in str(e).lower(), "Should mention Inf in error"
        trial.set_user_attr.assert_called_with('failed_reason', 'nan_or_inf_loss')


if __name__ == "__main__":
    print("Testing NaN detection with different check frequencies...\n")
    
    print("Test 1: NaN detection with check_every_n_steps=10")
    test_nan_detection_at_step_10()
    print()
    
    print("Test 2: NaN detection with check_every_n_steps=1")
    test_nan_detection_immediate()
    print()
    
    print("Test 3: Inf detection")
    test_inf_detection()
    print()
    
    print("✅ All NaN/Inf detection tests passed!")