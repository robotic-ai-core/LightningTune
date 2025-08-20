"""
Tests for NaN/Inf detection callbacks.

Tests the NaNDetectionCallback and EnhancedOptunaPruningCallback
to ensure they properly detect and handle NaN/Inf values during training.
"""

import math
import pytest
import torch
import optuna
from unittest.mock import Mock, MagicMock, patch
import lightning as L

# Import the callbacks
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from LightningTune.optuna.nan_detection_callback import (
    NaNDetectionCallback,
    EnhancedOptunaPruningCallback
)


class TestNaNDetectionCallback:
    """Test the NaN detection callback."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trial = Mock(spec=optuna.Trial)
        self.trial.number = 42
        self.trial.set_user_attr = Mock()
        
        self.trainer = Mock(spec=L.Trainer)
        self.trainer.global_step = 100
        self.trainer.current_epoch = 5
        # Initialize callback_metrics and logged_metrics as empty dicts
        self.trainer.callback_metrics = {}
        self.trainer.logged_metrics = {}
        
        self.pl_module = Mock(spec=L.LightningModule)
    
    def test_init_default_values(self):
        """Test callback initialization with default values."""
        callback = NaNDetectionCallback(self.trial)
        
        assert callback.trial == self.trial
        assert callback.monitor == "val_loss"
        assert callback.check_train_loss == True
        assert callback.check_every_n_steps == 100
        assert callback.verbose == True
    
    def test_check_value_detects_nan(self):
        """Test that _check_value correctly identifies NaN."""
        callback = NaNDetectionCallback(self.trial, verbose=False)
        
        assert callback._check_value(float('nan'), "test") == True
        assert callback._check_value(float('inf'), "test") == True
        assert callback._check_value(float('-inf'), "test") == True
        assert callback._check_value(1.0, "test") == False
        assert callback._check_value(0.0, "test") == False
    
    def test_on_train_batch_end_with_nan_in_outputs_dict(self):
        """Test NaN detection when outputs is a dict with NaN loss."""
        callback = NaNDetectionCallback(
            self.trial, 
            check_every_n_steps=1,  # Check every step for testing
            verbose=False
        )
        
        # Test with various loss keys
        test_cases = [
            {'loss': float('nan')},
            {'primary_loss': float('nan'), 'total_loss': 1.0},
            {'reconstruction_loss': 1.0, 'total_loss': float('inf')},
            {'CUSTOM_LOSS': float('nan')},  # Case insensitive
        ]
        
        for outputs in test_cases:
            with pytest.raises(optuna.TrialPruned) as exc_info:
                callback.on_train_batch_end(
                    self.trainer, self.pl_module, outputs, None, 0
                )
            assert "NaN/Inf" in str(exc_info.value)
            
            # Reset step count for next test
            callback.step_count = 0
    
    def test_on_train_batch_end_with_nan_tensor(self):
        """Test NaN detection when outputs is a tensor with NaN."""
        callback = NaNDetectionCallback(
            self.trial,
            check_every_n_steps=1,
            verbose=False
        )
        
        outputs = torch.tensor(float('nan'))
        
        with pytest.raises(optuna.TrialPruned) as exc_info:
            callback.on_train_batch_end(
                self.trainer, self.pl_module, outputs, None, 0
            )
        assert "NaN/Inf" in str(exc_info.value)
    
    def test_on_train_batch_end_checks_callback_metrics(self):
        """Test that callback checks trainer.callback_metrics for NaN."""
        callback = NaNDetectionCallback(
            self.trial,
            check_every_n_steps=1,
            verbose=False
        )
        
        # Outputs is None, but metrics contain NaN
        outputs = None
        self.trainer.callback_metrics = {
            'train/loss': torch.tensor(float('nan')),
            'val/loss': torch.tensor(1.0),  # Should ignore val during training
        }
        
        with pytest.raises(optuna.TrialPruned) as exc_info:
            callback.on_train_batch_end(
                self.trainer, self.pl_module, outputs, None, 0
            )
        assert "NaN/Inf" in str(exc_info.value)
    
    def test_on_train_batch_end_checks_multiple_losses(self):
        """Test that callback checks all loss components."""
        callback = NaNDetectionCallback(
            self.trial,
            check_every_n_steps=1,
            verbose=False
        )
        
        # Multiple losses, one is NaN
        outputs = {
            'primary_loss': torch.tensor(0.5),
            'auxiliary_loss': torch.tensor(0.3),
            'total_loss': torch.tensor(float('nan')),
            'accuracy': torch.tensor(0.95),  # Non-loss metric, should ignore
        }
        
        with pytest.raises(optuna.TrialPruned) as exc_info:
            callback.on_train_batch_end(
                self.trainer, self.pl_module, outputs, None, 0
            )
        assert "total_loss" in str(exc_info.value)
    
    def test_on_train_batch_end_respects_check_frequency(self):
        """Test that callback respects check_every_n_steps."""
        callback = NaNDetectionCallback(
            self.trial,
            check_every_n_steps=10,
            verbose=False
        )
        
        outputs = {'loss': float('nan')}
        
        # First 9 calls should not check
        for i in range(9):
            callback.on_train_batch_end(
                self.trainer, self.pl_module, outputs, None, i
            )  # Should not raise
        
        # 10th call should check and raise
        with pytest.raises(optuna.TrialPruned):
            callback.on_train_batch_end(
                self.trainer, self.pl_module, outputs, None, 9
            )
    
    def test_on_train_batch_end_skips_non_numeric(self):
        """Test that callback skips non-numeric values gracefully."""
        callback = NaNDetectionCallback(
            self.trial,
            check_every_n_steps=1,
            verbose=False
        )
        
        outputs = {
            'loss': torch.tensor(1.0),  # Valid loss
            'some_string_loss': "not_a_number",  # Should skip
            'none_loss': None,  # Should skip
        }
        
        # Should not raise since valid loss is OK
        callback.on_train_batch_end(
            self.trainer, self.pl_module, outputs, None, 0
        )
    
    def test_on_validation_end_with_nan(self):
        """Test validation NaN detection."""
        callback = NaNDetectionCallback(self.trial, verbose=False)
        
        self.trainer.callback_metrics = {
            'val_loss': torch.tensor(float('nan'))
        }
        
        with pytest.raises(optuna.TrialPruned) as exc_info:
            callback.on_validation_end(self.trainer, self.pl_module)
        assert "val_loss is NaN/Inf" in str(exc_info.value)
    
    def test_trial_user_attr_set_on_nan(self):
        """Test that trial user attribute is set when NaN is detected."""
        callback = NaNDetectionCallback(
            self.trial,
            check_every_n_steps=1,
            verbose=False
        )
        
        outputs = {'loss': float('nan')}
        
        with pytest.raises(optuna.TrialPruned):
            callback.on_train_batch_end(
                self.trainer, self.pl_module, outputs, None, 0
            )
        
        self.trial.set_user_attr.assert_called_with('failed_reason', 'nan_or_inf_loss')


class TestEnhancedOptunaPruningCallback:
    """Test the enhanced Optuna pruning callback with NaN detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trial = Mock(spec=optuna.Trial)
        self.trial.number = 42
        self.trial.set_user_attr = Mock()
        self.trial.report = Mock()
        self.trial.should_prune = Mock(return_value=False)
        
        self.trainer = Mock(spec=L.Trainer)
        self.trainer.current_epoch = 5
        
        self.pl_module = Mock(spec=L.LightningModule)
    
    def test_enhanced_callback_detects_nan(self):
        """Test that enhanced callback detects NaN in validation."""
        callback = EnhancedOptunaPruningCallback(
            self.trial,
            monitor="val_loss",
            check_nan=True,
            verbose=False
        )
        
        self.trainer.callback_metrics = {
            'val_loss': torch.tensor(float('nan'))
        }
        
        with pytest.raises(optuna.TrialPruned) as exc_info:
            callback.on_validation_end(self.trainer, self.pl_module)
        assert "NaN/Inf" in str(exc_info.value)
        
        # Should set user attribute
        self.trial.set_user_attr.assert_called_with('failed_reason', 'nan_or_inf_loss')
        # Should NOT report to Optuna (because it's NaN)
        self.trial.report.assert_not_called()
    
    def test_enhanced_callback_normal_pruning(self):
        """Test that enhanced callback still does normal pruning."""
        callback = EnhancedOptunaPruningCallback(
            self.trial,
            monitor="val_loss",
            check_nan=True,
            verbose=False
        )
        
        # Valid loss value
        self.trainer.callback_metrics = {
            'val_loss': torch.tensor(0.5)
        }
        
        # Mock should_prune to return True
        self.trial.should_prune.return_value = True
        
        with pytest.raises(optuna.TrialPruned):
            callback.on_validation_end(self.trainer, self.pl_module)
        
        # Should report the value
        self.trial.report.assert_called_with(0.5, 5)
        # Should check for pruning
        self.trial.should_prune.assert_called()
    
    def test_enhanced_callback_nan_check_disabled(self):
        """Test that NaN check can be disabled."""
        callback = EnhancedOptunaPruningCallback(
            self.trial,
            monitor="val_loss",
            check_nan=False,  # Disabled
            verbose=False
        )
        
        self.trainer.callback_metrics = {
            'val_loss': torch.tensor(float('nan'))
        }
        
        # Should still call report with NaN (no exception)
        callback.on_validation_end(self.trainer, self.pl_module)
        
        # NaN gets reported (which Optuna handles)
        assert self.trial.report.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])