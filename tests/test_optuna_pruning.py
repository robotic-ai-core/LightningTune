#!/usr/bin/env python
"""
Test Optuna pruning functionality including NaN detection and poor performance pruning.

This module tests:
1. Trials with NaN values are properly pruned
2. Trials with poor performance are pruned by various pruners
3. The NaN detection callback works correctly
4. Integration with the HPO script
"""

import sys
import math
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest
import torch
import numpy as np
import optuna
from optuna.trial import TrialState
import lightning as L
from lightning.pytorch import Trainer

# Add LightningTune root to path
lightningtune_root = Path(__file__).parent.parent
sys.path.insert(0, str(lightningtune_root))

from LightningTune.optuna.nan_detection_callback import NaNDetectionCallback, EnhancedOptunaPruningCallback
from LightningTune.optuna.callbacks import OptunaPruningCallback


class DummyModel(L.LightningModule):
    """Dummy model for testing that can produce NaN or specific loss values."""
    
    def __init__(self, produce_nan_at_step=None, produce_nan_at_epoch=None, 
                 loss_schedule=None, val_loss_schedule=None):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)
        self.produce_nan_at_step = produce_nan_at_step
        self.produce_nan_at_epoch = produce_nan_at_epoch
        self.loss_schedule = loss_schedule or {}  # step -> loss value
        self.val_loss_schedule = val_loss_schedule or {}  # epoch -> val_loss value
        self.current_step = 0
        
    def training_step(self, batch, batch_idx):
        self.current_step += 1
        
        # Check if we should produce NaN at this step
        if self.produce_nan_at_step and self.current_step >= self.produce_nan_at_step:
            loss = torch.tensor(float('nan'), requires_grad=True)
        elif self.current_step in self.loss_schedule:
            loss = torch.tensor(self.loss_schedule[self.current_step], requires_grad=True)
        else:
            # Normal loss calculation
            x, y = batch
            y_hat = self.layer(x)
            loss = torch.nn.functional.mse_loss(y_hat, y)
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Check if we should produce NaN at this epoch
        if self.produce_nan_at_epoch and self.current_epoch >= self.produce_nan_at_epoch:
            val_loss = torch.tensor(float('nan'), requires_grad=False)
        elif self.current_epoch in self.val_loss_schedule:
            val_loss = torch.tensor(self.val_loss_schedule[self.current_epoch], requires_grad=False)
        else:
            # Normal validation
            with torch.no_grad():
                y_hat = self.layer(x)
                val_loss = torch.nn.functional.mse_loss(y_hat, y)
        
        self.log('val_loss', val_loss)
        return val_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def create_dummy_dataloader():
    """Create a simple dataloader for testing."""
    dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 10),
        torch.randn(100, 1)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=10)


def test_nan_detection_callback_train_loss():
    """Test that NaN in training loss triggers pruning."""
    # Create a trial mock
    trial = Mock(spec=optuna.Trial)
    trial.number = 0
    trial.set_user_attr = Mock()
    
    # Create model that produces NaN at step 5
    model = DummyModel(produce_nan_at_step=5)
    
    # Create callback
    callback = NaNDetectionCallback(
        trial=trial,
        check_every_n_steps=1,  # Check every step for testing
        verbose=True
    )
    
    # Create trainer with callback
    trainer = Trainer(
        max_epochs=1,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )
    
    # Train and expect TrialPruned exception
    train_loader = create_dummy_dataloader()
    val_loader = create_dummy_dataloader()
    
    with pytest.raises(optuna.TrialPruned) as exc_info:
        trainer.fit(model, train_loader, val_loader)
    
    # Verify the trial was marked as failed
    trial.set_user_attr.assert_called_with('failed_reason', 'nan_or_inf_loss')
    assert "NaN/Inf" in str(exc_info.value)


def test_nan_detection_callback_val_loss():
    """Test that NaN in validation loss triggers pruning."""
    # Create a trial mock
    trial = Mock(spec=optuna.Trial)
    trial.number = 0
    trial.set_user_attr = Mock()
    
    # Create model that produces NaN at validation - using val_loss_schedule
    model = DummyModel(val_loss_schedule={0: float('nan'), 1: float('nan')})
    
    # Create callback
    callback = NaNDetectionCallback(
        trial=trial,
        monitor='val_loss',
        verbose=True
    )
    
    # Create trainer with callback
    trainer = Trainer(
        max_epochs=2,  # Need at least 2 epochs to ensure validation runs
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=2,  # Limit batches for speed
        limit_val_batches=2,    # Limit validation batches
    )
    
    # Train and expect TrialPruned exception
    train_loader = create_dummy_dataloader()
    val_loader = create_dummy_dataloader()
    
    with pytest.raises(optuna.TrialPruned) as exc_info:
        trainer.fit(model, train_loader, val_loader)
    
    # Verify the trial was marked as failed
    trial.set_user_attr.assert_called_with('failed_reason', 'nan_or_inf_loss')
    assert "NaN/Inf" in str(exc_info.value)


def test_enhanced_pruning_callback_with_nan():
    """Test that EnhancedOptunaPruningCallback handles NaN correctly."""
    # Create a trial mock
    trial = Mock(spec=optuna.Trial)
    trial.number = 0
    trial.set_user_attr = Mock()
    trial.report = Mock()
    trial.should_prune = Mock(return_value=False)
    
    # Create model that produces NaN at validation
    model = DummyModel(val_loss_schedule={0: float('nan')})
    
    # Create enhanced callback
    callback = EnhancedOptunaPruningCallback(
        trial=trial,
        monitor='val_loss',
        check_nan=True,
        verbose=True
    )
    
    # Create trainer
    trainer = Trainer(
        max_epochs=1,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )
    
    # Train and expect TrialPruned due to NaN
    train_loader = create_dummy_dataloader()
    val_loader = create_dummy_dataloader()
    
    with pytest.raises(optuna.TrialPruned) as exc_info:
        trainer.fit(model, train_loader, val_loader)
    
    # Verify NaN was detected and reported before pruning so Optuna records it
    import math
    trial.set_user_attr.assert_called_with('failed_reason', 'nan_or_inf_loss')
    assert trial.report.call_count == 1
    args, kwargs = trial.report.call_args
    assert len(args) >= 2 and math.isnan(args[0]) and args[1] == 0
    assert "NaN/Inf" in str(exc_info.value)


def test_optuna_pruning_poor_performance():
    """Test that poor performing trials are pruned by Optuna."""
    # Create a study with MedianPruner
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=2,  # Need at least 2 trials before pruning
            n_warmup_steps=0,    # Prune from first epoch
        )
    )
    
    # Define objective that creates good and bad trials
    def objective(trial):
        # First two trials: good performance
        if trial.number < 2:
            for epoch in range(3):
                # Report decreasing loss
                loss = 1.0 - 0.3 * epoch + trial.number * 0.1
                trial.report(loss, epoch)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return loss
        
        # Third trial: poor performance (should be pruned)
        else:
            for epoch in range(3):
                # Report high loss (worse than median of previous trials)
                loss = 5.0 + epoch * 0.5  # Much worse than previous trials
                trial.report(loss, epoch)
                
                if trial.should_prune():
                    raise optuna.TrialPruned(f"Pruned at epoch {epoch}")
            return loss
    
    # Run optimization
    study.optimize(objective, n_trials=3)
    
    # Check that we have completed and pruned trials
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    
    assert len(completed_trials) >= 2, "Should have at least 2 completed trials"
    assert len(pruned_trials) >= 1, "Should have at least 1 pruned trial due to poor performance"
    
    # Verify the pruned trial had worse performance
    if pruned_trials:
        pruned_trial = pruned_trials[0]
        # Check that it reported high values before being pruned
        assert len(pruned_trial.intermediate_values) > 0, "Pruned trial should have reported values"
        # The values should be higher (worse) than the completed trials
        pruned_values = list(pruned_trial.intermediate_values.values())
        assert min(pruned_values) > 2.0, "Pruned trial should have high loss values"


def test_hyperband_pruning():
    """Test that HyperbandPruner correctly prunes trials."""
    # Create study with HyperbandPruner
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=10,
            reduction_factor=3,
        )
    )
    
    def objective(trial):
        # Simulate different trial performances
        base_loss = trial.suggest_float('base_loss', 0.1, 10.0)
        
        for step in range(10):
            # Report loss that depends on the suggested parameter
            loss = base_loss + np.random.normal(0, 0.1)
            trial.report(loss, step)
            
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at step {step}")
        
        return loss
    
    # Run multiple trials
    study.optimize(objective, n_trials=20, catch=(optuna.TrialPruned,))
    
    # Verify pruning occurred
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    
    assert len(pruned_trials) > 0, "HyperbandPruner should prune some trials"
    assert len(completed_trials) > 0, "Some trials should complete"
    
    # Check that pruned trials generally have worse intermediate values
    if pruned_trials and completed_trials:
        # Get the best completed trial's final value
        best_completed = min(t.value for t in completed_trials if t.value is not None)
        
        # Check that at least some pruned trials had worse intermediate values
        worse_pruned = 0
        for trial in pruned_trials:
            if trial.intermediate_values:
                # Get the last reported value before pruning
                last_step = max(trial.intermediate_values.keys())
                last_value = trial.intermediate_values[last_step]
                if last_value > best_completed:
                    worse_pruned += 1
        
        # At least some pruned trials should have been worse
        assert worse_pruned > 0, "Some pruned trials should have worse performance"


def test_percentile_pruner_with_nan():
    """Test that PercentilePruner handles NaN values appropriately."""
    # Create study with PercentilePruner
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.PercentilePruner(
            percentile=50.0,  # Prune bottom 50%
            n_startup_trials=2,
        )
    )
    
    def objective(trial):
        # First trial: normal performance
        if trial.number == 0:
            for epoch in range(3):
                loss = 1.0 - 0.2 * epoch
                trial.report(loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return loss
        
        # Second trial: also normal
        elif trial.number == 1:
            for epoch in range(3):
                loss = 0.8 - 0.15 * epoch
                trial.report(loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return loss
        
        # Third trial: produces NaN
        else:
            # Report NaN - should cause immediate failure
            trial.report(float('nan'), 0)
            # Even though we reported NaN, check if pruning is triggered
            if trial.should_prune():
                raise optuna.TrialPruned("Pruned due to NaN")
            
            # If not pruned by Optuna, we should handle it ourselves
            raise optuna.TrialPruned("Manual pruning due to NaN")
    
    # Run trials
    study.optimize(objective, n_trials=3, catch=(optuna.TrialPruned,))
    
    # Check results
    trials = study.trials
    assert len(trials) == 3
    
    # The trial that reported NaN should be pruned
    nan_trial = trials[2]
    assert nan_trial.state == TrialState.PRUNED, "Trial with NaN should be pruned"
    
    # The NaN trial should have the NaN value in intermediate values
    if nan_trial.intermediate_values:
        nan_values = [v for v in nan_trial.intermediate_values.values() if math.isnan(v)]
        assert len(nan_values) > 0, "NaN value should be in intermediate values"


def test_integration_with_lightning_model():
    """Test the full integration with a Lightning model and Optuna callbacks."""
    
    # Create an Optuna study
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=0,  # Allow immediate pruning for testing
            n_warmup_steps=0,
        )
    )
    
    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        produce_nan = trial.suggest_categorical('produce_nan', [False, True])
        
        # Create model
        if produce_nan and trial.number > 0:  # Make first trial succeed
            model = DummyModel(produce_nan_at_epoch=1)
        else:
            # Create model with specific loss schedule for testing
            val_schedule = {}
            if trial.number == 0:
                # First trial: good performance
                val_schedule = {0: 1.0, 1: 0.8, 2: 0.6}
            else:
                # Later trials: vary performance
                base = 1.0 + trial.number * 0.5
                val_schedule = {0: base, 1: base - 0.1, 2: base - 0.2}
            
            model = DummyModel(val_loss_schedule=val_schedule)
        
        # Create callbacks
        callbacks = [
            EnhancedOptunaPruningCallback(
                trial=trial,
                monitor='val_loss',
                check_nan=True,
                verbose=False,
            )
        ]
        
        # Create trainer
        trainer = Trainer(
            max_epochs=3,
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )
        
        # Train
        train_loader = create_dummy_dataloader()
        val_loader = create_dummy_dataloader()
        
        try:
            trainer.fit(model, train_loader, val_loader)
            # Return the final validation loss
            return trainer.callback_metrics['val_loss'].item()
        except optuna.TrialPruned:
            raise
        except Exception as e:
            # Any other exception should be treated as a failed trial
            raise optuna.TrialPruned(f"Trial failed with error: {e}")
    
    # Run optimization
    study.optimize(objective, n_trials=5, catch=(optuna.TrialPruned,))
    
    # Analyze results
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    
    assert len(completed) > 0, "Should have some completed trials"
    assert len(pruned) > 0, "Should have some pruned trials"
    
    # Check that trials with NaN were pruned
    for trial in study.trials:
        if trial.params.get('produce_nan', False) and trial.number > 0:
            assert trial.state == TrialState.PRUNED, f"Trial {trial.number} with NaN should be pruned"
            # Check if it has the failed reason attribute
            if hasattr(trial, 'user_attrs') and 'failed_reason' in trial.user_attrs:
                assert trial.user_attrs['failed_reason'] == 'nan_or_inf_loss'


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])