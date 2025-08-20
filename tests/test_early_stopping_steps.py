"""
Tests for EarlyStoppingSteps callback.

Verifies that the callback stops training at the specified step
without affecting the learning rate scheduler.
"""

import pytest
import torch
import lightning as L
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from LightningTune.callbacks.early_stopping_steps import EarlyStoppingSteps


class TestEarlyStoppingSteps:
    """Test the early stopping steps callback."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trainer = Mock(spec=L.Trainer)
        self.trainer.global_step = 0
        self.trainer.should_stop = False
        
        self.pl_module = Mock(spec=L.LightningModule)
    
    def test_init(self):
        """Test callback initialization."""
        callback = EarlyStoppingSteps(stopping_steps=1000, verbose=True)
        
        assert callback.stopping_steps == 1000
        assert callback.verbose == True
        assert callback._should_stop == False
    
    def test_stops_at_target_steps(self):
        """Test that training stops at the target step count."""
        callback = EarlyStoppingSteps(stopping_steps=100, verbose=False)
        
        # Simulate training for 99 steps - should not stop
        for step in range(99):
            self.trainer.global_step = step
            callback.on_train_batch_end(
                self.trainer, self.pl_module, None, None, step
            )
            assert self.trainer.should_stop == False
        
        # Step 100 - should stop
        self.trainer.global_step = 100
        callback.on_train_batch_end(
            self.trainer, self.pl_module, None, None, 100
        )
        assert self.trainer.should_stop == True
        assert callback._should_stop == True
    
    def test_stops_at_validation_end(self):
        """Test that training can also stop at validation end."""
        callback = EarlyStoppingSteps(stopping_steps=100, verbose=False)
        
        # Set global step past threshold
        self.trainer.global_step = 105
        
        callback.on_validation_end(self.trainer, self.pl_module)
        
        assert self.trainer.should_stop == True
        assert callback._should_stop == True
    
    def test_verbose_logging(self, caplog):
        """Test that verbose mode logs the stopping message."""
        callback = EarlyStoppingSteps(stopping_steps=50, verbose=True)
        
        self.trainer.global_step = 50
        
        with caplog.at_level("INFO"):
            callback.on_train_batch_end(
                self.trainer, self.pl_module, None, None, 50
            )
        
        assert "EarlyStoppingSteps: Stopping at step 50" in caplog.text
        assert "(target: 50)" in caplog.text
    
    def test_no_double_logging(self, caplog):
        """Test that stopping message is only logged once."""
        callback = EarlyStoppingSteps(stopping_steps=50, verbose=True)
        
        self.trainer.global_step = 50
        
        with caplog.at_level("INFO"):
            # First call - should log
            callback.on_train_batch_end(
                self.trainer, self.pl_module, None, None, 50
            )
            initial_count = caplog.text.count("EarlyStoppingSteps")
            
            # Second call - should not log again
            callback.on_train_batch_end(
                self.trainer, self.pl_module, None, None, 51
            )
            final_count = caplog.text.count("EarlyStoppingSteps")
        
        assert initial_count == final_count == 1
    
    def test_does_not_affect_lr_scheduler(self):
        """
        Test that EarlyStoppingSteps doesn't affect LR scheduler calculation.
        
        This is a conceptual test - the actual LR scheduler uses
        trainer.estimated_stepping_batches which is calculated from
        trainer.max_steps, not affected by our callback.
        """
        # Create a mock trainer with max_steps set
        trainer = Mock(spec=L.Trainer)
        trainer.max_steps = 10000  # Full training length
        trainer.global_step = 0
        trainer.should_stop = False
        
        # Calculate estimated steps (simplified version)
        # In reality, Lightning does: min(max_steps, len(train_dataloader) * max_epochs)
        estimated_steps = trainer.max_steps
        
        # Create callback with early stopping
        callback = EarlyStoppingSteps(stopping_steps=1000)  # Stop at 10% of max_steps
        
        # Verify that max_steps is unchanged
        assert trainer.max_steps == 10000
        
        # Simulate training stop
        trainer.global_step = 1000
        callback.on_train_batch_end(trainer, self.pl_module, None, None, 1000)
        
        # Verify trainer.should_stop is set but max_steps unchanged
        assert trainer.should_stop == True
        assert trainer.max_steps == 10000  # Still the original value
        
        # This means LR scheduler would still calculate based on 10000 steps
        assert estimated_steps == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])