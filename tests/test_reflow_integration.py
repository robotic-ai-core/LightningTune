"""
Tests for LightningReflow HPO integration.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch import Trainer

# Import the integration module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from LightningTune.core.reflow_integration import (
    ReflowHPOCallback,
    ReflowTrainable,
    create_reflow_trainable,
    REFLOW_AVAILABLE
)


# Test fixtures
class DummyModel(pl.LightningModule):
    """Simple model for testing."""
    
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.layer = nn.Linear(10, 2)
        self.save_hyperparameters()
    
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


class DummyDataModule(pl.LightningDataModule):
    """Simple datamodule for testing."""
    
    def __init__(self, batch_size=4):
        super().__init__()
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        # Create dummy data
        self.train_data = torch.utils.data.TensorDataset(
            torch.randn(20, 10),
            torch.randn(20, 2)
        )
        self.val_data = torch.utils.data.TensorDataset(
            torch.randn(10, 10),
            torch.randn(10, 2)
        )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False
        )


# Unit tests for ReflowHPOCallback
class TestReflowHPOCallback:
    """Test suite for ReflowHPOCallback."""
    
    def test_init(self):
        """Test callback initialization."""
        # Without should_pause
        callback = ReflowHPOCallback()
        assert callback.should_pause_fn is None
        assert callback._pause_triggered is False
        
        # With should_pause
        should_pause = lambda: True
        callback = ReflowHPOCallback(should_pause=should_pause)
        assert callback.should_pause_fn == should_pause
        assert callback._pause_triggered is False
    
    def test_checkpoint_dir_creation(self, tmp_path):
        """Test that checkpoint directory is created."""
        checkpoint_dir = tmp_path / "checkpoints"
        callback = ReflowHPOCallback(checkpoint_dir=str(checkpoint_dir))
        assert checkpoint_dir.exists()
    
    @patch('LightningTune.core.reflow_integration.PauseAction')
    def test_on_validation_end_no_pause(self, mock_pause_action):
        """Test validation end when pause is not triggered."""
        callback = ReflowHPOCallback(should_pause=lambda: False)
        
        # Mock the state machine
        callback._state_machine = Mock()
        
        # Create mock trainer and model
        trainer = Mock()
        pl_module = Mock()
        
        # Call on_validation_end
        callback.on_validation_end(trainer, pl_module)
        
        # Should not trigger pause
        callback._state_machine.transition.assert_not_called()
        assert callback._pause_triggered is False
    
    @patch('LightningTune.core.reflow_integration.PauseAction')
    def test_on_validation_end_with_pause(self, mock_pause_action):
        """Test validation end when pause is triggered."""
        callback = ReflowHPOCallback(should_pause=lambda: True)
        
        # Mock the state machine
        callback._state_machine = Mock()
        mock_pause_action.TOGGLE_PAUSE = "toggle_pause"
        
        # Create mock trainer and model
        trainer = Mock()
        pl_module = Mock()
        
        # Call on_validation_end
        callback.on_validation_end(trainer, pl_module)
        
        # Should trigger pause
        callback._state_machine.transition.assert_called_once_with("toggle_pause")
        assert callback._pause_triggered is True
    
    def test_pause_triggered_only_once(self):
        """Test that pause is only triggered once per session."""
        callback = ReflowHPOCallback(should_pause=lambda: True)
        
        # Mock the state machine
        callback._state_machine = Mock()
        
        # Create mock trainer and model
        trainer = Mock()
        pl_module = Mock()
        
        # First call should trigger
        callback.on_validation_end(trainer, pl_module)
        assert callback._state_machine.transition.call_count == 1
        
        # Second call should not trigger again
        callback.on_validation_end(trainer, pl_module)
        assert callback._state_machine.transition.call_count == 1
    
    def test_reset_pause_trigger(self):
        """Test resetting the pause trigger."""
        callback = ReflowHPOCallback(should_pause=lambda: True)
        callback._pause_triggered = True
        
        callback.reset_pause_trigger()
        assert callback._pause_triggered is False


# Unit tests for ReflowTrainable
class TestReflowTrainable:
    """Test suite for ReflowTrainable."""
    
    def test_init(self):
        """Test trainable initialization."""
        trainable = ReflowTrainable(
            model_class=DummyModel,
            datamodule_class=DummyDataModule,
            base_config={"trainer": {"max_epochs": 10}},
            should_pause=lambda: False
        )
        
        assert trainable.model_class == DummyModel
        assert trainable.datamodule_class == DummyDataModule
        assert trainable.base_config == {"trainer": {"max_epochs": 10}}
        assert trainable.should_pause() is False
        assert trainable.last_checkpoint_path is None
    
    @patch('LightningTune.core.reflow_integration.LightningReflow')
    @patch('LightningTune.core.reflow_integration.ReflowHPOCallback')
    def test_call_without_checkpoint(self, mock_callback_class, mock_reflow_class):
        """Test calling trainable without checkpoint."""
        # Setup mocks
        mock_callback = Mock()
        mock_callback.get_last_checkpoint.return_value = None
        mock_callback_class.return_value = mock_callback
        
        mock_reflow = Mock()
        mock_reflow.fit.return_value = {"status": "completed"}
        # Use _callback_metrics for mock compatibility
        import torch
        mock_reflow.trainer._callback_metrics = {
            "val_loss": torch.tensor(0.5),
            "train_loss": torch.tensor(0.4)
        }
        mock_reflow_class.return_value = mock_reflow
        
        # Create trainable
        trainable = ReflowTrainable(
            model_class=DummyModel,
            should_pause=lambda: False
        )
        
        # Call with config
        config = {"learning_rate": 1e-3}
        result = trainable(config)
        
        # Check LightningReflow was created correctly
        mock_reflow_class.assert_called_once()
        call_kwargs = mock_reflow_class.call_args.kwargs
        assert call_kwargs["model_class"] == DummyModel
        assert call_kwargs["config_overrides"] == config
        assert mock_callback in call_kwargs["callbacks"]
        
        # Check fit was called
        mock_reflow.fit.assert_called_once_with(ckpt_path=None)
        
        # Check result
        assert result["checkpoint_path"] is None
        assert abs(result["val_loss"] - 0.5) < 1e-6  # Use approximate equality
        assert abs(result["metrics"]["train_loss"] - 0.4) < 1e-6  # Use approximate equality
    
    @patch('LightningTune.core.reflow_integration.LightningReflow')
    @patch('LightningTune.core.reflow_integration.ReflowHPOCallback')
    def test_call_with_checkpoint(self, mock_callback_class, mock_reflow_class):
        """Test calling trainable with checkpoint to resume from."""
        # Setup mocks
        checkpoint_path = Path("/tmp/checkpoint.ckpt")
        
        mock_callback = Mock()
        mock_callback.get_last_checkpoint.return_value = checkpoint_path
        mock_callback_class.return_value = mock_callback
        
        mock_reflow = Mock()
        mock_reflow.fit.return_value = {"status": "completed"}
        # Use _callback_metrics for mock compatibility
        import torch
        mock_reflow.trainer._callback_metrics = {"val_loss": torch.tensor(0.3)}
        mock_reflow_class.return_value = mock_reflow
        
        # Create trainable
        trainable = ReflowTrainable(
            model_class=DummyModel,
            should_pause=lambda: True
        )
        
        # Call with checkpoint
        result = trainable({}, checkpoint_dir="/tmp/resume.ckpt")
        
        # Check fit was called with checkpoint
        mock_reflow.fit.assert_called_once_with(ckpt_path="/tmp/resume.ckpt")
        
        # Check result has checkpoint path
        assert result["checkpoint_path"] == str(checkpoint_path)
        assert trainable.last_checkpoint_path == str(checkpoint_path)
    
    def test_config_merging(self):
        """Test that base config and trial config are merged correctly."""
        with patch('LightningTune.core.reflow_integration.LightningReflow') as mock_reflow_class:
            mock_reflow = Mock()
            mock_reflow.fit.return_value = {}
            mock_reflow.trainer._callback_metrics = {}
            mock_reflow_class.return_value = mock_reflow
            
            trainable = ReflowTrainable(
                model_class=DummyModel,
                base_config={"trainer": {"max_epochs": 10}, "model": {"dropout": 0.1}}
            )
            
            trial_config = {"model": {"dropout": 0.2}, "optimizer": {"lr": 1e-3}}
            trainable(trial_config)
            
            # Check merged config
            call_kwargs = mock_reflow_class.call_args.kwargs
            expected_config = {
                "trainer": {"max_epochs": 10},
                "model": {"dropout": 0.2},  # Overridden by trial
                "optimizer": {"lr": 1e-3}
            }
            assert call_kwargs["config_overrides"] == expected_config


# Unit tests for factory function
class TestCreateReflowTrainable:
    """Test suite for create_reflow_trainable factory."""
    
    @patch('ray.tune')
    def test_with_ray_tune(self, mock_tune):
        """Test factory with Ray Tune available."""
        mock_tune.should_checkpoint.return_value = True
        
        trainable = create_reflow_trainable(
            model_class=DummyModel,
            base_config={"trainer": {"max_epochs": 10}}
        )
        
        assert isinstance(trainable, ReflowTrainable)
        assert trainable.model_class == DummyModel
        assert trainable.base_config == {"trainer": {"max_epochs": 10}}
        
        # Check that should_pause uses tune.should_checkpoint
        assert trainable.should_pause() is True
        mock_tune.should_checkpoint.assert_called()
    
    @patch.dict('sys.modules', {'ray.tune': None})
    def test_without_ray_tune(self):
        """Test factory when Ray Tune is not available."""
        # Need to reload module without ray.tune
        import importlib
        import LightningTune.core.reflow_integration as reflow_integration
        importlib.reload(reflow_integration)
        
        trainable = reflow_integration.create_reflow_trainable(
            model_class=DummyModel
        )
        
        assert isinstance(trainable, reflow_integration.ReflowTrainable)
        # Should default to False when Ray Tune not available
        assert trainable.should_pause() is False


# Integration test with real components
@pytest.mark.integration
class TestReflowIntegration:
    """Integration tests with more realistic scenarios."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.mark.skipif(not REFLOW_AVAILABLE, reason="LightningReflow not installed")
    def test_full_training_cycle(self, temp_dir):
        """Test a full training cycle with pause and resume."""
        # This would require LightningReflow to be installed
        # We'll mock it for now
        with patch('LightningTune.core.reflow_integration.LightningReflow') as mock_reflow_class:
            # Setup mock
            mock_reflow = Mock()
            mock_reflow.fit.return_value = {"epochs_trained": 5}
            import torch
            mock_reflow.trainer._callback_metrics = {"val_loss": torch.tensor(0.2)}
            mock_reflow_class.return_value = mock_reflow
            
            # Create trainable
            pause_count = 0
            def should_pause():
                nonlocal pause_count
                pause_count += 1
                return pause_count == 2  # Pause on second check
            
            trainable = ReflowTrainable(
                model_class=DummyModel,
                datamodule_class=DummyDataModule,
                should_pause=should_pause
            )
            
            # First training run
            config = {"learning_rate": 1e-3}
            result1 = trainable(config)
            
            # Simulate pause checkpoint
            with patch.object(trainable, 'last_checkpoint_path', str(temp_dir / "checkpoint.ckpt")):
                # Resume training
                result2 = trainable(config, checkpoint_dir=trainable.last_checkpoint_path)
                
                # Check that fit was called twice
                assert mock_reflow.fit.call_count == 2
                
                # Check checkpoint was used for resume
                second_call = mock_reflow.fit.call_args_list[1]
                assert second_call.kwargs["ckpt_path"] == str(temp_dir / "checkpoint.ckpt")
    
    def test_error_handling(self):
        """Test error handling in trainable."""
        with patch('LightningTune.core.reflow_integration.LightningReflow') as mock_reflow_class:
            # Make fit raise an exception
            mock_reflow = Mock()
            mock_reflow.fit.side_effect = RuntimeError("Training failed")
            mock_reflow_class.return_value = mock_reflow
            
            trainable = ReflowTrainable(model_class=DummyModel)
            
            # Should propagate the exception
            with pytest.raises(RuntimeError, match="Training failed"):
                trainable({})