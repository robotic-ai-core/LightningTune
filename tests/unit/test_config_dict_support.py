"""
Unit tests for dict configuration support in ConfigDrivenOptimizer.

Tests that the optimizer can accept both file paths and dict configurations.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from LightningTune.core.config import ConfigManager
from LightningTune.core.optimizer import ConfigDrivenOptimizer
from LightningTune.core.strategies import RandomSearchStrategy


class TestConfigManager:
    """Test ConfigManager with dict support."""
    
    def test_init_with_dict(self):
        """Test ConfigManager initialization with a dictionary."""
        config_dict = {
            "model": {
                "learning_rate": 0.001,
                "hidden_dim": 128
            },
            "data": {
                "batch_size": 32
            }
        }
        
        manager = ConfigManager(base_config_source=config_dict)
        
        assert manager.base_config == config_dict
        assert manager.base_config_path is None
        assert manager.base_config_source == config_dict
    
    def test_init_with_path(self):
        """Test ConfigManager initialization with a file path."""
        config_dict = {
            "model": {"learning_rate": 0.001},
            "data": {"batch_size": 32}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            config_path = Path(f.name)
        
        try:
            manager = ConfigManager(base_config_source=config_path)
            
            assert manager.base_config == config_dict
            assert manager.base_config_path == config_path
            assert manager.base_config_source == config_path
        finally:
            config_path.unlink()
    
    def test_init_with_string_path(self):
        """Test ConfigManager initialization with a string path."""
        config_dict = {
            "model": {"learning_rate": 0.001},
            "data": {"batch_size": 32}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            config_path = f.name
        
        try:
            manager = ConfigManager(base_config_source=config_path)
            
            assert manager.base_config == config_dict
            assert manager.base_config_path == Path(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_init_with_none(self):
        """Test ConfigManager initialization with None."""
        manager = ConfigManager(base_config_source=None)
        
        assert manager.base_config == {}
        assert manager.base_config_path is None
        assert manager.base_config_source is None
    
    def test_merge_with_dict_source(self):
        """Test config merging when initialized with dict."""
        base_dict = {
            "model": {
                "learning_rate": 0.001,
                "hidden_dim": 128
            },
            "data": {
                "batch_size": 32
            }
        }
        
        manager = ConfigManager(base_config_source=base_dict)
        
        overrides = {
            "model.learning_rate": 0.01,
            "model.dropout": 0.2
        }
        
        merged = manager.merge_configs(manager.base_config, overrides)
        
        assert merged["model"]["learning_rate"] == 0.01
        assert merged["model"]["hidden_dim"] == 128
        assert merged["model"]["dropout"] == 0.2
        assert merged["data"]["batch_size"] == 32


class TestConfigDrivenOptimizer:
    """Test ConfigDrivenOptimizer with dict support."""
    
    def test_optimizer_with_dict_config(self):
        """Test optimizer initialization with dict configuration."""
        config_dict = {
            "model": {
                "class_path": "test.Model",
                "init_args": {
                    "learning_rate": 0.001,
                    "hidden_dim": 128
                }
            },
            "data": {
                "class_path": "test.DataModule",
                "init_args": {
                    "batch_size": 32
                }
            },
            "trainer": {
                "max_epochs": 10
            }
        }
        
        search_space = {
            "model.init_args.learning_rate": [0.001, 0.01],
        }
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source=config_dict,
            search_space=search_space,
            strategy=RandomSearchStrategy(num_samples=1),
        )
        
        assert optimizer.base_config_dict == config_dict
        assert optimizer.base_config_path is None
        assert optimizer.config_manager.base_config == config_dict
    
    def test_optimizer_with_path_config(self):
        """Test optimizer initialization with path configuration."""
        config_dict = {
            "model": {"learning_rate": 0.001},
            "data": {"batch_size": 32},
            "trainer": {"max_epochs": 10}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            config_path = Path(f.name)
        
        try:
            search_space = {
                "model.learning_rate": [0.001, 0.01],
            }
            
            optimizer = ConfigDrivenOptimizer(
                base_config_source=config_path,
                search_space=search_space,
                strategy=RandomSearchStrategy(num_samples=1),
            )
            
            assert optimizer.base_config_path == config_path
            assert optimizer.base_config_dict is None
            assert optimizer.config_manager.base_config == config_dict
        finally:
            config_path.unlink()
    
    def test_backward_compatibility_warning(self):
        """Test that using old base_config_path parameter shows deprecation warning."""
        config_dict = {"model": {}, "data": {}, "trainer": {}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            config_path = Path(f.name)
        
        try:
            with pytest.warns(DeprecationWarning, match="base_config_path is deprecated"):
                optimizer = ConfigDrivenOptimizer(
                    base_config_source={},  # Provide empty dict
                    base_config_path=config_path,  # Use deprecated parameter
                    search_space={},
                    strategy=RandomSearchStrategy(num_samples=1),
                )
                
                # Should use the deprecated parameter value
                assert optimizer.base_config_path == config_path
        finally:
            config_path.unlink()
    
    def test_optimizer_trainable_with_dict(self):
        """Test that trainable function works with dict config."""
        config_dict = {
            "model": {
                "class_path": "fixtures.dummy_model.DummyModel",
                "init_args": {"learning_rate": 0.001}
            },
            "data": {
                "class_path": "fixtures.dummy_model.DummyDataModule",
                "init_args": {"batch_size": 32}
            },
            "trainer": {"max_epochs": 1}
        }
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source=config_dict,
            search_space={"model.init_args.learning_rate": [0.001, 0.01]},
            strategy=RandomSearchStrategy(num_samples=1),
        )
        
        # Create trainable function
        trainable_fn = optimizer._create_trainable()
        
        # Verify it's callable
        assert callable(trainable_fn)
        
        # Test that config manager merges correctly
        test_config = {"model.init_args.learning_rate": 0.01}
        merged = optimizer.config_manager.merge_configs(
            optimizer.config_manager.base_config,
            test_config
        )
        
        assert merged["model"]["init_args"]["learning_rate"] == 0.01
        assert merged["data"]["init_args"]["batch_size"] == 32
    
    def test_dict_vs_file_equivalence(self):
        """Test that dict and file configs produce equivalent results."""
        config_dict = {
            "model": {
                "class_path": "test.Model",
                "init_args": {"learning_rate": 0.001, "hidden_dim": 128}
            },
            "data": {
                "class_path": "test.DataModule",
                "init_args": {"batch_size": 32}
            },
            "trainer": {"max_epochs": 10}
        }
        
        # Create optimizer with dict
        optimizer_dict = ConfigDrivenOptimizer(
            base_config_source=config_dict,
            search_space={"model.init_args.learning_rate": [0.001, 0.01]},
            strategy=RandomSearchStrategy(num_samples=1),
        )
        
        # Create optimizer with file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            config_path = Path(f.name)
        
        try:
            optimizer_file = ConfigDrivenOptimizer(
                base_config_source=config_path,
                search_space={"model.init_args.learning_rate": [0.001, 0.01]},
                strategy=RandomSearchStrategy(num_samples=1),
            )
            
            # Both should have the same base config
            assert optimizer_dict.config_manager.base_config == optimizer_file.config_manager.base_config
            
            # Test merging produces same results
            test_overrides = {"model.init_args.learning_rate": 0.005}
            
            merged_dict = optimizer_dict.config_manager.merge_configs(
                optimizer_dict.config_manager.base_config,
                test_overrides
            )
            
            merged_file = optimizer_file.config_manager.merge_configs(
                optimizer_file.config_manager.base_config,
                test_overrides
            )
            
            assert merged_dict == merged_file
            
        finally:
            config_path.unlink()


class TestConfigDictIntegration:
    """Integration tests for dict config support."""
    
    @patch('LightningTune.core.optimizer.ray')
    @patch('LightningTune.core.optimizer.tune.Tuner')
    def test_run_optimization_with_dict(self, mock_tuner, mock_ray):
        """Test running optimization with dict config."""
        # Setup mocks
        mock_ray.is_initialized.return_value = False
        mock_results = MagicMock()
        mock_results.get_best_result.return_value = MagicMock(
            config={"model.init_args.learning_rate": 0.005},
            metrics={"val_loss": 0.5}
        )
        mock_results.__len__ = MagicMock(return_value=1)
        mock_tuner.return_value.fit.return_value = mock_results
        
        config_dict = {
            "model": {
                "class_path": "test.Model",
                "init_args": {"learning_rate": 0.001}
            },
            "data": {
                "class_path": "test.DataModule",
                "init_args": {"batch_size": 32}
            },
            "trainer": {"max_epochs": 1}
        }
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source=config_dict,
            search_space={"model.init_args.learning_rate": [0.001, 0.01]},
            strategy=RandomSearchStrategy(num_samples=2),
        )
        
        # Run optimization
        results = optimizer.run()
        
        # Verify results
        assert results is not None
        assert mock_ray.init.called
        assert mock_tuner.return_value.fit.called
        
        # Check best config retrieval
        best_config = optimizer.get_best_config()
        assert best_config == {"model.init_args.learning_rate": 0.005}
    
    def test_complex_nested_dict_config(self):
        """Test with complex nested dictionary configuration."""
        config_dict = {
            "model": {
                "class_path": "model.ComplexModel",
                "init_args": {
                    "encoder": {
                        "hidden_dims": [128, 256, 512],
                        "dropout": 0.2,
                        "activation": "relu"
                    },
                    "decoder": {
                        "hidden_dims": [512, 256, 128],
                        "dropout": 0.3,
                        "activation": "tanh"
                    },
                    "optimizer": {
                        "type": "adam",
                        "lr": 0.001,
                        "betas": [0.9, 0.999]
                    }
                }
            },
            "data": {
                "train": {
                    "batch_size": 32,
                    "shuffle": True
                },
                "val": {
                    "batch_size": 64,
                    "shuffle": False
                }
            }
        }
        
        optimizer = ConfigDrivenOptimizer(
            base_config_source=config_dict,
            search_space={
                "model.init_args.encoder.dropout": [0.1, 0.2, 0.3],
                "model.init_args.optimizer.lr": [0.0001, 0.001, 0.01]
            },
            strategy=RandomSearchStrategy(num_samples=1),
        )
        
        # Test merging with complex structure
        overrides = {
            "model.init_args.encoder.dropout": 0.15,
            "model.init_args.optimizer.lr": 0.0005,
            "data.train.batch_size": 64
        }
        
        merged = optimizer.config_manager.merge_configs(
            optimizer.config_manager.base_config,
            overrides
        )
        
        # Verify complex merge
        assert merged["model"]["init_args"]["encoder"]["dropout"] == 0.15
        assert merged["model"]["init_args"]["encoder"]["hidden_dims"] == [128, 256, 512]
        assert merged["model"]["init_args"]["decoder"]["dropout"] == 0.3
        assert merged["model"]["init_args"]["optimizer"]["lr"] == 0.0005
        assert merged["model"]["init_args"]["optimizer"]["betas"] == [0.9, 0.999]
        assert merged["data"]["train"]["batch_size"] == 64
        assert merged["data"]["val"]["batch_size"] == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])