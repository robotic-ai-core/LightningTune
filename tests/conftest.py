"""
Pytest configuration and fixtures.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import checking
try:
    import pytorch_lightning
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def basic_config_file(temp_dir):
    """Create a basic config file for testing."""
    config = {
        "model": {
            "class_path": "tests.fixtures.dummy_model.DummyModel",
            "init_args": {
                "input_dim": 10,
                "hidden_dim": 32,
                "output_dim": 2,
                "learning_rate": 0.001,
            }
        },
        "data": {
            "class_path": "tests.fixtures.dummy_model.DummyDataModule",
            "init_args": {
                "batch_size": 32,
                "num_samples": 100,
            }
        },
        "trainer": {
            "max_epochs": 2,
            "accelerator": "cpu",
            "devices": 1,
            "enable_progress_bar": False,
            "logger": False,
        }
    }
    
    config_path = temp_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


@pytest.fixture
def mock_search_space():
    """Create a mock search space."""
    from LightningTune import SearchSpace
    
    class MockSearchSpace(SearchSpace):
        def get_search_space(self):
            return {
                "model.init_args.learning_rate": [0.001, 0.01],
                "model.init_args.hidden_dim": [16, 32, 64],
            }
        
        def get_metric_config(self):
            return {"metric": "val_loss", "mode": "min"}
    
    return MockSearchSpace()


@pytest.fixture
def mock_strategy():
    """Create a mock strategy."""
    from LightningTune.optuna.strategies import OptunaBOHBStrategy
    
    return OptunaBOHBStrategy(
        n_trials=10,
    )


# Skip markers for missing dependencies
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_pytorch: mark test as requiring PyTorch"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on available dependencies."""
    skip_pytorch = pytest.mark.skip(reason="PyTorch not installed")
    
    for item in items:
        if "requires_pytorch" in item.keywords and not PYTORCH_AVAILABLE:
            item.add_marker(skip_pytorch)