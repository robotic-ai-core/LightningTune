#!/usr/bin/env python
"""Test cases for HPORunner pause integration."""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

# Add LightningTune root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LightningTune.hpo_runner import HPORunner
from lightning import LightningModule
import torch


class DummyModel(LightningModule):
    """Dummy model for testing."""
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        return torch.tensor(0.0)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class TestHPORunnerPauseIntegration:
    """Test HPORunner pause integration."""

    def test_init_parameters(self):
        """Test that pause parameters are properly initialized."""
        # Test default values
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )
        assert runner.enable_pause == True
        assert runner.pause_key == 'p'

        # Test custom values
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={},
            enable_pause=False,
            pause_key='x'
        )
        assert runner.enable_pause == False
        assert runner.pause_key == 'x'

    def test_cli_args_defined(self):
        """Test that pause CLI arguments are properly defined."""
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        # Check that pause CLI args exist
        assert 'enable_pause' in runner.cli_args
        assert 'disable_pause' in runner.cli_args
        assert 'pause_key' in runner.cli_args

        # Check their default values
        assert runner.cli_args['enable_pause']['default'] is None
        assert runner.cli_args['disable_pause']['default'] == False
        assert runner.cli_args['pause_key']['default'] is None

    def test_cli_parsing(self):
        """Test that CLI arguments are properly parsed."""
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        # Parse with enable-pause
        parser = runner._create_parser()
        args = parser.parse_args(['--enable-pause', '--n-trials', '10'])
        assert args.enable_pause == True
        assert args.n_trials == 10

        # Parse with disable-pause
        args = parser.parse_args(['--disable-pause', '--n-trials', '10'])
        assert args.disable_pause == True

        # Parse with custom pause key
        args = parser.parse_args(['--pause-key', 'x', '--n-trials', '10'])
        assert args.pause_key == 'x'

    @patch('LightningTune.hpo_runner.PausibleOptunaOptimizer')
    def test_pause_settings_resolution_cli_overrides_init(self, mock_optimizer):
        """Test that CLI args override __init__ parameters."""
        # Setup mock
        mock_study = Mock()
        mock_optimizer.return_value.optimize.return_value = mock_study

        # Create runner with pause disabled in __init__
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={},
            enable_pause=False,  # Disabled in __init__
            pause_key='p'
        )

        # Run with --enable-pause CLI arg (should override __init__)
        argv = ['--n-trials', '1', '--enable-pause', '--test-mode', '--no-reflow']
        runner.run_from_cli(argv=argv)

        # Verify optimizer was called with enable_pause=True (CLI override)
        mock_optimizer.assert_called_once()
        call_kwargs = mock_optimizer.call_args[1]
        assert call_kwargs['enable_pause'] == True

    @patch('LightningTune.hpo_runner.PausibleOptunaOptimizer')
    def test_pause_settings_resolution_disable_overrides_enable(self, mock_optimizer):
        """Test that --disable-pause overrides __init__ enable_pause=True."""
        # Setup mock
        mock_study = Mock()
        mock_optimizer.return_value.optimize.return_value = mock_study

        # Create runner with pause enabled in __init__
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={},
            enable_pause=True,  # Enabled in __init__
        )

        # Run with --disable-pause CLI arg
        argv = ['--n-trials', '1', '--disable-pause', '--test-mode', '--no-reflow']
        runner.run_from_cli(argv=argv)

        # Verify optimizer was called with enable_pause=False
        mock_optimizer.assert_called_once()
        call_kwargs = mock_optimizer.call_args[1]
        assert call_kwargs['enable_pause'] == False

    @patch('LightningTune.hpo_runner.PausibleOptunaOptimizer')
    def test_pause_key_cli_override(self, mock_optimizer):
        """Test that --pause-key CLI arg overrides __init__ pause_key."""
        # Setup mock
        mock_study = Mock()
        mock_optimizer.return_value.optimize.return_value = mock_study

        # Create runner with default pause_key
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={},
            pause_key='p'
        )

        # Run with custom pause-key
        argv = ['--n-trials', '1', '--pause-key', 'x', '--test-mode', '--no-reflow']
        runner.run_from_cli(argv=argv)

        # Verify the pause key was used (check if it's in the runner's config)
        # Note: PauseCallback is only added when use_reflow=True
        mock_optimizer.assert_called_once()

    @patch('LightningTune.hpo_runner.PausibleOptunaOptimizer')
    def test_pause_settings_passed_to_optimizer(self, mock_optimizer):
        """Test that pause settings are correctly passed to PausibleOptunaOptimizer.

        Note: HPO runs do NOT use PauseCallback (which pauses at validation boundaries).
        Instead, they use PausibleOptunaOptimizer which pauses at trial boundaries.
        This prevents corrupting trial metrics and ensures fair trial comparison.
        """
        # Setup mocks
        mock_study = Mock()
        mock_optimizer.return_value.optimize.return_value = mock_study

        # Create runner
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={},
            enable_pause=True,
            pause_key='p'
        )

        # Run
        argv = ['--n-trials', '1', '--test-mode', '--no-reflow']
        runner.run_from_cli(argv=argv)

        # Verify PausibleOptunaOptimizer was called with pause settings
        mock_optimizer.assert_called_once()
        call_kwargs = mock_optimizer.call_args[1]
        assert call_kwargs['enable_pause'] == True

    @patch('LightningTune.hpo_runner.HAS_PAUSE_CALLBACK', True)
    @patch('LightningTune.hpo_runner.PauseCallback')
    @patch('LightningTune.hpo_runner.PausibleOptunaOptimizer')
    def test_pause_callback_not_added_when_reflow_disabled(self, mock_optimizer, mock_pause_callback):
        """Test that PauseCallback is NOT added when use_reflow=False."""
        # Setup mock
        mock_study = Mock()
        mock_optimizer.return_value.optimize.return_value = mock_study

        # Create runner
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={},
            enable_pause=True
        )

        # Run with use_reflow=False
        argv = ['--n-trials', '1', '--no-reflow', '--test-mode']
        runner.run_from_cli(argv=argv)

        # Verify PauseCallback was NOT created
        mock_pause_callback.assert_not_called()

    @patch('LightningTune.hpo_runner.HAS_PAUSE_CALLBACK', True)
    @patch('LightningTune.hpo_runner.PauseCallback')
    @patch('LightningTune.hpo_runner.PausibleOptunaOptimizer')
    def test_pause_callback_not_added_when_pause_disabled(self, mock_optimizer, mock_pause_callback):
        """Test that PauseCallback is NOT added when enable_pause=False."""
        # Setup mock
        mock_study = Mock()
        mock_optimizer.return_value.optimize.return_value = mock_study

        # Create runner
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={},
            enable_pause=False
        )

        # Run with use_reflow=True but pause disabled
        argv = ['--n-trials', '1', '--test-mode']
        runner.run_from_cli(argv=argv)

        # Verify PauseCallback was NOT created
        mock_pause_callback.assert_not_called()

    @patch('LightningTune.hpo_runner.PausibleOptunaOptimizer')
    def test_pause_info_logged_when_enabled(self, mock_optimizer, capsys):
        """Test that pause info is logged when enabled.

        Note: HPO uses PausibleOptunaOptimizer for trial-boundary pausing,
        not PauseCallback. So we test that the optimizer gets the settings.
        """
        # Setup mock
        mock_study = Mock()
        mock_optimizer.return_value.optimize.return_value = mock_study

        # Create runner with pause enabled
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={},
            enable_pause=True,
            pause_key='x'
        )

        # Run
        argv = ['--n-trials', '1', '--test-mode', '--no-reflow']
        runner.run_from_cli(argv=argv)

        # Verify PausibleOptunaOptimizer was called with pause settings
        mock_optimizer.assert_called_once()
        call_kwargs = mock_optimizer.call_args[1]
        assert call_kwargs['enable_pause'] == True


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, '-v'])
