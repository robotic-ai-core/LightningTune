#!/usr/bin/env python
"""Test cases for --config CLI argument in HPORunner."""

import pytest
import sys
import os
import tempfile
from unittest.mock import Mock, patch
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


class TestConfigCLIArgument:
    """Test --config CLI argument integration."""

    def test_config_arg_defined(self):
        """Test that --config is defined in CLI arguments."""
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        # Check that config CLI arg exists
        assert 'config' in runner.cli_args
        assert runner.cli_args['config']['type'] == str
        assert runner.cli_args['config']['default'] is None

    def test_config_arg_parsing(self):
        """Test that --config CLI arg is properly parsed."""
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        # Parse with --config
        parser = runner._create_parser()
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp:
            tmp.write(b"model:\n  init_args:\n    param: value\n")
            tmp.flush()
            config_path = tmp.name

        try:
            args = parser.parse_args(['--config', config_path, '--n-trials', '10'])
            assert args.config == config_path
            assert args.n_trials == 10
        finally:
            os.unlink(config_path)

    @patch('LightningTune.hpo_runner.PausibleOptunaOptimizer')
    def test_config_cli_overrides_init(self, mock_optimizer):
        """Test that --config CLI arg overrides __init__ base_config."""
        # Setup mock
        mock_study = Mock()
        mock_optimizer.return_value.optimize.return_value = mock_study

        # Create temporary config files
        with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as init_config:
            init_config.write("model:\n  init_args:\n    learning_rate: 1e-3\n")
            init_config.flush()
            init_config_path = init_config.name

        with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as cli_config:
            cli_config.write("model:\n  init_args:\n    learning_rate: 1e-4\n")
            cli_config.flush()
            cli_config_path = cli_config.name

        try:
            # Create runner with base_config from __init__
            runner = HPORunner(
                model_class=DummyModel,
                datamodule_class=None,
                search_space=lambda trial: {},
                base_config=init_config_path  # From __init__
            )

            # Run with --config CLI arg (should override __init__)
            argv = [
                '--config', cli_config_path,  # CLI override
                '--n-trials', '1',
                '--test-mode',
                '--no-reflow'
            ]
            runner.run_from_cli(argv=argv)

            # Verify base_config was overridden by CLI
            assert runner.base_config == cli_config_path
        finally:
            os.unlink(init_config_path)
            os.unlink(cli_config_path)

    @patch('LightningTune.hpo_runner.PausibleOptunaOptimizer')
    def test_config_validation_file_not_found(self, mock_optimizer):
        """Test that --config validates file existence."""
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        # Run with non-existent config file (should exit with error)
        argv = [
            '--config', '/nonexistent/config.yaml',
            '--n-trials', '1',
            '--test-mode',
            '--no-reflow'
        ]

        with pytest.raises(SystemExit) as exc_info:
            runner.run_from_cli(argv=argv)

        assert exc_info.value.code == 1

    @patch('LightningTune.hpo_runner.PausibleOptunaOptimizer')
    def test_config_without_cli_arg_uses_init(self, mock_optimizer):
        """Test that without --config CLI arg, __init__ base_config is used."""
        # Setup mock
        mock_study = Mock()
        mock_optimizer.return_value.optimize.return_value = mock_study

        # Create temporary config file
        with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as config:
            config.write("model:\n  init_args:\n    learning_rate: 1e-3\n")
            config.flush()
            config_path = config.name

        try:
            # Create runner with base_config from __init__
            runner = HPORunner(
                model_class=DummyModel,
                datamodule_class=None,
                search_space=lambda trial: {},
                base_config=config_path  # From __init__
            )

            # Run without --config CLI arg
            argv = ['--n-trials', '1', '--test-mode', '--no-reflow']
            runner.run_from_cli(argv=argv)

            # Verify base_config from __init__ is still used
            assert runner.base_config == config_path
        finally:
            os.unlink(config_path)

    @patch('LightningTune.hpo_runner.PausibleOptunaOptimizer')
    def test_config_with_dot_notation_args(self, mock_optimizer):
        """Test that --config works together with dot-notation arguments."""
        # Setup mock
        mock_study = Mock()
        mock_optimizer.return_value.optimize.return_value = mock_study

        # Create temporary config file
        with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as config:
            config.write("model:\n  init_args:\n    learning_rate: 1e-3\n")
            config.flush()
            config_path = config.name

        try:
            runner = HPORunner(
                model_class=DummyModel,
                datamodule_class=None,
                search_space=lambda trial: {},
                base_config={}
            )

            # Run with both --config and dot-notation args
            argv = [
                '--config', config_path,
                '--data.batch_size', '512',
                '--model.learning_rate', '1e-4',
                '--n-trials', '1',
                '--test-mode',
                '--no-reflow'
            ]
            runner.run_from_cli(argv=argv)

            # Verify both config and dot-notation args are applied
            assert runner.base_config == config_path
            assert 'data.batch_size' in runner.config_overrides
            assert runner.config_overrides['data.batch_size'] == 512
        finally:
            os.unlink(config_path)

    @patch('LightningTune.hpo_runner.PausibleOptunaOptimizer')
    def test_config_logging(self, mock_optimizer, caplog):
        """Test that --config CLI arg is logged."""
        # Setup mock
        mock_study = Mock()
        mock_optimizer.return_value.optimize.return_value = mock_study

        # Create temporary config file
        with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as config:
            config.write("model:\n  init_args:\n    param: value\n")
            config.flush()
            config_path = config.name

        try:
            runner = HPORunner(
                model_class=DummyModel,
                datamodule_class=None,
                search_space=lambda trial: {},
                base_config={}
            )

            import logging
            with caplog.at_level(logging.INFO):
                argv = [
                    '--config', config_path,
                    '--n-trials', '1',
                    '--test-mode',
                    '--no-reflow'
                ]
                runner.run_from_cli(argv=argv)

            # Check that config was logged
            log_messages = [record.message for record in caplog.records]
            assert any('Using config from CLI' in msg and config_path in msg for msg in log_messages)
        finally:
            os.unlink(config_path)

    def test_config_cli_help_text(self):
        """Test that --config has proper help text."""
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        parser = runner._create_parser()

        # Get help text
        import io
        from contextlib import redirect_stdout

        help_output = io.StringIO()
        with redirect_stdout(help_output):
            try:
                parser.parse_args(['--help'])
            except SystemExit:
                pass

        help_text = help_output.getvalue()
        assert '--config' in help_text
        assert 'YAML' in help_text or 'configuration' in help_text


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, '-v'])
