#!/usr/bin/env python
"""Test cases for Lightning CLI-style dot-notation argument parsing."""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, Mock

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


class TestDotNotationArgumentParsing:
    """Test Lightning CLI-style dot-notation argument parsing."""

    def test_parse_dot_notation_args_basic(self):
        """Test basic dot-notation argument parsing."""
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        unknown_args = ['--data.batch_size', '512', '--model.learning_rate', '1e-4']
        result = runner._parse_dot_notation_args(unknown_args)

        assert 'data.batch_size' in result
        assert result['data.batch_size'] == 512  # Should be int
        assert 'model.learning_rate' in result
        assert result['model.learning_rate'] == 1e-4  # Should be float

    def test_parse_dot_notation_args_multiple_levels(self):
        """Test parsing nested dot-notation arguments."""
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        unknown_args = [
            '--model.init_args.learning_rate', '0.001',
            '--trainer.logger.name', 'my_experiment'
        ]
        result = runner._parse_dot_notation_args(unknown_args)

        assert result['model.init_args.learning_rate'] == 0.001
        assert result['trainer.logger.name'] == 'my_experiment'

    def test_parse_dot_notation_args_mixed_types(self):
        """Test parsing arguments with different types."""
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        unknown_args = [
            '--data.batch_size', '256',  # int
            '--model.learning_rate', '1e-4',  # float (scientific notation)
            '--trainer.enable_checkpointing', 'false',  # bool
            '--trainer.log_dir', 'logs/experiment',  # string
        ]
        result = runner._parse_dot_notation_args(unknown_args)

        assert result['data.batch_size'] == 256
        assert isinstance(result['data.batch_size'], int)

        assert result['model.learning_rate'] == 1e-4
        assert isinstance(result['model.learning_rate'], float)

        assert result['trainer.enable_checkpointing'] == False
        assert isinstance(result['trainer.enable_checkpointing'], bool)

        assert result['trainer.log_dir'] == 'logs/experiment'
        assert isinstance(result['trainer.log_dir'], str)

    def test_parse_dot_notation_args_flag_without_value(self):
        """Test parsing flag arguments without values."""
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        unknown_args = ['--trainer.fast_dev_run', '--data.batch_size', '128']
        result = runner._parse_dot_notation_args(unknown_args)

        assert result['trainer.fast_dev_run'] == True  # Flag without value = True
        assert result['data.batch_size'] == 128

    def test_parse_dot_notation_args_empty(self):
        """Test parsing empty unknown args."""
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        result = runner._parse_dot_notation_args([])
        assert result == {}

    def test_parse_dot_notation_args_non_config_args_ignored(self):
        """Test that non-config arguments are ignored."""
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        unknown_args = [
            '--data.batch_size', '512',
            '--some-flag',  # No dot, should be ignored
            '--model.learning_rate', '1e-4'
        ]
        result = runner._parse_dot_notation_args(unknown_args)

        # Only dot-notation args should be captured
        assert 'data.batch_size' in result
        assert 'model.learning_rate' in result
        assert 'some-flag' not in result
        assert 'some_flag' not in result

    def test_auto_convert_type_int(self):
        """Test automatic type conversion for integers."""
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        assert runner._auto_convert_type('123') == 123
        assert isinstance(runner._auto_convert_type('123'), int)

        assert runner._auto_convert_type('0') == 0
        assert runner._auto_convert_type('-42') == -42

    def test_auto_convert_type_float(self):
        """Test automatic type conversion for floats."""
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        assert runner._auto_convert_type('3.14') == 3.14
        assert isinstance(runner._auto_convert_type('3.14'), float)

        assert runner._auto_convert_type('1e-4') == 1e-4
        assert runner._auto_convert_type('2.5e3') == 2500.0
        assert runner._auto_convert_type('-0.001') == -0.001

    def test_auto_convert_type_bool(self):
        """Test automatic type conversion for booleans."""
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        assert runner._auto_convert_type('true') == True
        assert runner._auto_convert_type('True') == True
        assert runner._auto_convert_type('TRUE') == True

        assert runner._auto_convert_type('false') == False
        assert runner._auto_convert_type('False') == False
        assert runner._auto_convert_type('FALSE') == False

    def test_auto_convert_type_string(self):
        """Test automatic type conversion for strings."""
        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        assert runner._auto_convert_type('hello') == 'hello'
        assert isinstance(runner._auto_convert_type('hello'), str)

        assert runner._auto_convert_type('logs/experiment') == 'logs/experiment'
        assert runner._auto_convert_type('some-flag-value') == 'some-flag-value'

    @patch('LightningTune.hpo_runner.PausibleOptunaOptimizer')
    def test_dot_notation_args_in_run_from_cli(self, mock_optimizer):
        """Test that dot-notation args are integrated in run_from_cli()."""
        # Setup mock
        mock_study = Mock()
        mock_optimizer.return_value.optimize.return_value = mock_study

        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        # Run with dot-notation arguments
        argv = [
            '--n-trials', '1',
            '--test-mode',
            '--no-reflow',
            '--data.batch_size', '512',
            '--model.learning_rate', '1e-4'
        ]

        runner.run_from_cli(argv=argv)

        # Verify config overrides include dot-notation args
        assert 'data.batch_size' in runner.config_overrides
        assert runner.config_overrides['data.batch_size'] == 512

        assert 'model.learning_rate' in runner.config_overrides
        assert runner.config_overrides['model.learning_rate'] == 1e-4

    @patch('LightningTune.hpo_runner.PausibleOptunaOptimizer')
    def test_dot_notation_args_passed_to_optimizer(self, mock_optimizer):
        """Test that dot-notation args are passed to optimizer via config_overrides."""
        # Setup mock
        mock_study = Mock()
        mock_optimizer.return_value.optimize.return_value = mock_study

        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        # Run with dot-notation arguments
        # Note: Don't use --test-mode to avoid env-specific overrides
        argv = [
            '--n-trials', '1',
            '--no-reflow',
            '--data.num_workers', '8'  # Use a key that won't be overridden
        ]

        runner.run_from_cli(argv=argv)

        # Verify optimize was called with config_overrides containing dot-notation args
        call_kwargs = mock_optimizer.return_value.optimize.call_args[1]
        assert 'config_overrides' in call_kwargs
        assert 'data.num_workers' in call_kwargs['config_overrides']
        assert call_kwargs['config_overrides']['data.num_workers'] == 8

    @patch('LightningTune.hpo_runner.PausibleOptunaOptimizer')
    def test_dot_notation_args_dont_override_explicit_config(self, mock_optimizer):
        """Test that dot-notation args don't override explicit config overrides."""
        # Setup mock
        mock_study = Mock()
        mock_optimizer.return_value.optimize.return_value = mock_study

        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        # The test is about priority: explicit config overrides > dot-notation args
        # In current implementation, dot-notation args are added only if key not in config_overrides
        # This test verifies that behavior

        # Use --test-mode which sets trainer.limit_train_batches in _build_config_overrides
        argv = ['--n-trials', '1', '--test-mode', '--no-reflow', '--trainer.limit_train_batches', '999']
        runner.run_from_cli(argv=argv)

        # test_mode adds trainer.limit_train_batches to config via _build_config_overrides
        # Then environment-specific code further modifies it to 1
        # dot-notation arg tries to set it to 999
        # The final value is from test_mode environment overrides (1), not dot-notation (999)
        assert runner.config_overrides['trainer.limit_train_batches'] == 1  # From test_mode env, not 999

    @patch('LightningTune.hpo_runner.PausibleOptunaOptimizer')
    def test_dot_notation_logging(self, mock_optimizer, caplog):
        """Test that dot-notation args are logged."""
        # Setup mock
        mock_study = Mock()
        mock_optimizer.return_value.optimize.return_value = mock_study

        runner = HPORunner(
            model_class=DummyModel,
            datamodule_class=None,
            search_space=lambda trial: {},
            base_config={}
        )

        import logging
        with caplog.at_level(logging.INFO):
            argv = [
                '--n-trials', '1',
                '--test-mode',
                '--no-reflow',
                '--data.batch_size', '512'
            ]
            runner.run_from_cli(argv=argv)

        # Check that dot-notation args were logged
        log_messages = [record.message for record in caplog.records]
        assert any('Parsed' in msg and 'dot-notation' in msg for msg in log_messages)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, '-v'])
