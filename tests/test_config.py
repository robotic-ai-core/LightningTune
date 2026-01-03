"""
Tests for HPOConfig dataclasses.
"""

import pytest
from pathlib import Path


class TestHPOPersistenceConfig:
    """Tests for HPOPersistenceConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        from LightningTune.config import HPOPersistenceConfig

        config = HPOPersistenceConfig()

        assert config.save_every_n_trials == 10
        assert config.local_checkpoint_dir is None
        assert config.wandb_project is None
        assert config.persist_args is True
        assert 'resume_from' in config.args_exclude
        assert 'study_name' in config.args_exclude

    def test_string_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        from LightningTune.config import HPOPersistenceConfig

        config = HPOPersistenceConfig(local_checkpoint_dir="/tmp/test")

        assert isinstance(config.local_checkpoint_dir, Path)
        assert config.local_checkpoint_dir == Path("/tmp/test")

    def test_list_to_set_conversion(self):
        """Test that list args_exclude is converted to set."""
        from LightningTune.config import HPOPersistenceConfig

        config = HPOPersistenceConfig(args_exclude=['arg1', 'arg2'])

        assert isinstance(config.args_exclude, set)
        assert 'arg1' in config.args_exclude
        assert 'arg2' in config.args_exclude


class TestHPOPauseConfig:
    """Tests for HPOPauseConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        from LightningTune.config import HPOPauseConfig

        config = HPOPauseConfig()

        assert config.enable_pause is True
        assert config.pause_key == 'p'
        assert config.restart_on_save is False
        assert config.restart_every_trial is True


class TestHPOSamplerConfig:
    """Tests for HPOSamplerConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        from LightningTune.config import HPOSamplerConfig

        config = HPOSamplerConfig()

        assert config.name == 'tpe'
        assert config.seed is None
        assert config.kwargs == {}

    def test_custom_values(self):
        """Test custom initialization."""
        from LightningTune.config import HPOSamplerConfig

        config = HPOSamplerConfig(
            name='random',
            seed=42,
            kwargs={'n_startup_trials': 10},
        )

        assert config.name == 'random'
        assert config.seed == 42
        assert config.kwargs['n_startup_trials'] == 10


class TestHPOPrunerConfig:
    """Tests for HPOPrunerConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        from LightningTune.config import HPOPrunerConfig

        config = HPOPrunerConfig()

        assert config.name == 'median'
        assert config.kwargs == {}


class TestHPOTrialConfig:
    """Tests for HPOTrialConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        from LightningTune.config import HPOTrialConfig

        config = HPOTrialConfig()

        assert config.n_trials == 50
        assert config.trial_steps is None
        assert config.metric == 'val_loss'
        assert config.direction == 'minimize'
        assert config.checkpoint_top_k == 0


class TestHPOConfig:
    """Tests for HPOConfig main dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        from LightningTune.config import HPOConfig

        config = HPOConfig()

        assert config.study_name == 'optuna_study'
        assert config.persistence.save_every_n_trials == 10
        assert config.pause.enable_pause is True
        assert config.sampler.name == 'tpe'
        assert config.pruner.name == 'median'
        assert config.trial.n_trials == 50

    def test_from_flat_params(self):
        """Test creating from flat parameters."""
        from LightningTune.config import HPOConfig

        config = HPOConfig.from_flat_params(
            study_name='test_study',
            save_every_n_trials=5,
            wandb_project='my-project',
            enable_pause=False,
            sampler_name='random',
            pruner_name='hyperband',
            n_trials=100,
        )

        assert config.study_name == 'test_study'
        assert config.persistence.save_every_n_trials == 5
        assert config.persistence.wandb_project == 'my-project'
        assert config.pause.enable_pause is False
        assert config.sampler.name == 'random'
        assert config.pruner.name == 'hyperband'
        assert config.trial.n_trials == 100

    def test_to_flat_dict(self):
        """Test conversion to flat dictionary."""
        from LightningTune.config import HPOConfig

        config = HPOConfig(
            study_name='test_study',
        )

        flat = config.to_flat_dict()

        assert flat['study_name'] == 'test_study'
        assert flat['save_every_n_trials'] == 10
        assert flat['enable_pause'] is True
        assert flat['sampler_name'] == 'tpe'
        assert flat['pruner_name'] == 'median'
        assert flat['n_trials'] == 50

    def test_nested_config_initialization(self):
        """Test initialization with nested config objects."""
        from LightningTune.config import (
            HPOConfig,
            HPOPersistenceConfig,
            HPOPauseConfig,
        )

        config = HPOConfig(
            study_name='nested_test',
            persistence=HPOPersistenceConfig(
                save_every_n_trials=20,
                wandb_project='nested-project',
            ),
            pause=HPOPauseConfig(
                enable_pause=False,
            ),
        )

        assert config.study_name == 'nested_test'
        assert config.persistence.save_every_n_trials == 20
        assert config.persistence.wandb_project == 'nested-project'
        assert config.pause.enable_pause is False


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
