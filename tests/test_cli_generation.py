"""
Unit tests for CLI generation utilities.
"""

import pytest
from LightningTune.utils.cli_generation import (
    validate_config_for_cli_generation,
    extract_cli_args_from_config,
    format_cli_command,
    describe_search_space,
    format_best_trial_results,
)


class TestValidateConfigForCLIGeneration:
    """Tests for validate_config_for_cli_generation function."""

    def test_valid_config(self):
        """Test with valid configuration."""
        config = {"model.init_args.learning_rate": 1e-5}
        validate_config_for_cli_generation(config)  # Should not raise

    def test_empty_config_raises(self):
        """Test that empty config raises ValueError."""
        with pytest.raises(ValueError, match="no parameters"):
            validate_config_for_cli_generation({})

    def test_only_hparams_raises(self):
        """Test that config with only hparams raises ValueError."""
        config = {"hparams.trial_number": 42}
        with pytest.raises(ValueError, match="no parameters"):
            validate_config_for_cli_generation(config)

    def test_non_dict_raises(self):
        """Test that non-dict raises ValueError."""
        with pytest.raises(ValueError, match="must be a dict"):
            validate_config_for_cli_generation([1, 2, 3])

    def test_custom_exclude_prefixes(self):
        """Test with custom exclude prefixes."""
        config = {"custom.param": 1}
        # Should raise when we exclude the custom prefix
        with pytest.raises(ValueError, match="no parameters"):
            validate_config_for_cli_generation(config, exclude_prefixes=("custom.",))


class TestExtractCLIArgsFromConfig:
    """Tests for extract_cli_args_from_config function."""

    def test_basic_extraction(self):
        """Test basic parameter extraction."""
        config = {
            "model.init_args.learning_rate": 1e-5,
            "model.init_args.weight_decay": 0.1,
        }
        args = extract_cli_args_from_config(config)

        assert "--model.init_args.learning_rate" in args
        assert "--model.init_args.weight_decay" in args
        assert "1e-05" in args or "0.00001" in args
        assert "0.1" in args

    def test_with_base_config(self):
        """Test with base config path."""
        config = {"model.init_args.learning_rate": 1e-5}
        args = extract_cli_args_from_config(
            config,
            base_config_path="configs/model.yaml"
        )

        assert args[0] == "--config"
        assert args[1] == "configs/model.yaml"

    def test_with_extra_args(self):
        """Test with extra arguments."""
        config = {"model.init_args.learning_rate": 1e-5}
        args = extract_cli_args_from_config(
            config,
            extra_args={"trainer.max_epochs": 2000}
        )

        assert "--trainer.max_epochs" in args
        assert "2000" in args

    def test_excludes_hparams(self):
        """Test that hparams are excluded by default."""
        config = {
            "model.init_args.learning_rate": 1e-5,
            "hparams.trial_number": 42,
        }
        args = extract_cli_args_from_config(config)

        assert "--hparams.trial_number" not in args
        assert "--model.init_args.learning_rate" in args

    def test_excludes_specified_params(self):
        """Test that excluded params are skipped."""
        config = {
            "model.init_args.learning_rate": 1e-5,
            "data.init_args.batch_size": 128,
        }
        args = extract_cli_args_from_config(
            config,
            excluded_params={"data.init_args.batch_size"}
        )

        assert "--data.init_args.batch_size" not in args
        assert "--model.init_args.learning_rate" in args

    def test_boolean_formatting(self):
        """Test boolean value formatting."""
        config = {"trainer.enable_progress_bar": True}
        args = extract_cli_args_from_config(config)

        assert "true" in args

    def test_int_formatting(self):
        """Test integer value formatting."""
        config = {"trainer.max_epochs": 100}
        args = extract_cli_args_from_config(config)

        assert "100" in args

    def test_float_formatting(self):
        """Test float value formatting (avoids scientific notation issues)."""
        config = {"model.init_args.learning_rate": 0.0001}
        args = extract_cli_args_from_config(config)

        # Should use general format
        assert any("0.0001" in arg or "1e-04" in arg for arg in args)

    def test_string_value(self):
        """Test string value handling."""
        config = {"model.init_args.loss_type": "l1"}
        args = extract_cli_args_from_config(config)

        assert "--model.init_args.loss_type" in args
        assert "l1" in args

    def test_skip_objects(self):
        """Test that objects are skipped when skip_objects=True."""
        class DummyModel:
            pass

        config = {
            "model.init_args.learning_rate": 1e-5,
            "model.init_args.dynamics_model": DummyModel(),
        }
        args = extract_cli_args_from_config(config, skip_objects=True)

        # Should not include the model object
        assert "--model.init_args.dynamics_model" not in args
        assert "--model.init_args.learning_rate" in args


class TestFormatCLICommand:
    """Tests for format_cli_command function."""

    def test_basic_formatting(self):
        """Test basic command formatting."""
        args = ["--config", "config.yaml", "--model.lr", "1e-5"]
        cmd = format_cli_command(args)

        assert "python train.py fit" in cmd
        assert "--config config.yaml" in cmd
        assert "--model.lr 1e-5" in cmd

    def test_custom_script(self):
        """Test with custom script."""
        args = ["--config", "config.yaml"]
        cmd = format_cli_command(args, script="python my_script.py fit")

        assert "python my_script.py fit" in cmd

    def test_line_continuation(self):
        """Test line continuation format."""
        args = ["--config", "config.yaml", "--model.lr", "1e-5"]
        cmd = format_cli_command(args)

        # Should have line continuations
        assert "\\" in cmd

    def test_boolean_flag(self):
        """Test boolean flag without value."""
        args = ["--verbose", "--config", "config.yaml"]
        cmd = format_cli_command(args)

        assert "--verbose" in cmd

    def test_custom_line_continuation(self):
        """Test custom line continuation string."""
        args = ["--config", "config.yaml"]
        cmd = format_cli_command(args, line_continuation=" \\\n    ")

        assert "\\\n    " in cmd


class TestDescribeSearchSpace:
    """Tests for describe_search_space function."""

    def test_basic_description(self):
        """Test basic search space description."""
        def search_space(trial):
            lr = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
            return {"lr": lr}

        desc = describe_search_space(search_space)

        assert "learning_rate" in desc
        assert "1e-6" in desc or "1e-06" in desc

    def test_categorical_params(self):
        """Test categorical parameter description."""
        def search_space(trial):
            loss = trial.suggest_categorical("loss_type", ["l1", "l2"])
            return {"loss": loss}

        desc = describe_search_space(search_space)

        assert "loss_type" in desc

    def test_int_params(self):
        """Test integer parameter description."""
        def search_space(trial):
            layers = trial.suggest_int("num_layers", 2, 10)
            return {"layers": layers}

        desc = describe_search_space(search_space)

        assert "num_layers" in desc

    def test_custom_header(self):
        """Test custom header."""
        def search_space(trial):
            return {}

        desc = describe_search_space(search_space, header="Custom Header:")

        assert "Custom Header:" in desc


class TestFormatBestTrialResults:
    """Tests for format_best_trial_results function."""

    def test_basic_formatting(self):
        """Test basic result formatting."""
        # Create a mock study
        class MockTrial:
            number = 42
            params = {"learning_rate": 1e-5, "weight_decay": 0.1}

        class MockStudy:
            best_trial = MockTrial()
            best_value = 0.001234
            best_params = MockTrial.params

        study = MockStudy()
        result = format_best_trial_results(study)

        assert "OPTIMIZATION COMPLETE" in result
        assert "#42" in result
        assert "0.001234" in result
        assert "learning_rate" in result
        assert "weight_decay" in result

    def test_custom_header(self):
        """Test custom header."""
        class MockTrial:
            number = 1
            params = {}

        class MockStudy:
            best_trial = MockTrial()
            best_value = 0.0
            best_params = {}

        study = MockStudy()
        result = format_best_trial_results(study, header="DONE!")

        assert "DONE!" in result

    def test_custom_width(self):
        """Test custom separator width."""
        class MockTrial:
            number = 1
            params = {}

        class MockStudy:
            best_trial = MockTrial()
            best_value = 0.0
            best_params = {}

        study = MockStudy()
        result = format_best_trial_results(study, width=50)

        # Should have separators of length 50
        assert "=" * 50 in result


class TestIntegration:
    """Integration tests for CLI generation workflow."""

    def test_full_workflow(self):
        """Test full workflow from config to formatted command."""
        config = {
            "model.init_args.learning_rate": 1e-5,
            "model.init_args.weight_decay": 0.1,
            "model.init_args.loss_type": "l1",
            "hparams.trial_number": 42,  # Should be excluded
        }

        # Extract CLI args
        args = extract_cli_args_from_config(
            config,
            base_config_path="configs/model.yaml",
            extra_args={"trainer.max_epochs": 2000},
            excluded_params=set(),
        )

        # Format command
        cmd = format_cli_command(
            args,
            script="python scripts/train.py fit"
        )

        # Verify command structure
        assert "python scripts/train.py fit" in cmd
        assert "--config configs/model.yaml" in cmd
        assert "--model.init_args.learning_rate" in cmd
        assert "--model.init_args.weight_decay" in cmd
        assert "--model.init_args.loss_type l1" in cmd
        assert "--trainer.max_epochs 2000" in cmd
        assert "hparams" not in cmd

    def test_empty_extra_args(self):
        """Test with empty extra_args."""
        config = {"model.init_args.learning_rate": 1e-5}
        args = extract_cli_args_from_config(config, extra_args={})

        assert len(args) == 2  # Just key and value
