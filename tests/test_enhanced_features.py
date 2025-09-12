"""Tests for enhanced PausibleOptunaOptimizer features."""

import pytest
import pickle
import optuna
from pathlib import Path
from unittest.mock import MagicMock, patch
import logging
import tempfile

from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer
from LightningTune.utils.param_utils import simplify_param_names, ParamNameSimplifier
from LightningTune.utils.torch_compile import get_compile_settings_for_mode


class TestAutoArgumentPersistence:
    """Test automatic argument persistence feature."""
    
    def test_args_automatically_persisted(self, tmp_path):
        """Test that args are automatically added to config_overrides."""
        
        # Create mock args object
        class Args:
            n_trials = 100
            trial_steps = 40000
            val_interval = 500
            test_mode = True
            resume_from = "latest"  # Should be excluded
            study_name = "test"  # Should be excluded
            wandb = "project"
            patience = 10
            compile_mode = "safe"
        
        args = Args()
        
        optimizer = PausibleOptunaOptimizer(
            base_config={"test": "config"},
            search_space=lambda trial: {"lr": trial.suggest_float("lr", 1e-4, 1e-2)},
            model_class=MagicMock,
            study_name="test_study",
            persist_args=True,
            args=args,
            args_exclude={'resume_from', 'study_name'},
            local_checkpoint_dir=tmp_path / "checkpoints"
        )
        
        # Mock the optimize method to check config_overrides
        with patch.object(optimizer, '_create_underlying_optimizer') as mock_create:
            mock_optimizer = MagicMock()
            mock_study = optuna.create_study()
            mock_optimizer.optimize.return_value = mock_study
            mock_create.return_value = mock_optimizer
            
            # Call optimize
            optimizer.optimize(n_trials=10)
            
            # Check that config_overrides were built from args
            assert optimizer.persistent_config_overrides is not None
            assert "args.n_trials" in optimizer.persistent_config_overrides
            assert optimizer.persistent_config_overrides["args.n_trials"] == 100
            assert "args.trial_steps" in optimizer.persistent_config_overrides
            assert optimizer.persistent_config_overrides["args.trial_steps"] == 40000
            assert "args.test_mode" in optimizer.persistent_config_overrides
            
            # Excluded args should not be present
            assert "args.resume_from" not in optimizer.persistent_config_overrides
            assert "args.study_name" not in optimizer.persistent_config_overrides
    
    def test_args_restored_on_resume(self, tmp_path):
        """Test that args are restored from saved session on resume."""
        
        # Create initial args with specific values
        class Args:
            trial_steps = 5000  # Default value
            n_trials = 50  # Default value
            val_interval = None
            test_mode = False
            wandb = None
            study_name = "test_study"
        
        args = Args()
        
        # Create saved session with different values
        saved_overrides = {
            "args.trial_steps": 40000,  # Different from default
            "args.n_trials": 100,  # Different from default
            "args.val_interval": 250,
            "args.test_mode": True,
            "trainer.val_check_interval": 250,
        }
        
        study = optuna.create_study()
        study.optimize(lambda trial: 0.5, n_trials=2)
        
        session_info = {
            "study": study,
            "total_trials_completed": 2,
            "sampler_name": "tpe",
            "pruner_name": "median",
            "study_name": "test_study",
            "config_overrides": saved_overrides
        }
        
        # Save session
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / "study.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(session_info, f)
        
        # Create optimizer with persist_args enabled
        optimizer = PausibleOptunaOptimizer(
            base_config={"test": "config"},
            search_space=lambda trial: {"lr": trial.suggest_float("lr", 1e-4, 1e-2)},
            model_class=MagicMock,
            study_name="test_study",
            persist_args=True,
            args=args,
            local_checkpoint_dir=checkpoint_dir
        )
        
        # Resume and check args are restored
        with patch('LightningTune.optuna.optimizer_reflow.ReflowOptunaDrivenOptimizer') as mock_reflow:
            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = study
            mock_reflow.return_value = mock_optimizer
            
            optimizer.optimize(n_trials=10, resume_from=str(checkpoint_dir))
            
            # Args should be restored to saved values
            assert args.trial_steps == 40000, f"Expected 40000, got {args.trial_steps}"
            assert args.n_trials == 100, f"Expected 100, got {args.n_trials}"
            assert args.val_interval == 250, f"Expected 250, got {args.val_interval}"
            assert args.test_mode == True, f"Expected True, got {args.test_mode}"


class TestConfigLayering:
    """Test built-in config layering support."""
    
    def test_override_config_merged_with_base(self, tmp_path):
        """Test that override config is automatically merged with base config."""
        
        # Create base config
        base_config = {
            "model": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "hidden_dim": 256
            },
            "trainer": {
                "max_epochs": 100
            }
        }
        
        # Create override config
        override_config = {
            "model": {
                "learning_rate": 0.01,  # Override
                "dropout": 0.5  # New key
            },
            "trainer": {
                "max_epochs": 50  # Override
            }
        }
        
        optimizer = PausibleOptunaOptimizer(
            base_config=base_config,
            override_config=override_config,
            search_space=lambda trial: {},
            model_class=MagicMock,
            study_name="test_study"
        )
        
        # Check merged config
        assert optimizer.base_config["model"]["learning_rate"] == 0.01  # Overridden
        assert optimizer.base_config["model"]["batch_size"] == 32  # Kept from base
        assert optimizer.base_config["model"]["hidden_dim"] == 256  # Kept from base
        assert optimizer.base_config["model"]["dropout"] == 0.5  # Added from override
        assert optimizer.base_config["trainer"]["max_epochs"] == 50  # Overridden


class TestParamSimplification:
    """Test parameter name simplification."""
    
    def test_simplify_param_names_function(self):
        """Test the simplify_param_names function."""
        
        params = {
            "model.init_args.learning_rate": 0.001,
            "model.init_args.transformer_hparams.num_layers": 12,
            "data.init_args.batch_size": 32,
            "trainer.max_epochs": 100,
            "model.init_args.adapter.init_args.hidden_dim": 256
        }
        
        simplified = simplify_param_names(params)
        
        assert simplified["learning_rate"] == 0.001
        assert simplified["transformer.num_layers"] == 12
        assert simplified["batch_size"] == 32
        assert simplified["max_epochs"] == 100
        assert simplified["adapter.hidden_dim"] == 256
    
    def test_param_name_simplifier_class(self):
        """Test the ParamNameSimplifier class with custom rules."""
        
        # Custom rules
        rules = [
            (r'\.init_args\.', '.'),
            (r'^model\.', 'mdl.'),  # Custom prefix
            (r'_hparams', ''),
        ]
        
        simplifier = ParamNameSimplifier(rules)
        
        params = {
            "model.init_args.learning_rate": 0.001,
            "model.transformer_hparams.layers": 12,
            "data.batch_size": 32
        }
        
        simplified = simplifier.simplify(params)
        
        assert simplified["mdl.learning_rate"] == 0.001
        assert simplified["mdl.transformer.layers"] == 12
        assert simplified["data.batch_size"] == 32


class TestCompileModes:
    """Test simplified compile mode settings."""
    
    def test_compile_mode_off(self):
        """Test 'off' compile mode."""
        settings = get_compile_settings_for_mode("off")
        assert settings["enabled"] is False
    
    def test_compile_mode_safe(self):
        """Test 'safe' compile mode."""
        settings = get_compile_settings_for_mode("safe")
        assert settings["enabled"] is True
        assert settings["backend"] == "inductor"
        assert "mode" not in settings  # Safe mode uses defaults
    
    def test_compile_mode_aggressive(self):
        """Test 'aggressive' compile mode."""
        settings = get_compile_settings_for_mode("aggressive")
        assert settings["enabled"] is True
        assert settings["backend"] == "inductor"
        assert settings["mode"] == "max-autotune"
        assert settings["options"]["triton.cudagraphs"] is True
    
    def test_invalid_compile_mode(self):
        """Test invalid compile mode raises error."""
        with pytest.raises(ValueError, match="Unknown compile mode"):
            get_compile_settings_for_mode("invalid")
    
    def test_compile_mode_integration(self, tmp_path):
        """Test compile mode integration in optimizer."""
        
        optimizer = PausibleOptunaOptimizer(
            base_config={"test": "config"},
            search_space=lambda trial: {},
            model_class=MagicMock,
            study_name="test_study",
            compile_mode="safe",
            local_checkpoint_dir=tmp_path / "checkpoints"
        )
        
        # Mock optimize to check config_overrides
        with patch('LightningTune.optuna.optimizer_reflow.ReflowOptunaDrivenOptimizer') as mock_reflow:
            mock_optimizer = MagicMock()
            mock_study = optuna.create_study()
            mock_optimizer.optimize.return_value = mock_study
            mock_reflow.return_value = mock_optimizer
            
            optimizer.optimize(n_trials=1)
            
            # Check compile settings were added to config
            assert "model.init_args.torch_compile_settings" in optimizer.persistent_config_overrides
            settings = optimizer.persistent_config_overrides["model.init_args.torch_compile_settings"]
            assert settings["enabled"] is True
            assert settings["backend"] == "inductor"


class TestStaticSessionLoader:
    """Test static session loader method."""
    
    def test_load_saved_session_from_local(self, tmp_path):
        """Test loading saved session from local file."""
        
        # Create a saved session
        study = optuna.create_study()
        study.optimize(lambda trial: 0.5, n_trials=2)
        
        session_info = {
            "study": study,
            "total_trials_completed": 2,
            "sampler_name": "tpe",
            "pruner_name": "median",
            "study_name": "test_study",
            "config_overrides": {"test": "value"}
        }
        
        # Save to file
        checkpoint_file = tmp_path / "study.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(session_info, f)
        
        # Load using static method
        loaded = PausibleOptunaOptimizer.load_saved_session(
            resume_from=str(checkpoint_file)
        )
        
        assert loaded is not None
        assert loaded["total_trials_completed"] == 2
        assert loaded["study_name"] == "test_study"
        assert loaded["config_overrides"]["test"] == "value"
    
    def test_load_saved_session_not_found(self):
        """Test loading non-existent session returns None."""
        
        loaded = PausibleOptunaOptimizer.load_saved_session(
            resume_from="/nonexistent/path"
        )
        
        assert loaded is None


class TestIntegration:
    """Integration tests for all enhanced features together."""
    
    def test_full_enhanced_workflow(self, tmp_path, caplog):
        """Test complete workflow with all enhanced features."""
        
        with caplog.at_level(logging.INFO):
            # Create args
            class Args:
                n_trials = 10
                trial_steps = 1000
                compile_mode = "safe"
                test_mode = True
                wandb = None
                study_name = "test_study"
            
            args = Args()
            
            # Create optimizer with all features
            optimizer = PausibleOptunaOptimizer(
                base_config={"base": "config"},
                override_config={"override": "config"},
                search_space=lambda trial: {"lr": trial.suggest_float("lr", 1e-4, 1e-2)},
                model_class=MagicMock,
                study_name="test_study",
                persist_args=True,
                args=args,
                simplify_param_names=True,
                compile_mode="safe",
                local_checkpoint_dir=tmp_path / "checkpoints"
            )
            
            # Mock the actual optimization
            with patch('LightningTune.optuna.optimizer_reflow.ReflowOptunaDrivenOptimizer') as mock_reflow:
                mock_optimizer = MagicMock()
                mock_study = optuna.create_study()
                
                # Add a trial with complex param names
                def objective(trial):
                    trial.suggest_float("model.init_args.learning_rate", 1e-4, 1e-2)
                    return 0.5
                
                mock_study.optimize(objective, n_trials=1)
                mock_optimizer.optimize.return_value = mock_study
                mock_reflow.return_value = mock_optimizer
                
                # Run optimization
                study = optimizer.optimize(n_trials=1)
                
                # Check all features worked
                assert optimizer.persistent_config_overrides is not None
                assert "args.n_trials" in optimizer.persistent_config_overrides
                assert "args.trial_steps" in optimizer.persistent_config_overrides
                assert "model.init_args.torch_compile_settings" in optimizer.persistent_config_overrides
                
                # Check logs for evidence of features
                log_text = caplog.text
                assert "safe torch.compile settings" in log_text or "Using safe" in log_text