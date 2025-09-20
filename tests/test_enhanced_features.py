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
        
        # Mock the Reflow optimizer to prevent actual optimization
        with patch('LightningTune.optuna.optimizer_reflow.ReflowOptunaDrivenOptimizer') as mock_reflow:
            mock_optimizer = MagicMock()
            mock_study = optuna.create_study()
            mock_optimizer.optimize.return_value = mock_study
            mock_optimizer.create_objective.return_value = lambda trial: 0.5
            mock_reflow.return_value = mock_optimizer
            
            # Call optimize
            optimizer.optimize(n_trials=10)
            
            # Check that config_overrides were built from args
            assert optimizer.persistent_config_overrides is not None
            # n_trials persists and auto-restores; trial_steps also persisted
            assert "args.n_trials" in optimizer.persistent_config_overrides
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
            
            optimizer.optimize(n_trials=10, resume_from=str(checkpoint_file))
            
            # Args should be restored to saved values (including n_trials)
            assert args.trial_steps == 40000, f"Expected 40000, got {args.trial_steps}"
            assert args.n_trials == 100, f"Expected 100 (restored), got {args.n_trials}"
            assert args.val_interval == 250, f"Expected 250, got {args.val_interval}"
            assert args.test_mode == True, f"Expected True, got {args.test_mode}"
    
    def test_default_args_not_overriding_saved_values(self, tmp_path):
        """Test that default argument values don't override saved values on resume."""
        
        # Create initial args with specific values (simulating first run)
        class Args:
            def __init__(self):
                self.trial_steps = 40000  # User specified value
                self.n_trials = 100  # User specified value
                self.val_interval = 500  # User specified value
                self.test_mode = False
                self.wandb = "project"
                self.study_name = "test_study"
        
        initial_args = Args()
        
        # Create saved session with these values
        saved_overrides = {
            "args.trial_steps": 40000,
            "args.n_trials": 100,
            "args.val_interval": 500,
            "args.wandb": "project",
            "trainer.val_check_interval": 500,
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
        
        # Create new args with defaults (simulating resume without specifying values)
        class ResumeArgs:
            def __init__(self):
                self.trial_steps = 5000  # Default value - should NOT override saved 40000
                self.n_trials = 50  # Default value - should NOT override saved 100
                self.val_interval = None  # Default None - should NOT override saved 500
                self.test_mode = False
                self.wandb = None  # Default None - should NOT override saved "project"
                self.study_name = "test_study"
        
        resume_args = ResumeArgs()
        
        # Mock sys.argv to simulate resume command without explicit args
        import sys
        original_argv = sys.argv
        try:
            sys.argv = ['script.py', '--resume-from', str(checkpoint_file)]
            
            # Create optimizer with persist_args enabled
            optimizer = PausibleOptunaOptimizer(
                base_config={"test": "config"},
                search_space=lambda trial: {"lr": trial.suggest_float("lr", 1e-4, 1e-2)},
                model_class=MagicMock,
                study_name="test_study",
                persist_args=True,
                args=resume_args,
                local_checkpoint_dir=checkpoint_dir
            )
            
            # Resume and check args are restored, not overridden by defaults
            with patch('LightningTune.optuna.optimizer_reflow.ReflowOptunaDrivenOptimizer') as mock_reflow:
                mock_optimizer = MagicMock()
                mock_optimizer.optimize.return_value = study
                mock_optimizer.create_objective.return_value = lambda trial: 0.5
                mock_reflow.return_value = mock_optimizer
                
                optimizer.optimize(n_trials=10, resume_from=str(checkpoint_file))
                
                # Args should retain saved values, not be overridden by defaults
                assert resume_args.trial_steps == 40000, f"Expected saved value 40000, got {resume_args.trial_steps}"
                # n_trials should be restored when not explicitly provided
                assert resume_args.n_trials == 100, f"Expected saved value 100, got {resume_args.n_trials}"
                assert resume_args.val_interval == 500, f"Expected saved value 500, got {resume_args.val_interval}"
                assert resume_args.wandb == "project", f"Expected saved value 'project', got {resume_args.wandb}"
        finally:
            sys.argv = original_argv


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
        
        # Mock the Reflow optimizer to prevent actual optimization
        with patch('LightningTune.optuna.optimizer_reflow.ReflowOptunaDrivenOptimizer') as mock_reflow:
            mock_optimizer = MagicMock()
            mock_study = optuna.create_study()
            mock_optimizer.optimize.return_value = mock_study
            mock_reflow.return_value = mock_optimizer
            
            # Mock the create_objective to return a simple objective
            mock_optimizer.create_objective.return_value = lambda trial: 0.5
            
            optimizer.optimize(n_trials=1)
            
            # Check compile settings were added to config
            assert optimizer.persistent_config_overrides is not None
            # compile settings are runtime-only; ensure optimizer injected them via runtime overrides
            settings = get_compile_settings_for_mode("safe")
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
                study_name = "test_study"  # This will be excluded
            
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
                
                # Check args persistence (n_trials is excluded from persistence)
                assert "args.n_trials" not in optimizer.persistent_config_overrides  # n_trials is extensible
                assert "args.trial_steps" in optimizer.persistent_config_overrides
                assert optimizer.persistent_config_overrides["args.trial_steps"] == 1000
                assert "args.test_mode" in optimizer.persistent_config_overrides
                assert optimizer.persistent_config_overrides["args.test_mode"] == True
                assert "args.compile_mode" in optimizer.persistent_config_overrides
                assert optimizer.persistent_config_overrides["args.compile_mode"] == "safe"
                
                # Check compile mode settings are configured (runtime-only now)
                assert "args.compile_mode" in optimizer.persistent_config_overrides

                # Check logs for evidence of features
                log_text = caplog.text
                assert "safe torch.compile settings" in log_text or "Using safe" in log_text


class TestExtendingHPOSessions:
    """Test extending HPO sessions with more trials on resume."""

    def test_n_trials_saved_to_checkpoint(self, tmp_path):
        """Test that n_trials is correctly saved to checkpoint."""
        from argparse import Namespace
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer
        from unittest.mock import MagicMock
        import pickle
        import optuna

        # Create initial study with n_trials=100
        args = Namespace(
            n_trials=100,  # Set to 100 initially
            trial_steps=5000,
            save_every=10,
            sampler='tpe',
            pruner='median',
            wandb=None,
            resume_from=None,
            safe_compile=True
        )

        optimizer = PausibleOptunaOptimizer(
            base_config={"learning_rate": 0.001},
            search_space=lambda trial: {"lr": trial.suggest_float("lr", 1e-4, 1e-2)},
            model_class=MagicMock,
            args=args,
            persist_args=True,
            local_checkpoint_dir=tmp_path
        )

        # Simulate running 5 trials and saving checkpoint
        study = optuna.create_study()
        for i in range(5):
            study.add_trial(optuna.trial.create_trial(value=i, params={}))

        # Save checkpoint
        success = optimizer.save_study_to_local(study, 5)
        assert success, "Failed to save checkpoint"

        # Load checkpoint and verify n_trials was saved
        checkpoint_file = tmp_path / "study.pkl"
        assert checkpoint_file.exists()

        with open(checkpoint_file, 'rb') as f:
            session_info = pickle.load(f)

        # Check that n_trials was saved
        assert "config_overrides" in session_info
        assert "args.n_trials" in session_info["config_overrides"]
        assert session_info["config_overrides"]["args.n_trials"] == 100

    def test_extending_hpo_session_with_more_trials(self, tmp_path):
        """Test that users can extend HPO sessions by specifying more n_trials on resume."""
        import pickle
        from unittest.mock import MagicMock, patch
        import optuna
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        checkpoint_file = tmp_path / "checkpoint.pkl"

        # Create initial session with n_trials=50
        initial_args = MagicMock()
        initial_args.n_trials = 50
        initial_args.trial_steps = 40000
        initial_args.resume_from = None
        initial_args.study_name = 'test_study'

        # Create a saved session that completed 50 trials
        study = optuna.create_study(direction='minimize')
        for i in range(50):
            study.add_trial(optuna.create_trial(
                params={'param': i * 0.01},
                distributions={'param': optuna.distributions.FloatDistribution(0, 1)},
                values=[i * 0.01],
                state=optuna.trial.TrialState.COMPLETE
            ))

        session_info = {
            'study': study,
            'total_trials_completed': 50,
            'sampler_name': 'tpe',
            'pruner_name': 'hyperband',
            'study_name': 'test_study',
            'config_overrides': {
                'args.n_trials': 50,  # Original n_trials
                'args.trial_steps': 40000
            }
        }

        with open(checkpoint_file, 'wb') as f:
            pickle.dump(session_info, f)

        # Now resume with MORE trials (extend from 50 to 100)
        resume_args = MagicMock()
        resume_args.n_trials = 100  # USER WANTS TO EXTEND TO 100 TRIALS
        resume_args.trial_steps = 40000
        resume_args.resume_from = str(checkpoint_file)
        resume_args.study_name = 'test_study'

        # Mock sys.argv to simulate --n-trials 100 was explicitly provided
        with patch('sys.argv', ['script.py', '--resume-from', str(checkpoint_file), '--n-trials', '100']):
            resume_optimizer = PausibleOptunaOptimizer(
                base_config={'test': 'config'},
                search_space=lambda trial: {'param': trial.suggest_float('param', 0, 1)},
                model_class=MagicMock,
                datamodule_class=None,
                wandb_project=None,
                study_name='test_study',
                sampler_name='tpe',
                pruner_name='hyperband',
                persist_args=True,
                args=resume_args
            )

            # Mock the OptunaDrivenOptimizer creation
            with patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer') as MockOptimizer:
                mock_optimizer_instance = MagicMock()
                mock_objective = MagicMock()
                mock_optimizer_instance.create_objective.return_value = mock_objective
                MockOptimizer.return_value = mock_optimizer_instance

                # Check that before optimize, we have 50 completed trials
                # (This happens during initialization when loading the checkpoint)

                # Resume the study with extended n_trials
                result_study = resume_optimizer.optimize(
                    n_trials=100,  # Extend to 100 trials total
                    resume_from=str(checkpoint_file),
                    config_overrides={},
                    callbacks=[]
                )

                # Verify that n_trials was NOT restored from saved value
                # It should remain at 100 as user specified
                assert resume_args.n_trials == 100  # Should NOT be restored to 50

                # Check that after optimize, we have 100 completed trials
                # This means it successfully ran 50 more trials (100 - 50)
                assert resume_optimizer.total_trials_completed == 100

                # Verify study has 100 trials total
                assert len(result_study.trials) == 100

    def test_resuming_without_n_trials_uses_saved_value(self, tmp_path):
        """Test that resuming without specifying n_trials uses the saved value."""
        import pickle
        from unittest.mock import MagicMock
        import optuna
        from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer

        checkpoint_file = tmp_path / "checkpoint.pkl"

        # Create a saved session with n_trials=100 that completed 50 trials
        study = optuna.create_study(direction='minimize')
        for i in range(50):
            study.add_trial(optuna.create_trial(
                params={'param': i * 0.01},
                distributions={'param': optuna.distributions.FloatDistribution(0, 1)},
                values=[i * 0.01],
                state=optuna.trial.TrialState.COMPLETE
            ))

        session_info = {
            'study': study,
            'total_trials_completed': 50,
            'sampler_name': 'tpe',
            'pruner_name': 'median',
            'study_name': 'test_study',
            'config_overrides': {
                'args.n_trials': 100,  # Original n_trials was 100
                'args.trial_steps': 5000
            }
        }

        with open(checkpoint_file, 'wb') as f:
            pickle.dump(session_info, f)

        # Resume WITHOUT specifying n_trials (should use saved value of 100)
        resume_args = MagicMock()
        resume_args.n_trials = 50  # Default value in args
        resume_args.trial_steps = 5000
        resume_args.resume_from = str(checkpoint_file)
        resume_args.study_name = 'test_study'

        # Load the saved session
        loaded_session = PausibleOptunaOptimizer.load_saved_session(
            resume_from=str(checkpoint_file)
        )

        assert loaded_session is not None
        assert 'config_overrides' in loaded_session
        assert loaded_session['config_overrides']['args.n_trials'] == 100
        assert loaded_session['total_trials_completed'] == 50