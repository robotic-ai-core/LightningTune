"""
Integration tests for BoTorchSampler support in LightningTune.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # For ProtoWorld

try:
    from optuna.integration.botorch import BoTorchSampler
    BOTORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    BOTORCH_AVAILABLE = False

from LightningTune import PausibleOptunaOptimizer
from LightningTune.optuna.factories import create_sampler, get_sampler_info


class TestBoTorchIntegration:
    """Test BoTorchSampler integration with LightningTune."""
    
    @pytest.mark.skipif(not BOTORCH_AVAILABLE, reason="BoTorch not installed")
    def test_botorch_in_factory(self):
        """Test that BoTorchSampler is available through factory."""
        sampler = create_sampler("botorch")
        assert isinstance(sampler, BoTorchSampler)
    
    @pytest.mark.skipif(not BOTORCH_AVAILABLE, reason="BoTorch not installed")
    def test_botorch_with_custom_params(self):
        """Test BoTorchSampler with custom parameters."""
        from optuna.samplers import RandomSampler
        
        sampler = create_sampler(
            "botorch",
            n_startup_trials=5,
            independent_sampler=RandomSampler(seed=42)
        )
        assert isinstance(sampler, BoTorchSampler)
        assert sampler._n_startup_trials == 5
    
    @pytest.mark.skipif(not BOTORCH_AVAILABLE, reason="BoTorch not installed")
    def test_botorch_in_sampler_info(self):
        """Test that BoTorchSampler appears in sampler info."""
        info = get_sampler_info()
        assert "botorch" in info
        assert "GP-based" in info["botorch"]
        assert "expensive" in info["botorch"].lower()
    
    @pytest.mark.skipif(not BOTORCH_AVAILABLE, reason="BoTorch not installed")
    def test_pausible_optimizer_with_botorch(self):
        """Test PausibleOptunaOptimizer works with BoTorchSampler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal config
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("""
model:
  learning_rate: 0.001
trainer:
  max_steps: 10
""")
            
            # Create optimizer with BoTorch
            optimizer = PausibleOptunaOptimizer(
                base_config=str(config_file),
                search_space=lambda trial: {"lr": trial.suggest_float("lr", 1e-5, 1e-3)},
                model_class=Mock,
                datamodule_class=Mock,
                sampler_name="botorch",
                pruner_name="median",
                save_every_n_trials=10,
                enable_pause=False,
                use_reflow=False,
            )
            
            # Verify sampler name is set correctly
            assert optimizer.sampler_name == "botorch"
    
    @pytest.mark.skipif(not BOTORCH_AVAILABLE, reason="BoTorch not installed")
    @patch('optuna.Study.optimize')
    @patch('LightningTune.optuna.optimizer.OptunaDrivenOptimizer')
    def test_botorch_in_optimization_loop(self, mock_optimizer_class, mock_optimize):
        """Test that BoTorchSampler works in actual optimization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("model: {}\ntrainer: {}")
            
            # Mock optimizer
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            mock_optimizer.create_objective.return_value = lambda trial: 0.5
            
            # Create pausible optimizer with BoTorch
            optimizer = PausibleOptunaOptimizer(
                base_config=str(config_file),
                search_space=lambda trial: {},
                model_class=Mock,
                sampler_name="botorch",
                pruner_name="none",
                save_every_n_trials=10,
                enable_pause=False,
            )
            
            # Run optimization
            study = optimizer.optimize(n_trials=1)
            
            # Verify optimize was called
            assert mock_optimize.called
    
    def test_botorch_graceful_fallback(self):
        """Test graceful handling when BoTorch is not available."""
        if not BOTORCH_AVAILABLE:
            # Should raise ValueError for unknown sampler
            with pytest.raises(ValueError, match="Unknown sampler"):
                create_sampler("botorch")
            
            # Sampler info should not include botorch
            info = get_sampler_info()
            assert "botorch" not in info
    
    @pytest.mark.skipif(not BOTORCH_AVAILABLE, reason="BoTorch not installed")
    def test_botorch_with_continuous_search_space(self):
        """Test BoTorchSampler with continuous parameters (its strength)."""
        import optuna
        
        sampler = create_sampler("botorch", n_startup_trials=2)
        study = optuna.create_study(sampler=sampler, direction="minimize")
        
        def objective(trial):
            # All continuous parameters - ideal for BoTorch
            x = trial.suggest_float("x", -10, 10)
            y = trial.suggest_float("y", -10, 10)
            return x**2 + y**2
        
        # Run a few trials
        study.optimize(objective, n_trials=5)
        
        assert len(study.trials) == 5
        assert study.best_value is not None
        # After startup trials, BoTorch should start improving
        later_values = [t.value for t in study.trials[2:]]
        assert min(later_values) <= study.trials[0].value  # Should improve
    
    @pytest.mark.skipif(not BOTORCH_AVAILABLE, reason="BoTorch not installed")
    def test_botorch_acquisition_functions(self):
        """Test different acquisition functions for BoTorchSampler."""
        for acq_func in ["EI", "UCB", "PI"]:  # Expected Improvement, Upper Confidence Bound, Probability of Improvement
            sampler = create_sampler(
                "botorch",
                n_startup_trials=2,
                acquisition_function=acq_func
            )
            assert isinstance(sampler, BoTorchSampler)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])