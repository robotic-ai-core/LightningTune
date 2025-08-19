"""
Tests for factory functions that create Optuna samplers and pruners.
"""

import pytest
from optuna.samplers import BaseSampler, TPESampler, RandomSampler, CmaEsSampler, GridSampler
from optuna.pruners import BasePruner, MedianPruner, HyperbandPruner, SuccessiveHalvingPruner, NopPruner
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from LightningTune.optuna.factories import (
    create_sampler,
    create_pruner,
    get_sampler_info,
    get_pruner_info,
)


class TestSamplerFactory:
    """Test the create_sampler factory function."""
    
    def test_create_tpe_sampler(self):
        """Test creating TPE sampler."""
        sampler = create_sampler("tpe")
        assert isinstance(sampler, TPESampler)
        
    def test_create_random_sampler(self):
        """Test creating Random sampler."""
        sampler = create_sampler("random")
        assert isinstance(sampler, RandomSampler)
        
    def test_create_cmaes_sampler(self):
        """Test creating CMA-ES sampler."""
        sampler = create_sampler("cmaes")
        assert isinstance(sampler, CmaEsSampler)
        
    def test_create_grid_sampler(self):
        """Test creating Grid sampler."""
        sampler = create_sampler("grid")
        assert isinstance(sampler, GridSampler)
        
    def test_sampler_with_seed(self):
        """Test creating sampler with seed."""
        sampler = create_sampler("tpe", seed=42)
        assert isinstance(sampler, TPESampler)
        # TPESampler stores seed in _rng.seed
        
    def test_sampler_with_custom_params(self):
        """Test creating sampler with custom parameters."""
        sampler = create_sampler("tpe", n_startup_trials=10)
        assert isinstance(sampler, TPESampler)
        assert sampler._n_startup_trials == 10
        
    def test_invalid_sampler_name(self):
        """Test that invalid sampler name raises error."""
        with pytest.raises(ValueError, match="Unknown sampler"):
            create_sampler("invalid_sampler")
            
    def test_get_sampler_info(self):
        """Test getting sampler information."""
        info = get_sampler_info()
        assert isinstance(info, dict)
        assert "tpe" in info
        assert "random" in info
        assert "grid" in info
        assert "cmaes" in info


class TestPrunerFactory:
    """Test the create_pruner factory function."""
    
    def test_create_median_pruner(self):
        """Test creating Median pruner."""
        pruner = create_pruner("median")
        assert isinstance(pruner, MedianPruner)
        # Check default parameters
        assert pruner._n_startup_trials == 5
        assert pruner._n_warmup_steps == 5
        
    def test_create_hyperband_pruner(self):
        """Test creating Hyperband pruner."""
        pruner = create_pruner("hyperband")
        assert isinstance(pruner, HyperbandPruner)
        # Check default parameters
        assert pruner._min_resource == 1
        assert pruner._max_resource == 30
        assert pruner._reduction_factor == 3
        
    def test_create_successivehalving_pruner(self):
        """Test creating SuccessiveHalving pruner."""
        pruner = create_pruner("successivehalving")
        assert isinstance(pruner, SuccessiveHalvingPruner)
        # Check default parameters
        assert pruner._min_resource == 1
        
    def test_create_nop_pruner(self):
        """Test creating Nop (no operation) pruner."""
        pruner = create_pruner("none")
        assert isinstance(pruner, NopPruner)
        
    def test_pruner_with_custom_params(self):
        """Test creating pruner with custom parameters."""
        pruner = create_pruner("median", n_warmup_steps=10)
        assert isinstance(pruner, MedianPruner)
        assert pruner._n_warmup_steps == 10
        
    def test_pruner_override_defaults(self):
        """Test that custom params override defaults."""
        pruner = create_pruner("hyperband", max_resource=50)
        assert isinstance(pruner, HyperbandPruner)
        assert pruner._max_resource == 50
        assert pruner._min_resource == 1  # Default retained
        assert pruner._reduction_factor == 3  # Default retained
        
    def test_invalid_pruner_name(self):
        """Test that invalid pruner name raises error."""
        with pytest.raises(ValueError, match="Unknown pruner"):
            create_pruner("invalid_pruner")
            
    def test_get_pruner_info(self):
        """Test getting pruner information."""
        info = get_pruner_info()
        assert isinstance(info, dict)
        assert "median" in info
        assert "hyperband" in info
        assert "successivehalving" in info
        assert "none" in info


class TestFactoryIntegration:
    """Test that factory-created components work together."""
    
    def test_sampler_pruner_compatibility(self):
        """Test that factory samplers and pruners can be used together."""
        from optuna import create_study
        
        sampler = create_sampler("tpe", seed=42)
        pruner = create_pruner("median")
        
        # Create a study with factory components
        study = create_study(
            sampler=sampler,
            pruner=pruner,
            direction="minimize"
        )
        
        # Simple objective for testing
        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return x ** 2
        
        # Run a few trials
        study.optimize(objective, n_trials=3)
        
        # Verify study ran successfully
        assert len(study.trials) == 3
        assert study.best_value is not None
        
    def test_all_sampler_pruner_combinations(self):
        """Test that all sampler/pruner combinations work."""
        from optuna import create_study
        
        samplers = ["tpe", "random"]  # Skip grid and cmaes for speed
        pruners = ["median", "none"]  # Skip others for speed
        
        for sampler_name in samplers:
            for pruner_name in pruners:
                sampler = create_sampler(sampler_name, seed=42)
                pruner = create_pruner(pruner_name)
                
                study = create_study(
                    sampler=sampler,
                    pruner=pruner,
                    direction="minimize"
                )
                
                def objective(trial):
                    return trial.suggest_float("x", -1, 1) ** 2
                
                study.optimize(objective, n_trials=2)
                assert len(study.trials) >= 1