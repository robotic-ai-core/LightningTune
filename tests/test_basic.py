"""
Basic tests that run without Ray/PyTorch dependencies.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_strategies_import():
    """Test that strategies can be imported."""
    from LightningTune.core.strategies import (
        BOHBStrategy,
        OptunaStrategy,
        RandomSearchStrategy,
    )
    
    # Create strategy
    strategy = BOHBStrategy(max_t=100)
    assert strategy.max_t == 100
    assert strategy.get_strategy_name() == "BOHB"
    
    print("✓ Strategies v2 import successful")


def test_pickle_strategies():
    """Test that strategies can be pickled."""
    import pickle
    from LightningTune.core.strategies import BOHBStrategy
    
    strategy = BOHBStrategy(
        max_t=100,
        reduction_factor=3,
        metric="accuracy",
        mode="max"
    )
    
    # Pickle and unpickle
    pickled = pickle.dumps(strategy)
    restored = pickle.loads(pickled)
    
    assert restored.max_t == 100
    assert restored.reduction_factor == 3
    assert restored.metric == "accuracy"
    assert restored.mode == "max"
    
    print("✓ Strategy pickling successful")


def test_session_state():
    """Test session state serialization concept."""
    import pickle
    import tempfile
    from pathlib import Path
    from LightningTune.core.strategies import RandomSearchStrategy
    
    # Test that strategies and configs can be pickled
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create components
        strategy = RandomSearchStrategy(num_samples=10)
        search_space = {"lr": [0.001, 0.01]}
        config = {"max_epochs": 50}
        
        # Pickle them
        strategy_pickle = pickle.dumps(strategy)
        search_space_pickle = pickle.dumps(search_space)
        config_pickle = pickle.dumps(config)
        
        # Save to file
        state_dict = {
            "experiment_name": "test",
            "session_id": "123",
            "strategy_pickle": strategy_pickle,
            "search_space_pickle": search_space_pickle,
            "config_pickle": config_pickle,
        }
        
        state_path = Path(tmpdir) / "state.pkl"
        with open(state_path, 'wb') as f:
            pickle.dump(state_dict, f)
        
        # Load from file
        with open(state_path, 'rb') as f:
            loaded_dict = pickle.load(f)
        
        assert loaded_dict["experiment_name"] == "test"
        assert loaded_dict["session_id"] == "123"
        
        # Verify pickled objects
        loaded_strategy = pickle.loads(loaded_dict["strategy_pickle"])
        assert loaded_strategy.get_strategy_name() == "RandomSearch"
        assert loaded_strategy.num_samples == 10
        
        loaded_search_space = pickle.loads(loaded_dict["search_space_pickle"])
        assert loaded_search_space == {"lr": [0.001, 0.01]}
        
        print("✓ Session state serialization successful")


def test_dependency_injection_pattern():
    """Test that strategies are self-contained."""
    from LightningTune.core.strategies import (
        BOHBStrategy,
        OptunaStrategy,
        RandomSearchStrategy,
        PBTStrategy,
        GridSearchStrategy,
    )
    
    strategies = [
        BOHBStrategy(max_t=100),
        OptunaStrategy(n_startup_trials=10),
        RandomSearchStrategy(num_samples=20),
        PBTStrategy(perturbation_interval=3),
        GridSearchStrategy(),
    ]
    
    for strategy in strategies:
        # Check required attributes
        assert hasattr(strategy, 'metric')
        assert hasattr(strategy, 'mode')
        assert hasattr(strategy, 'get_strategy_name')
        
        # Check picklability
        import pickle
        pickled = pickle.dumps(strategy)
        restored = pickle.loads(pickled)
        assert restored.get_strategy_name() == strategy.get_strategy_name()
    
    print("✓ Dependency injection pattern validated")


def test_pause_callback_basic():
    """Test pause callback without Ray."""
    import tempfile
    import json
    import time
    from pathlib import Path
    from LightningTune.callbacks.tune_pause_callback import TunePauseCallback
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pause_file = Path(tmpdir) / "pause.signal"
        
        # Create callback
        callback = TunePauseCallback(
            pause_signal_file=pause_file,
            check_interval=0,
        )
        
        # Initially no signal
        assert callback._check_pause_signal() == False
        
        # Write signal
        signal_data = {
            "pause_requested": True,
            "timestamp": time.time(),
        }
        pause_file.write_text(json.dumps(signal_data))
        
        # Now should detect
        assert callback._check_pause_signal() == True
        
        print("✓ Pause callback basic functionality works")


def run_all_tests():
    """Run all basic tests."""
    tests = [
        test_strategies_import,
        test_pickle_strategies,
        test_session_state,
        test_dependency_injection_pattern,
        test_pause_callback_basic,
    ]
    
    print("\n" + "="*50)
    print("Running Basic Tests (No Ray/PyTorch Required)")
    print("="*50 + "\n")
    
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed.append(test.__name__)
    
    print("\n" + "="*50)
    if not failed:
        print("✅ All basic tests passed!")
    else:
        print(f"❌ {len(failed)} tests failed: {', '.join(failed)}")
    print("="*50)
    
    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)