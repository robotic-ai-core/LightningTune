"""
Unit tests for config utility functions.
"""

import pytest
from LightningTune.utils.config_utils import (
    deep_merge_configs,
    apply_dotted_updates,
    merge_with_dotted_updates,
)


class TestDeepMergeConfigs:
    """Tests for deep_merge_configs function."""
    
    def test_simple_merge(self):
        """Test merging flat dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge_configs(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}
        # Ensure base is not modified
        assert base == {"a": 1, "b": 2}
    
    def test_nested_merge(self):
        """Test merging nested dictionaries."""
        base = {
            "model": {"lr": 0.1, "layers": 3},
            "data": {"batch_size": 32}
        }
        override = {
            "model": {"lr": 0.01, "dropout": 0.5}
        }
        result = deep_merge_configs(base, override)
        assert result == {
            "model": {"lr": 0.01, "layers": 3, "dropout": 0.5},
            "data": {"batch_size": 32}
        }
    
    def test_deep_nested_merge(self):
        """Test merging deeply nested structures."""
        base = {
            "trainer": {
                "callbacks": {
                    "early_stopping": {"patience": 5},
                    "checkpoint": {"save_top_k": 1}
                }
            }
        }
        override = {
            "trainer": {
                "callbacks": {
                    "early_stopping": {"patience": 10, "monitor": "val_loss"}
                }
            }
        }
        result = deep_merge_configs(base, override)
        assert result == {
            "trainer": {
                "callbacks": {
                    "early_stopping": {"patience": 10, "monitor": "val_loss"},
                    "checkpoint": {"save_top_k": 1}
                }
            }
        }
    
    def test_override_with_non_dict(self):
        """Test that non-dict values completely override."""
        base = {"model": {"type": "cnn", "layers": 3}}
        override = {"model": "transformer"}
        result = deep_merge_configs(base, override)
        assert result == {"model": "transformer"}
    
    def test_empty_configs(self):
        """Test edge cases with empty configs."""
        assert deep_merge_configs({}, {}) == {}
        assert deep_merge_configs({"a": 1}, {}) == {"a": 1}
        assert deep_merge_configs({}, {"a": 1}) == {"a": 1}
    
    def test_list_values(self):
        """Test that lists are replaced, not merged."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = deep_merge_configs(base, override)
        assert result == {"items": [4, 5]}


class TestApplyDottedUpdates:
    """Tests for apply_dotted_updates function."""
    
    def test_simple_dotted_update(self):
        """Test basic dotted notation updates."""
        config = {"model": {"lr": 0.1}}
        updates = {"model.lr": 0.01}
        result = apply_dotted_updates(config, updates)
        assert result == {"model": {"lr": 0.01}}
        # Ensure original is not modified
        assert config == {"model": {"lr": 0.1}}
    
    def test_create_nested_path(self):
        """Test creating new nested paths."""
        config = {"model": {}}
        updates = {"model.optimizer.lr": 0.01}
        result = apply_dotted_updates(config, updates)
        assert result == {"model": {"optimizer": {"lr": 0.01}}}
    
    def test_multiple_dotted_updates(self):
        """Test multiple dotted updates."""
        config = {"model": {"lr": 0.1}, "data": {"batch_size": 32}}
        updates = {
            "model.lr": 0.01,
            "model.weight_decay": 0.001,
            "data.batch_size": 64,
            "trainer.max_epochs": 100
        }
        result = apply_dotted_updates(config, updates)
        assert result == {
            "model": {"lr": 0.01, "weight_decay": 0.001},
            "data": {"batch_size": 64},
            "trainer": {"max_epochs": 100}
        }
    
    def test_deep_dotted_path(self):
        """Test deeply nested dotted paths."""
        config = {}
        updates = {"a.b.c.d.e": "value"}
        result = apply_dotted_updates(config, updates)
        assert result == {"a": {"b": {"c": {"d": {"e": "value"}}}}}
    
    def test_non_dotted_keys(self):
        """Test that non-dotted keys work as regular updates."""
        config = {"a": 1}
        updates = {"b": 2, "c": {"d": 3}}
        result = apply_dotted_updates(config, updates)
        assert result == {"a": 1, "b": 2, "c": {"d": 3}}
    
    def test_override_existing_structure(self):
        """Test that dotted updates can override existing structures."""
        config = {"model": {"optimizer": {"type": "adam", "lr": 0.1}}}
        updates = {"model.optimizer.lr": 0.01}
        result = apply_dotted_updates(config, updates)
        assert result == {"model": {"optimizer": {"type": "adam", "lr": 0.01}}}
    
    def test_empty_updates(self):
        """Test with empty updates."""
        config = {"a": 1}
        result = apply_dotted_updates(config, {})
        assert result == {"a": 1}


class TestMergeWithDottedUpdates:
    """Tests for merge_with_dotted_updates function."""
    
    def test_combined_merge_and_dotted(self):
        """Test combining deep merge and dotted updates."""
        base = {"model": {"lr": 0.1}, "data": {"batch_size": 32}}
        override = {"model": {"weight_decay": 0.01}}
        dotted = {"model.lr": 0.001}
        
        result = merge_with_dotted_updates(base, override, dotted)
        assert result == {
            "model": {"lr": 0.001, "weight_decay": 0.01},
            "data": {"batch_size": 32}
        }
    
    def test_only_base(self):
        """Test with only base config."""
        base = {"a": 1}
        result = merge_with_dotted_updates(base)
        assert result == {"a": 1}
    
    def test_only_override(self):
        """Test with only override."""
        base = {"a": 1}
        override = {"b": 2}
        result = merge_with_dotted_updates(base, override)
        assert result == {"a": 1, "b": 2}
    
    def test_only_dotted(self):
        """Test with only dotted updates."""
        base = {"model": {"lr": 0.1}}
        dotted = {"model.lr": 0.01}
        result = merge_with_dotted_updates(base, dotted_updates=dotted)
        assert result == {"model": {"lr": 0.01}}
    
    def test_order_of_operations(self):
        """Test that dotted updates are applied after deep merge."""
        base = {"model": {"lr": 0.1}}
        override = {"model": {"lr": 0.05}}  # This should be applied first
        dotted = {"model.lr": 0.001}  # This should override the merge
        
        result = merge_with_dotted_updates(base, override, dotted)
        assert result["model"]["lr"] == 0.001
    
    def test_complex_scenario(self):
        """Test a complex real-world scenario."""
        base = {
            "model": {
                "class_path": "MyModel",
                "init_args": {
                    "lr": 0.1,
                    "layers": 3
                }
            },
            "trainer": {
                "max_epochs": 100,
                "callbacks": []
            }
        }
        override = {
            "model": {
                "init_args": {
                    "dropout": 0.5
                }
            },
            "trainer": {
                "callbacks": ["checkpoint"]
            }
        }
        dotted = {
            "model.init_args.lr": 0.001,
            "trainer.max_epochs": 50
        }
        
        result = merge_with_dotted_updates(base, override, dotted)
        
        assert result == {
            "model": {
                "class_path": "MyModel",
                "init_args": {
                    "lr": 0.001,
                    "layers": 3,
                    "dropout": 0.5
                }
            },
            "trainer": {
                "max_epochs": 50,
                "callbacks": ["checkpoint"]
            }
        }


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_none_values(self):
        """Test handling of None values."""
        assert deep_merge_configs({"a": 1}, {"a": None}) == {"a": None}
        assert apply_dotted_updates({"a": 1}, {"a": None}) == {"a": None}
    
    def test_special_characters_in_keys(self):
        """Test keys with special characters (non-dotted)."""
        config = {}
        updates = {"model-name": "test", "data_source": "file"}
        result = apply_dotted_updates(config, updates)
        assert result == {"model-name": "test", "data_source": "file"}
    
    def test_numeric_values(self):
        """Test various numeric types."""
        base = {"int": 1, "float": 1.0}
        override = {"int": 2.5, "float": 2}
        result = deep_merge_configs(base, override)
        assert result == {"int": 2.5, "float": 2}
    
    def test_boolean_values(self):
        """Test boolean values."""
        config = {"enabled": True}
        updates = {"enabled": False}
        result = apply_dotted_updates(config, updates)
        assert result == {"enabled": False}