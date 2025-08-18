# Dict Configuration Support for LightningTune

## ✨ New Feature

ConfigDrivenOptimizer now accepts Python dictionaries directly as configuration, in addition to file paths. This makes it easier to programmatically generate configurations without creating temporary files.

## 📋 Changes Made

### 1. Modified Classes
- **ConfigManager**: Now accepts `base_config_source` which can be:
  - A dictionary with configuration
  - A path (str or Path) to a YAML/JSON file
  - None for empty configuration

- **ConfigDrivenOptimizer**: Updated to use `base_config_source` instead of `base_config_path`
  - Maintains backward compatibility with deprecation warning
  - Handles both dict and file inputs seamlessly

### 2. API Changes

#### Old API (Still Works with Warning):
```python
optimizer = ConfigDrivenOptimizer(
    base_config_path="config.yaml",  # Deprecated
    search_space=search_space,
    strategy=strategy,
)
```

#### New API with File:
```python
optimizer = ConfigDrivenOptimizer(
    base_config_source="config.yaml",  # File path
    search_space=search_space,
    strategy=strategy,
)
```

#### New API with Dict:
```python
config_dict = {
    "model": {
        "class_path": "model.MyModel",
        "init_args": {"learning_rate": 0.001}
    },
    "data": {
        "class_path": "data.MyDataModule",
        "init_args": {"batch_size": 32}
    },
    "trainer": {"max_epochs": 10}
}

optimizer = ConfigDrivenOptimizer(
    base_config_source=config_dict,  # Dict directly!
    search_space=search_space,
    strategy=strategy,
)
```

## ✅ Testing

### Test Coverage
- **12 new unit tests** in `tests/unit/test_config_dict_support.py`
- **All existing tests updated** to use new parameter name
- **Backward compatibility tested** with deprecation warnings

### Test Results
```
✅ 55 tests passed
✅ 1 skipped
✅ 0 failures
```

## 🎯 Benefits

1. **No Temporary Files**: No need to create YAML/JSON files for simple configs
2. **Programmatic Generation**: Easy to generate configs dynamically in code
3. **Testing**: Simpler unit tests without file I/O
4. **Backward Compatible**: Old code continues to work with deprecation warning
5. **Flexible**: Choose between dict or file based on use case

## 📚 Examples

Three examples demonstrate the feature:
1. **dict_config_example.py** - Shows both dict and file approaches
2. **simple_demo.py** - Updated to use new API
3. **minimal_example.py** - Updated to use new API

## 🔄 Migration Guide

### For Existing Code
Simply replace `base_config_path` with `base_config_source`:

```python
# Old
optimizer = ConfigDrivenOptimizer(
    base_config_path="config.yaml",
    ...
)

# New
optimizer = ConfigDrivenOptimizer(
    base_config_source="config.yaml",  # Works the same!
    ...
)
```

### For New Code
Consider using dict config for dynamic configurations:

```python
# Generate config programmatically
config = generate_config(model_type, dataset_size)

# Use directly without file
optimizer = ConfigDrivenOptimizer(
    base_config_source=config,  # No file needed!
    ...
)
```

## 🏗️ Implementation Details

- **ConfigManager** constructor now handles type detection
- Dict configs stored in `base_config` directly
- File configs loaded as before
- Merging logic unchanged
- All optimization functionality remains the same

## ✨ Summary

This feature makes LightningTune more flexible and easier to use programmatically while maintaining full backward compatibility. Choose dict config for dynamic/test scenarios and file config for persistent/production configurations.