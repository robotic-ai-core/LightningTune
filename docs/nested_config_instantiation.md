# Nested Config Instantiation in LightningTune

LightningTune supports nested configuration instantiation using `class_path` and `init_args` dictionaries, similar to Lightning CLI and LightningReflow. This allows you to specify complex hierarchical models and components directly in configuration files or dictionaries.

## Overview

Nested config instantiation enables:
- **Hierarchical model composition**: Wrap models within adapters or containers
- **Dynamic class loading**: Specify classes by their import paths
- **Declarative configuration**: Define entire model architectures in YAML/dict
- **HPO-friendly design**: Pass complex models to `search_space()` functions

This functionality is automatically available when using `use_reflow=True` (default) in HPORunner.

## Basic Syntax

### Dictionary Format

```python
config = {
    "class_path": "my_module.MyClass",
    "init_args": {
        "param1": value1,
        "param2": value2,
        # Nested components can also use class_path
        "nested_component": {
            "class_path": "my_module.NestedClass",
            "init_args": {
                "nested_param": value
            }
        }
    }
}
```

### YAML Format

```yaml
class_path: my_module.MyClass
init_args:
  param1: value1
  param2: value2
  nested_component:
    class_path: my_module.NestedClass
    init_args:
      nested_param: value
```

## Real-World Example: World Model HPO

Here's how ProtoWorld uses nested config instantiation for hyperparameter optimization:

### Python Search Space

```python
from world_model.models.dynamics import AdapterDynamicsModel, ForwardTransformerDynamicsModel

def search_space(trial):
    """HPO search space with nested model instantiation."""

    # Sample hyperparameters
    internal_latent_dim = trial.suggest_categorical("internal_latent_dim", [190, 238, 286])
    num_heads = trial.suggest_categorical("num_heads", [8, 12, 16])
    num_layers = trial.suggest_int("num_layers", 8, 20, step=2)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Create nested dynamics model
    dynamics_model = AdapterDynamicsModel(
        wrapped_model=ForwardTransformerDynamicsModel(
            latent_dim=internal_latent_dim,
            action_dim=2,
            sequence_length=4,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        ),
        vae_latent_dim=576,
        internal_latent_dim=internal_latent_dim,
    )

    # Return config with instantiated model
    return {
        "model.init_args.dynamics_model": dynamics_model,
        "model.init_args.learning_rate": trial.suggest_float("lr", 1e-6, 1e-5, log=True),
        "model.init_args.weight_decay": trial.suggest_float("weight_decay", 0.05, 1.0, log=True),
    }
```

### Equivalent YAML Config (for reference)

The above could be represented in YAML as:

```yaml
model:
  init_args:
    dynamics_model:
      class_path: world_model.models.dynamics.AdapterDynamicsModel
      init_args:
        wrapped_model:
          class_path: world_model.models.dynamics.ForwardTransformerDynamicsModel
          init_args:
            latent_dim: 238
            action_dim: 2
            sequence_length: 4
            num_heads: 12
            num_layers: 14
            dropout: 0.3
        vae_latent_dim: 576
        internal_latent_dim: 238
    learning_rate: 5.0e-6
    weight_decay: 0.2
```

## Pattern: Factory Functions for HPO

For cleaner HPO code, use factory functions to create complex nested structures:

```python
def create_nested_model(latent_dim: int, num_heads: int, num_layers: int, dropout: float):
    """Factory function for creating nested models."""
    transformer = ForwardTransformerDynamicsModel(
        latent_dim=latent_dim,
        action_dim=2,
        sequence_length=4,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )

    return AdapterDynamicsModel(
        wrapped_model=transformer,
        vae_latent_dim=576,
        internal_latent_dim=latent_dim,
    )


def search_space(trial):
    """Clean search space using factory function."""
    dynamics_model = create_nested_model(
        latent_dim=trial.suggest_categorical("latent_dim", [190, 238, 286]),
        num_heads=trial.suggest_categorical("num_heads", [8, 12, 16]),
        num_layers=trial.suggest_int("num_layers", 8, 20, step=2),
        dropout=trial.suggest_float("dropout", 0.1, 0.5),
    )

    return {
        "model.init_args.dynamics_model": dynamics_model,
        "model.init_args.learning_rate": trial.suggest_float("lr", 1e-6, 1e-5, log=True),
    }
```

## How It Works

### With LightningReflow (use_reflow=True, default)

When `use_reflow=True`, LightningTune leverages LightningReflow's config system:

1. **Config Parsing**: LightningReflow's `LightningArgumentParser` parses nested configs
2. **Class Resolution**: `class_path` strings are resolved to actual Python classes
3. **Instantiation**: Classes are instantiated with their `init_args` recursively
4. **Validation**: Type checking ensures correct parameter types

### Without LightningReflow (use_reflow=False)

When `use_reflow=False`:
- Nested config instantiation is **not available**
- You must pass pre-instantiated objects directly
- Use factory functions to create complex models

## Best Practices

### 1. Use Factory Functions

**Good** - Clear, testable, reusable:
```python
def _create_dynamics_model(latent_dim: int, num_heads: int, num_layers: int, dropout: float):
    transformer = ForwardTransformerDynamicsModel(...)
    return AdapterDynamicsModel(wrapped_model=transformer, ...)

def search_space(trial):
    dynamics_model = _create_dynamics_model(
        latent_dim=trial.suggest_categorical("latent_dim", [190, 238]),
        num_heads=trial.suggest_categorical("num_heads", [8, 12]),
        num_layers=trial.suggest_int("num_layers", 8, 20),
        dropout=trial.suggest_float("dropout", 0.1, 0.5),
    )
    return {"model.init_args.dynamics_model": dynamics_model}
```

**Bad** - Duplicated, hard to test:
```python
def search_space(trial):
    return {
        "model.init_args.dynamics_model": AdapterDynamicsModel(
            wrapped_model=ForwardTransformerDynamicsModel(
                latent_dim=trial.suggest_categorical("latent_dim", [190, 238]),
                ...
            ),
            ...
        )
    }
```

### 2. Centralize Defaults

Create a defaults class for HPO-specific constants:

```python
class HPODefaults:
    """Centralized HPO defaults."""
    VAE_LATENT_DIM = 576
    ACTION_DIM = 2
    SEQUENCE_LENGTH = 4
    BATCH_SIZE = 128

def _create_dynamics_model(latent_dim: int, num_heads: int, num_layers: int, dropout: float):
    transformer = ForwardTransformerDynamicsModel(
        latent_dim=latent_dim,
        action_dim=HPODefaults.ACTION_DIM,  # Use centralized default
        sequence_length=HPODefaults.SEQUENCE_LENGTH,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
    return AdapterDynamicsModel(
        wrapped_model=transformer,
        vae_latent_dim=HPODefaults.VAE_LATENT_DIM,
        internal_latent_dim=latent_dim,
    )
```

### 3. Keep Search Space Functions Clean

**Good** - Focused on hyperparameter logic:
```python
def search_space(trial):
    # Sample hyperparameters
    lr = trial.suggest_float("lr", 1e-6, 1e-5, log=True)
    wd = trial.suggest_float("weight_decay", 0.05, 1.0, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Create model using factory
    dynamics_model = _create_dynamics_model(
        latent_dim=trial.suggest_categorical("latent_dim", [190, 238]),
        num_heads=trial.suggest_categorical("num_heads", [8, 12]),
        num_layers=trial.suggest_int("num_layers", 8, 20),
        dropout=dropout,
    )

    # Return clean config
    return {
        "model.init_args.learning_rate": lr,
        "model.init_args.weight_decay": wd,
        "model.init_args.dynamics_model": dynamics_model,
    }
```

**Bad** - Mixing concerns, hard to read:
```python
def search_space(trial):
    # Everything inline, hard to understand the search space
    return {
        "model.init_args.learning_rate": trial.suggest_float("lr", 1e-6, 1e-5, log=True),
        "model.init_args.weight_decay": trial.suggest_float("weight_decay", 0.05, 1.0, log=True),
        "model.init_args.dynamics_model": AdapterDynamicsModel(
            wrapped_model=ForwardTransformerDynamicsModel(
                latent_dim=trial.suggest_categorical("latent_dim", [190, 238, 286]),
                action_dim=2,
                sequence_length=4,
                num_heads=trial.suggest_categorical("num_heads", [8, 12, 16]),
                num_layers=trial.suggest_int("num_layers", 8, 20, step=2),
                dropout=trial.suggest_float("dropout", 0.1, 0.5),
            ),
            vae_latent_dim=576,
            internal_latent_dim=trial.suggest_categorical("latent_dim", [190, 238, 286]),
        ),
    }
```

## Testing Nested Configs

Test that your factory functions create valid models:

```python
def test_create_dynamics_model():
    """Test factory function creates valid model."""
    model = _create_dynamics_model(
        latent_dim=238,
        num_heads=12,
        num_layers=14,
        dropout=0.3,
    )

    assert isinstance(model, AdapterDynamicsModel)
    assert isinstance(model.wrapped_model, ForwardTransformerDynamicsModel)
    assert model.internal_latent_dim == 238
    assert model.wrapped_model.num_heads == 12
```

## Troubleshooting

### Model not instantiated correctly

**Problem**: Model attributes are None or have wrong values

**Solution**: Check that:
1. `use_reflow=True` in HPORunner
2. Factory function returns correctly instantiated model
3. All required parameters are provided

### Class path not found

**Problem**: `ImportError` or `ModuleNotFoundError` when using `class_path`

**Solution**:
- Ensure the module is on Python path
- Use fully qualified paths (e.g., `my_package.my_module.MyClass`)
- Verify class can be imported: `from my_package.my_module import MyClass`

### Nested parameters not logged to WandB

**Problem**: Hyperparameters from nested models not appearing in WandB

**Solution**: Use the helper function in `_build_base_config()`:

```python
def search_space(trial):
    config = {...}

    # Automatically add all trial hyperparameters for WandB logging
    for param_name, param_value in trial.params.items():
        config[f"hparams.{param_name}"] = param_value

    return config
```

## Summary

Nested config instantiation in LightningTune provides a powerful way to define complex model architectures:

- ✅ Use factory functions for model creation
- ✅ Centralize defaults in a dedicated class
- ✅ Keep search space functions focused on hyperparameter logic
- ✅ Test factory functions independently
- ✅ Works automatically with `use_reflow=True` (default)

For complete examples, see:
- `scripts/world_model_hpo.py` in ProtoWorld
- `tests/test_hpo_lr_schedule_inheritance.py` for testing patterns
