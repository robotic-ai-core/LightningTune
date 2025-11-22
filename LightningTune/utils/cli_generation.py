"""
CLI generation utilities for HPO results.

These utilities convert HPO config dictionaries to CLI arguments that can be
used to reproduce the best trial with a training script.
"""

from typing import Dict, Any, List, Callable, Set


def validate_config_for_cli_generation(
    config: Dict[str, Any],
    exclude_prefixes: tuple = ("hparams.",),
) -> None:
    """
    Validate that config has the required structure for CLI generation.

    Args:
        config: Configuration dictionary to validate
        exclude_prefixes: Prefixes to exclude when counting meaningful keys

    Raises:
        ValueError: If config is missing required elements or has invalid structure

    Example:
        >>> config = {"model.init_args.learning_rate": 1e-5}
        >>> validate_config_for_cli_generation(config)  # OK
        >>> validate_config_for_cli_generation({})  # Raises ValueError
    """
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dict, got {type(config)}")

    # Check for at least some meaningful parameters
    meaningful_keys = [
        k for k in config.keys()
        if not any(k.startswith(prefix) for prefix in exclude_prefixes)
    ]
    if not meaningful_keys:
        raise ValueError("Config contains no parameters for CLI generation")


def extract_cli_args_from_config(
    config: Dict[str, Any],
    *,
    base_config_path: str = None,
    extra_args: Dict[str, Any] = None,
    excluded_params: Set[str] = None,
    exclude_prefixes: tuple = ("hparams.",),
    skip_objects: bool = True,
) -> List[str]:
    """
    Extract CLI arguments from a config dict.

    This function converts config dictionaries (typically from HPO search_space)
    to CLI arguments for Lightning CLI scripts.

    Args:
        config: Config dict (e.g., {"model.init_args.learning_rate": 1e-5})
        base_config_path: Optional path to base config file (added as --config)
        extra_args: Additional CLI arguments to append (e.g., {"trainer.max_epochs": 2000})
        excluded_params: Set of parameter keys to exclude from CLI generation
        exclude_prefixes: Prefixes to skip (default: hparams for WandB logging)
        skip_objects: If True, skip non-primitive values (objects, lists, dicts)

    Returns:
        List of CLI argument strings (e.g., ["--config", "...", "--model.learning_rate", "1e-5"])

    Example:
        >>> config = {
        ...     "model.init_args.learning_rate": 1e-5,
        ...     "model.init_args.weight_decay": 0.1,
        ...     "hparams.trial_number": 42,  # Skipped (hparams prefix)
        ... }
        >>> args = extract_cli_args_from_config(
        ...     config,
        ...     base_config_path="configs/model.yaml",
        ...     extra_args={"trainer.max_epochs": 2000}
        ... )
        >>> print(args)
        ['--config', 'configs/model.yaml',
         '--model.init_args.learning_rate', '1e-05',
         '--model.init_args.weight_decay', '0.1',
         '--trainer.max_epochs', '2000']
    """
    # Validate config structure
    validate_config_for_cli_generation(config, exclude_prefixes)

    cli_args = []
    excluded_params = excluded_params or set()

    # Start with base config if provided
    if base_config_path:
        cli_args.extend(["--config", base_config_path])

    # Process config dict
    for key, value in sorted(config.items()):
        # Skip excluded prefixes (e.g., hparams for WandB)
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue

        # Skip excluded parameters
        if key in excluded_params:
            continue

        # Handle different value types
        if isinstance(value, (int, float, str, bool)):
            # Simple value - add directly
            # Format floats to avoid scientific notation issues
            if isinstance(value, float):
                str_value = f"{value:g}"  # Use general format
            elif isinstance(value, bool):
                str_value = str(value).lower()
            else:
                str_value = str(value)
            cli_args.extend([f"--{key}", str_value])

        elif skip_objects:
            # Skip complex objects (models, etc.)
            continue

        else:
            # Try to convert to string (may not work for all objects)
            cli_args.extend([f"--{key}", str(value)])

    # Add extra arguments
    if extra_args:
        for key, value in sorted(extra_args.items()):
            if isinstance(value, float):
                str_value = f"{value:g}"
            elif isinstance(value, bool):
                str_value = str(value).lower()
            else:
                str_value = str(value)
            cli_args.extend([f"--{key}", str_value])

    return cli_args


def format_cli_command(
    cli_args: List[str],
    script: str = "python train.py fit",
    line_continuation: str = " \\\n  ",
) -> str:
    """
    Format CLI arguments as a readable multi-line command.

    Args:
        cli_args: List of CLI arguments (e.g., ["--config", "...", "--model.lr", "1e-5"])
        script: Script path and subcommand (e.g., "python train.py fit")
        line_continuation: String to use for line continuation

    Returns:
        Formatted command string with line continuations

    Example:
        >>> args = ["--config", "config.yaml", "--model.lr", "1e-5"]
        >>> print(format_cli_command(args))
        python train.py fit \\
          --config config.yaml \\
          --model.lr 1e-5
    """
    lines = [script]

    # Group args in pairs (flag + value)
    i = 0
    while i < len(cli_args):
        if cli_args[i].startswith('--'):
            # This is a flag
            if i + 1 < len(cli_args) and not cli_args[i + 1].startswith('--'):
                # Flag with value
                lines.append(f"{cli_args[i]} {cli_args[i + 1]}")
                i += 2
            else:
                # Boolean flag without value
                lines.append(cli_args[i])
                i += 1
        else:
            # Standalone value (shouldn't happen with proper input)
            lines.append(cli_args[i])
            i += 1

    return line_continuation.join(lines)


def describe_search_space(
    search_space_fn: Callable,
    *,
    header: str = "Hyperparameter ranges:",
    indent: str = "  ",
) -> str:
    """
    Generate a description of a search space from its function.

    This inspects the search_space function's source code or trial suggestions
    to generate a human-readable description of the hyperparameter ranges.

    Args:
        search_space_fn: Search space function that takes a trial
        header: Header text for the description
        indent: Indentation for each parameter

    Returns:
        Formatted string describing the search space

    Note:
        This is a basic implementation that relies on function introspection.
        For complex search spaces, consider using a SearchSpaceDescriptor class.
    """
    import inspect

    lines = [header]

    try:
        source = inspect.getsource(search_space_fn)

        # Parse common Optuna patterns
        import re

        # Match suggest_float patterns
        float_pattern = r'trial\.suggest_float\(\s*["\'](\w+)["\'],\s*([^,]+),\s*([^,\)]+)(?:,\s*log\s*=\s*True)?'
        for match in re.finditer(float_pattern, source):
            name, low, high = match.groups()
            log_str = ", log" if "log=True" in match.group(0) else ""
            lines.append(f"{indent}- {name}: [{low.strip()}, {high.strip()}]{log_str}")

        # Match suggest_int patterns
        int_pattern = r'trial\.suggest_int\(\s*["\'](\w+)["\'],\s*([^,]+),\s*([^,\)]+)'
        for match in re.finditer(int_pattern, source):
            name, low, high = match.groups()
            lines.append(f"{indent}- {name}: [{low.strip()}, {high.strip()}]")

        # Match suggest_categorical patterns
        cat_pattern = r'trial\.suggest_categorical\(\s*["\'](\w+)["\'],\s*\[([^\]]+)\]'
        for match in re.finditer(cat_pattern, source):
            name, options = match.groups()
            lines.append(f"{indent}- {name}: [{options.strip()}]")

    except (OSError, TypeError):
        lines.append(f"{indent}(Unable to introspect search space)")

    return "\n".join(lines)


def format_best_trial_results(
    study,
    *,
    header: str = "OPTIMIZATION COMPLETE",
    width: int = 80,
) -> str:
    """
    Format the best trial results from an Optuna study.

    Args:
        study: Optuna study object
        header: Header text
        width: Width of separator lines

    Returns:
        Formatted string with best trial information

    Example:
        >>> results = format_best_trial_results(study)
        >>> print(results)
        ================================================================================
        OPTIMIZATION COMPLETE
        ================================================================================

        Best trial: #42
        Best validation loss: 0.001234

        Best hyperparameters:
        --------------------------------------------------------------------------------
          learning_rate                  = 1e-05
          weight_decay                   = 0.1
        ================================================================================
    """
    lines = []
    separator = "=" * width

    lines.append(separator)
    lines.append(header)
    lines.append(separator)
    lines.append("")
    lines.append(f"Best trial: #{study.best_trial.number}")
    lines.append(f"Best validation loss: {study.best_value:.6f}")
    lines.append("")
    lines.append("Best hyperparameters:")
    lines.append("-" * width)

    for key, value in study.best_params.items():
        lines.append(f"  {key:30s} = {value}")

    lines.append(separator)

    return "\n".join(lines)
