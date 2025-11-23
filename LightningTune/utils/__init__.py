"""Utility functions for LightningTune."""

from .config_utils import (
    deep_merge_configs,
    apply_dotted_updates,
    merge_with_dotted_updates,
    load_yaml_config,
)

from .cli_generation import (
    config_diff_to_dotted,
    validate_config_for_cli_generation,
    extract_cli_args_from_config,
    format_cli_command,
    describe_search_space,
    format_best_trial_results,
)

from .crash_logger import (
    CrashResistantLogger,
    CircularBufferHandler,
    TeeLogger,
    setup_crash_resistant_logging,
)

from .thread_monitor import ThreadMonitor

__all__ = [
    # Config utilities
    "deep_merge_configs",
    "apply_dotted_updates",
    "merge_with_dotted_updates",
    "load_yaml_config",
    # CLI generation utilities
    "config_diff_to_dotted",
    "validate_config_for_cli_generation",
    "extract_cli_args_from_config",
    "format_cli_command",
    "describe_search_space",
    "format_best_trial_results",
    # Crash logging utilities
    "CrashResistantLogger",
    "CircularBufferHandler",
    "TeeLogger",
    "setup_crash_resistant_logging",
    # Thread monitoring
    "ThreadMonitor",
]