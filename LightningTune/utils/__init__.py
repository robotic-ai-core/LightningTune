"""Utility functions for LightningTune."""

from .config_utils import (
    deep_merge_configs,
    apply_dotted_updates,
    merge_with_dotted_updates,
    load_yaml_config,
)

from .cli_generation import (
    validate_config_for_cli_generation,
    extract_cli_args_from_config,
    format_cli_command,
    describe_search_space,
    format_best_trial_results,
)

# Import debugging utilities from lightning_reflow (canonical source)
from lightning_reflow.utils.debugging import (
    CrashResistantLogger,
    CircularBufferHandler,
    TeeLogger,
    setup_crash_resistant_logging,
    ThreadMonitor,
    monitor_threads_for_duration,
)

__all__ = [
    # Config utilities
    "deep_merge_configs",
    "apply_dotted_updates",
    "merge_with_dotted_updates",
    "load_yaml_config",
    # CLI generation utilities
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
    "monitor_threads_for_duration",
]