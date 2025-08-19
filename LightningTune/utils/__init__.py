"""Utility functions for LightningTune."""

from .config_utils import (
    deep_merge_configs,
    apply_dotted_updates,
    merge_with_dotted_updates,
)

__all__ = [
    "deep_merge_configs",
    "apply_dotted_updates", 
    "merge_with_dotted_updates",
]