"""
Argument persistence helpers to unify CLI vs saved-args precedence and n_trials extension.
"""

from __future__ import annotations

import sys
from typing import Dict, Any, Iterable, Optional, Tuple


def was_arg_specified(arg_name: str, argv: Optional[Iterable[str]] = None) -> bool:
    patterns = [f"--{arg_name.replace('_', '-')}", f"--{arg_name}"]
    cmd_args = ' '.join(argv) if argv is not None else ' '.join(sys.argv)
    return any(p in cmd_args for p in patterns)


def merge_args_with_saved(
    args_obj: Any,
    saved_overrides: Dict[str, Any],
    *,
    non_persistent: Optional[Iterable[str]] = None,
    argv: Optional[Iterable[str]] = None,
) -> Tuple[int, int]:
    """Restore args from saved_overrides into args_obj unless explicitly specified.

    Returns (restored_count, overridden_count) for logging.
    """
    non_persistent = set(non_persistent or ())
    restored = 0
    overridden = 0
    for key, value in (saved_overrides or {}).items():
        if not key.startswith('args.'):
            continue
        arg_name = key[5:]
        if arg_name in non_persistent:
            continue
        if hasattr(args_obj, arg_name):
            if was_arg_specified(arg_name, argv=argv):
                if getattr(args_obj, arg_name) != value:
                    overridden += 1
            else:
                setattr(args_obj, arg_name, value)
                restored += 1
    return restored, overridden


def normalize_n_trials_in_overrides(saved: Dict[str, Any]) -> None:
    """Normalize legacy 'n_trials' to 'args.n_trials' and resolve conflicts."""
    if 'n_trials' in saved:
        bare = saved.get('n_trials')
        args_nt = saved.get('args.n_trials')
        if args_nt is None:
            saved['args.n_trials'] = bare
        elif bare and args_nt and bare != args_nt:
            saved['args.n_trials'] = max(bare, args_nt)
        del saved['n_trials']


def extend_or_align_n_trials(current_n_trials: int, saved_n_trials: Optional[int], *, cli_specified: bool) -> Tuple[int, bool]:
    """Return possibly-extended n_trials and a flag if extended.

    Rules:
    - If saved exists and current > saved → extend (return current, extended=True)
    - If saved exists and current < saved:
      - if CLI specified, respect the lower value (return current)
      - else, use saved (return saved)
    - Else, unchanged
    """
    extended = False
    if saved_n_trials and current_n_trials > saved_n_trials:
        extended = True
        return current_n_trials, extended
    if saved_n_trials and current_n_trials < saved_n_trials:
        if cli_specified:
            return current_n_trials, False
        return saved_n_trials, False
    return current_n_trials, False


