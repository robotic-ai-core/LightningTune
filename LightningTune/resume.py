"""
Resume manager for HPO study continuation.

This module consolidates all resume-related logic including:
- Loading checkpoints from various sources (local, WandB)
- Restoring command-line arguments from saved state
- Building resume commands for display
- Managing the resume workflow
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .persistence import (
    BasePersistence,
    CompositePersistence,
    LocalPersistence,
    StudyMetadata,
    WandBPersistence,
    load_saved_session,
    parse_cli_arg,
)
from .arg_persistence import (
    extend_or_align_n_trials,
    merge_args_with_saved,
    normalize_n_trials_in_overrides,
    was_arg_specified,
)

logger = logging.getLogger(__name__)


class ResumeManager:
    """
    Manages HPO study resume/checkpoint logic.

    This class consolidates all resume-related operations that were previously
    spread across HPORunner and PausibleOptunaOptimizer, providing a single
    source of truth for resume behavior.

    Example:
        >>> manager = ResumeManager(
        ...     local_checkpoint_dir=Path("checkpoints/my_study"),
        ...     wandb_project="my-project",
        ...     study_name="my_study",
        ... )
        >>> checkpoint = manager.load_checkpoint("latest")
        >>> if checkpoint:
        ...     study = checkpoint["study"]
        ...     manager.restore_args(args, checkpoint)
    """

    def __init__(
        self,
        local_checkpoint_dir: Optional[Path] = None,
        wandb_project: Optional[str] = None,
        study_name: Optional[str] = None,
        persist_args: bool = True,
        args_exclude: Optional[Set[str]] = None,
    ):
        """
        Initialize resume manager.

        Args:
            local_checkpoint_dir: Directory for local checkpoints.
            wandb_project: WandB project name.
            study_name: Study name for WandB artifacts.
            persist_args: Whether to persist/restore command-line arguments.
            args_exclude: Set of argument names to exclude from persistence.
        """
        self.local_checkpoint_dir = Path(local_checkpoint_dir) if local_checkpoint_dir else None
        self.wandb_project = wandb_project
        self.study_name = study_name
        self.persist_args = persist_args
        self.args_exclude = args_exclude or {'resume_from', 'study_name'}

        # Create persistence backends
        self._persistence = self._create_persistence()

    def _create_persistence(self) -> Optional[BasePersistence]:
        """Create appropriate persistence backend(s)."""
        backends: List[BasePersistence] = []

        if self.local_checkpoint_dir:
            backends.append(LocalPersistence(self.local_checkpoint_dir))

        if self.wandb_project and self.study_name:
            backends.append(WandBPersistence(self.wandb_project, self.study_name))

        if not backends:
            return None
        elif len(backends) == 1:
            return backends[0]
        else:
            # Composite: local is primary (index 0)
            return CompositePersistence(backends, primary_index=0)

    @property
    def persistence(self) -> Optional[BasePersistence]:
        """Get the persistence backend."""
        return self._persistence

    def load_checkpoint(
        self,
        resume_from: str,
        *,
        prefer_local: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint from various sources.

        Tries sources in order:
        1. If resume_from is a valid local path, load from there
        2. If prefer_local and local_checkpoint_dir exists, try local
        3. If wandb_project/study_name set, try WandB
        4. Fall back to legacy load_saved_session

        Args:
            resume_from: Path, "latest", or WandB version string.
            prefer_local: Whether to prefer local over WandB.

        Returns:
            Session info dict with 'study', metadata, etc., or None.
        """
        # Direct local path
        if os.path.exists(resume_from):
            logger.info(f"📁 Loading from local path: {resume_from}")
            if self.local_checkpoint_dir:
                local = LocalPersistence(Path(resume_from))
                return local.load_study("latest")
            return load_saved_session(resume_from)

        # Try persistence backends
        if self._persistence:
            if prefer_local and isinstance(self._persistence, CompositePersistence):
                # Local first
                result = self._persistence.load_study(resume_from)
            elif prefer_local and isinstance(self._persistence, LocalPersistence):
                result = self._persistence.load_study(resume_from)
            else:
                result = self._persistence.load_study(resume_from)

            if result is not None:
                return result

        # Legacy fallback
        return load_saved_session(
            resume_from,
            wandb_project=self.wandb_project,
            study_name=self.study_name,
        )

    def restore_args(
        self,
        args_obj: Any,
        checkpoint: Dict[str, Any],
        *,
        argv: Optional[List[str]] = None,
    ) -> Tuple[int, int]:
        """
        Restore command-line arguments from a checkpoint.

        Only restores arguments that were not explicitly specified on the
        current command line.

        Args:
            args_obj: Parsed arguments object (e.g., from argparse).
            checkpoint: Loaded checkpoint dictionary.
            argv: Command line arguments (defaults to sys.argv).

        Returns:
            Tuple of (restored_count, overridden_count).
        """
        if not self.persist_args:
            return 0, 0

        config_overrides = checkpoint.get("config_overrides", {})
        if not config_overrides:
            return 0, 0

        # Normalize legacy n_trials format
        normalize_n_trials_in_overrides(config_overrides)

        return merge_args_with_saved(
            args_obj,
            config_overrides,
            non_persistent=self.args_exclude,
            argv=argv or sys.argv,
        )

    def handle_n_trials_extension(
        self,
        current_n_trials: int,
        checkpoint: Dict[str, Any],
        *,
        argv: Optional[List[str]] = None,
    ) -> Tuple[int, bool]:
        """
        Handle n_trials extension for resume scenarios.

        When resuming:
        - If current > saved: extend (return current, extended=True)
        - If current < saved and CLI specified: respect lower value
        - If current < saved and not specified: use saved value

        Args:
            current_n_trials: Currently configured n_trials.
            checkpoint: Loaded checkpoint dictionary.
            argv: Command line arguments.

        Returns:
            Tuple of (final_n_trials, was_extended).
        """
        config_overrides = checkpoint.get("config_overrides", {})
        saved_n_trials = config_overrides.get("args.n_trials")

        cli_specified = was_arg_specified("n_trials", argv or sys.argv)

        return extend_or_align_n_trials(
            current_n_trials,
            saved_n_trials,
            cli_specified=cli_specified,
        )

    def build_resume_command(
        self,
        original_argv: Optional[List[str]] = None,
        default_script: str = "scripts/hpo.py",
        *,
        use_local: bool = True,
    ) -> str:
        """
        Build a resume command for display to user.

        Args:
            original_argv: Original command line arguments.
            default_script: Default script path if not in argv.
            use_local: Whether to generate local resume command.

        Returns:
            Shell command string for resuming.
        """
        argv = original_argv or sys.argv
        script = argv[0] if argv else default_script

        parts = ["python", script]

        # Add preserved arguments
        if self.wandb_project:
            parts.extend(["--wandb", self.wandb_project])
        if self.study_name:
            parts.extend(["--study-name", self.study_name])

        # Preserve trial-steps if present
        trial_steps = parse_cli_arg(argv, "trial-steps")
        if trial_steps:
            parts.extend(["--trial-steps", trial_steps])

        # Add resume-from
        if use_local and self.local_checkpoint_dir:
            parts.extend(["--resume-from", str(self.local_checkpoint_dir)])
        else:
            parts.extend(["--resume-from", "latest"])

        return " ".join(parts)

    def build_wandb_resume_command(
        self,
        original_argv: Optional[List[str]] = None,
        default_script: str = "scripts/hpo.py",
    ) -> str:
        """Build a WandB resume command."""
        return self.build_resume_command(
            original_argv,
            default_script,
            use_local=False,
        )

    def build_local_resume_command(
        self,
        original_argv: Optional[List[str]] = None,
        default_script: str = "scripts/hpo.py",
    ) -> str:
        """Build a local resume command."""
        return self.build_resume_command(
            original_argv,
            default_script,
            use_local=True,
        )

    def save_checkpoint(
        self,
        study,
        total_trials_completed: int,
        sampler_name: str,
        pruner_name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        *,
        last_wandb_upload_trial_count: Optional[int] = None,
    ) -> bool:
        """
        Save a checkpoint using the configured persistence.

        Args:
            study: Optuna study object.
            total_trials_completed: Number of completed trials.
            sampler_name: Name of the sampler.
            pruner_name: Name of the pruner.
            config_overrides: Configuration overrides to persist.
            last_wandb_upload_trial_count: Trial count at last WandB upload.

        Returns:
            True if save succeeded, False otherwise.
        """
        if not self._persistence:
            logger.warning("No persistence configured, cannot save checkpoint")
            return False

        metadata = StudyMetadata(
            study_name=self.study_name or "optuna_study",
            total_trials_completed=total_trials_completed,
            sampler_name=sampler_name,
            pruner_name=pruner_name,
            config_overrides=config_overrides or {},
            last_wandb_upload_trial_count=last_wandb_upload_trial_count,
        )

        return self._persistence.save_study(study, metadata)

    def get_trials_since_last_upload(
        self,
        current_trials: int,
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Calculate trials since last WandB upload.

        Used to track when to upload to WandB when restart_every_trial=True.

        Args:
            current_trials: Current total trials completed.
            checkpoint: Loaded checkpoint with last upload info.

        Returns:
            Number of trials since last upload.
        """
        if not checkpoint:
            return current_trials

        last_upload = checkpoint.get("last_wandb_upload_trial_count")
        if last_upload is None:
            return current_trials

        return current_trials - last_upload

    def should_upload_to_wandb(
        self,
        current_trials: int,
        save_every_n_trials: int,
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if it's time to upload to WandB.

        Args:
            current_trials: Current total trials completed.
            save_every_n_trials: Upload frequency.
            checkpoint: Loaded checkpoint with last upload info.

        Returns:
            True if should upload, False otherwise.
        """
        if not self.wandb_project:
            return False

        trials_since = self.get_trials_since_last_upload(current_trials, checkpoint)
        return trials_since >= save_every_n_trials


def create_resume_manager_from_args(
    args,
    *,
    default_checkpoint_dir: Optional[Path] = None,
) -> ResumeManager:
    """
    Create a ResumeManager from parsed command-line arguments.

    This helper function extracts relevant arguments and creates a
    properly configured ResumeManager.

    Args:
        args: Parsed argparse namespace with HPO arguments.
        default_checkpoint_dir: Default checkpoint directory if not in args.

    Returns:
        Configured ResumeManager instance.
    """
    # Extract checkpoint directory
    checkpoint_dir = getattr(args, 'checkpoint_dir', None)
    if checkpoint_dir is None:
        checkpoint_dir = default_checkpoint_dir

    # Extract other parameters
    wandb_project = getattr(args, 'wandb', None)
    study_name = getattr(args, 'study_name', None)
    persist_args = getattr(args, 'persist_args', True)

    # Build args_exclude set
    args_exclude = {'resume_from', 'study_name'}
    if hasattr(args, 'args_exclude') and args.args_exclude:
        args_exclude.update(args.args_exclude)

    return ResumeManager(
        local_checkpoint_dir=checkpoint_dir,
        wandb_project=wandb_project,
        study_name=study_name,
        persist_args=persist_args,
        args_exclude=args_exclude,
    )
