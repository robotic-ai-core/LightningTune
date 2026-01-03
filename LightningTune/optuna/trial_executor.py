"""
Trial executor for HPO optimization loops.

This module provides the TrialExecutor class that manages the execution
of Optuna trials with support for:
- Pause/resume at trial boundaries
- Checkpoint saving
- Keyboard interrupt handling
- Memory cleanup between trials
"""

from __future__ import annotations

import gc
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import optuna

if TYPE_CHECKING:
    from ..input.keyboard_handler import HPOKeyboardHandler
    from ..persistence import BasePersistence, StudyMetadata
    from ..resume import ResumeManager

logger = logging.getLogger(__name__)


@dataclass
class TrialExecutorConfig:
    """Configuration for trial execution.

    Attributes:
        n_trials: Total number of trials to run.
        save_every_n_trials: Save checkpoint every N trials.
        restart_on_save: Exit for subprocess restart after saving.
        restart_every_trial: When restart_on_save, restart after every trial.
        enable_pause: Enable 'p' key pause functionality.
        cleanup_between_trials: Run garbage collection between trials.
    """
    n_trials: int = 50
    save_every_n_trials: int = 10
    restart_on_save: bool = False
    restart_every_trial: bool = True
    enable_pause: bool = True
    cleanup_between_trials: bool = True


@dataclass
class TrialExecutorState:
    """Runtime state for trial execution.

    Attributes:
        trials_completed: Total trials completed this session.
        trials_completed_this_run: Trials completed in current run (since last restart).
        pause_requested: Whether pause was requested.
        quit_requested: Whether quit was requested.
        should_exit_for_restart: Whether we should exit for subprocess restart.
        last_wandb_upload_count: Trial count at last WandB upload.
    """
    trials_completed: int = 0
    trials_completed_this_run: int = 0
    pause_requested: bool = False
    quit_requested: bool = False
    should_exit_for_restart: bool = False
    last_wandb_upload_count: int = 0


class TrialExecutor:
    """
    Executes Optuna trials with pause/resume and checkpoint support.

    This class extracts the trial loop logic from PausibleOptunaOptimizer
    into a reusable component that handles:
    - Running the optimization loop
    - Pause/resume at trial boundaries
    - Periodic checkpointing
    - Keyboard interrupt handling
    - Memory cleanup

    Example:
        >>> executor = TrialExecutor(
        ...     study=study,
        ...     objective=my_objective,
        ...     config=TrialExecutorConfig(n_trials=50),
        ... )
        >>> executor.run()
    """

    def __init__(
        self,
        study: optuna.Study,
        objective: Callable[[optuna.Trial], float],
        config: TrialExecutorConfig,
        *,
        keyboard_handler: Optional["HPOKeyboardHandler"] = None,
        persistence: Optional["BasePersistence"] = None,
        resume_manager: Optional["ResumeManager"] = None,
        on_trial_complete: Optional[Callable[[optuna.Trial, int], None]] = None,
        on_save: Optional[Callable[[int], None]] = None,
        on_pause: Optional[Callable[[], None]] = None,
        metadata_builder: Optional[Callable[[int], "StudyMetadata"]] = None,
    ):
        """
        Initialize trial executor.

        Args:
            study: Optuna study to optimize.
            objective: Objective function for trials.
            config: Execution configuration.
            keyboard_handler: Optional keyboard handler for pause control.
            persistence: Optional persistence backend for checkpoints.
            resume_manager: Optional resume manager for checkpoint handling.
            on_trial_complete: Callback after each trial completes.
            on_save: Callback after checkpoint save.
            on_pause: Callback when paused.
            metadata_builder: Function to build StudyMetadata for saves.
        """
        self.study = study
        self.objective = objective
        self.config = config
        self.keyboard_handler = keyboard_handler
        self.persistence = persistence
        self.resume_manager = resume_manager

        # Callbacks
        self._on_trial_complete = on_trial_complete
        self._on_save = on_save
        self._on_pause = on_pause
        self._metadata_builder = metadata_builder

        # State
        self.state = TrialExecutorState()

        # Signal handling
        self._original_sigint = None
        self._interrupted = False

    @property
    def should_pause(self) -> bool:
        """Check if pause is requested."""
        if self.keyboard_handler:
            return self.keyboard_handler.pause_requested
        return self.state.pause_requested

    @should_pause.setter
    def should_pause(self, value: bool):
        """Set pause state."""
        if self.keyboard_handler:
            self.keyboard_handler.pause_requested = value
        self.state.pause_requested = value

    @property
    def should_quit(self) -> bool:
        """Check if quit is requested."""
        if self.keyboard_handler:
            return self.keyboard_handler.quit_requested
        return self.state.quit_requested

    def run(
        self,
        *,
        initial_trials_completed: int = 0,
        last_wandb_upload_count: int = 0,
    ) -> optuna.Study:
        """
        Run the trial execution loop.

        Args:
            initial_trials_completed: Number of trials already completed (from resume).
            last_wandb_upload_count: Trial count at last WandB upload.

        Returns:
            The optimized study.
        """
        self.state.trials_completed = initial_trials_completed
        self.state.last_wandb_upload_count = last_wandb_upload_count
        self.state.trials_completed_this_run = 0

        # Setup keyboard handler
        if self.keyboard_handler and self.config.enable_pause:
            self.keyboard_handler.start()

        # Setup signal handler
        self._setup_signal_handler()

        try:
            return self._run_loop()
        finally:
            self._cleanup()

    def _run_loop(self) -> optuna.Study:
        """Main trial execution loop."""
        remaining_trials = self.config.n_trials - self.state.trials_completed

        logger.info(f"▶️  Starting optimization: {remaining_trials} trials remaining")

        while self.state.trials_completed < self.config.n_trials:
            # Check for pause/quit before starting trial
            if self._check_stop_conditions():
                break

            # Run single trial
            try:
                self._run_single_trial()
            except KeyboardInterrupt:
                logger.info("\n⏸️  KeyboardInterrupt - pausing at trial boundary")
                self.should_pause = True
                break
            except optuna.TrialPruned:
                # Pruned trials still count as completed
                self.state.trials_completed += 1
                self.state.trials_completed_this_run += 1
            except Exception as e:
                logger.error(f"Trial failed with error: {e}")
                # Continue to next trial
                self.state.trials_completed += 1
                self.state.trials_completed_this_run += 1

            # Post-trial actions
            self._on_trial_finished()

            # Check if we should exit for restart
            if self._check_restart_condition():
                break

        return self.study

    def _run_single_trial(self):
        """Run a single trial."""
        trial = self.study.ask()
        try:
            value = self.objective(trial)
            self.study.tell(trial, value)
            self.state.trials_completed += 1
            self.state.trials_completed_this_run += 1

            # Callback
            if self._on_trial_complete:
                self._on_trial_complete(trial, self.state.trials_completed)

        except optuna.TrialPruned:
            self.study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            raise
        except Exception as e:
            # Mark trial as failed
            try:
                self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
            except Exception:
                pass
            raise

    def _on_trial_finished(self):
        """Actions to perform after each trial."""
        # Memory cleanup
        if self.config.cleanup_between_trials:
            self._cleanup_memory()

        # Check if we should save
        if self._should_save_checkpoint():
            self._save_checkpoint()

    def _check_stop_conditions(self) -> bool:
        """Check if we should stop the loop."""
        if self._interrupted:
            logger.info("⏸️  Interrupted - stopping at trial boundary")
            return True

        if self.should_pause:
            logger.info("⏸️  Pause requested - stopping at trial boundary")
            self._handle_pause()
            return True

        if self.should_quit:
            logger.info("🛑 Quit requested - stopping")
            return True

        return False

    def _check_restart_condition(self) -> bool:
        """Check if we should exit for subprocess restart."""
        if not self.config.restart_on_save:
            return False

        if self.config.restart_every_trial:
            # Restart after every trial for memory isolation
            self.state.should_exit_for_restart = True
            return True

        return False

    def _should_save_checkpoint(self) -> bool:
        """Check if it's time to save a checkpoint."""
        trials_since_upload = (
            self.state.trials_completed - self.state.last_wandb_upload_count
        )
        return trials_since_upload >= self.config.save_every_n_trials

    def _save_checkpoint(self):
        """Save a checkpoint."""
        if not self.persistence:
            return

        if not self._metadata_builder:
            logger.warning("No metadata builder configured, skipping checkpoint")
            return

        try:
            metadata = self._metadata_builder(self.state.trials_completed)
            success = self.persistence.save_study(self.study, metadata)

            if success:
                self.state.last_wandb_upload_count = self.state.trials_completed
                if self._on_save:
                    self._on_save(self.state.trials_completed)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _handle_pause(self):
        """Handle pause request."""
        # Save before pausing
        self._save_checkpoint()

        if self._on_pause:
            self._on_pause()

    def _cleanup_memory(self):
        """Clean up memory between trials."""
        gc.collect()

        # Try CUDA cleanup if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    def _setup_signal_handler(self):
        """Setup signal handler for graceful interruption."""
        def signal_handler(signum, frame):
            self._interrupted = True
            logger.info("\n⏸️  Signal received - will pause at trial boundary")

        self._original_sigint = signal.signal(signal.SIGINT, signal_handler)

    def _cleanup(self):
        """Cleanup after execution."""
        # Restore original signal handler
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)

        # Stop keyboard handler
        if self.keyboard_handler:
            self.keyboard_handler.stop()


class SimpleTrialExecutor:
    """
    Simplified trial executor without pause/persistence support.

    Use this for basic optimization loops that don't need interactive
    pause or checkpointing features.

    Example:
        >>> executor = SimpleTrialExecutor(study, objective, n_trials=50)
        >>> executor.run()
    """

    def __init__(
        self,
        study: optuna.Study,
        objective: Callable[[optuna.Trial], float],
        n_trials: int,
        *,
        cleanup_between_trials: bool = True,
        on_trial_complete: Optional[Callable[[optuna.Trial, int], None]] = None,
    ):
        """
        Initialize simple trial executor.

        Args:
            study: Optuna study.
            objective: Objective function.
            n_trials: Number of trials to run.
            cleanup_between_trials: Run GC between trials.
            on_trial_complete: Callback after each trial.
        """
        self.study = study
        self.objective = objective
        self.n_trials = n_trials
        self.cleanup_between_trials = cleanup_between_trials
        self._on_trial_complete = on_trial_complete

    def run(self) -> optuna.Study:
        """Run the optimization loop."""
        for i in range(self.n_trials):
            trial = self.study.ask()
            try:
                value = self.objective(trial)
                self.study.tell(trial, value)
            except optuna.TrialPruned:
                self.study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            except Exception:
                self.study.tell(trial, state=optuna.trial.TrialState.FAIL)

            if self._on_trial_complete:
                self._on_trial_complete(trial, i + 1)

            if self.cleanup_between_trials:
                gc.collect()

        return self.study
