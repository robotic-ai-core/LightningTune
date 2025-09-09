"""
Pausible Optuna optimizer with WandB artifact storage for checkpointing.

This module provides a wrapper around OptunaDrivenOptimizer that adds:
1. Pause/resume capability at trial boundaries
2. WandB artifact storage for study persistence
3. Periodic checkpointing during optimization
4. Study integrity verification

The optimizer ensures clean trial boundaries - only COMPLETE and PRUNED trials
are saved, never RUNNING or WAITING trials.
"""

import os
import pickle
import tempfile
import logging
from typing import Optional, Dict, Any, Callable, Union, Type, List
import sys
from pathlib import Path
import threading
import time

import optuna
import wandb
from lightning import LightningModule
from lightning.pytorch.callbacks import Callback

from .optimizer import OptunaDrivenOptimizer
from .optimizer_reflow import ReflowOptunaDrivenOptimizer
from .factories import create_sampler, create_pruner
try:
    # Ensure Reflow package is importable when used as a submodule
    reflow_path = Path(__file__).parent.parent.parent.parent / "LightningReflow"
    if reflow_path.exists():
        sys.path.insert(0, str(reflow_path))
    # Prefer Reflow's robust keyboard handler to improve TTY restore and Ctrl+C behavior
    from lightning_reflow.callbacks.pause.improved_keyboard_handler import (
        create_improved_keyboard_handler,
    )
except Exception:  # Fallback if Reflow is not available
    create_improved_keyboard_handler = None  # type: ignore

logger = logging.getLogger(__name__)


class PausibleOptunaOptimizer:
    """
    Wrapper around OptunaDrivenOptimizer with pause/resume via WandB artifacts.
    
    This optimizer adds pausibility and checkpointing to standard Optuna optimization:
    - Press 'p' to pause at the next trial boundary
    - Automatically saves study state to WandB artifacts
    - Resume from any saved checkpoint
    - Handles PRUNED trials correctly as valid outcomes
    
    Example:
        >>> optimizer = PausibleOptunaOptimizer(
        ...     base_config="config.yaml",
        ...     search_space=lambda trial: {...},
        ...     wandb_project="my-project",
        ...     study_name="my-study",
        ...     save_every_n_trials=5
        ... )
        >>> # Run optimization (press 'p' to pause)
        >>> study = optimizer.optimize(n_trials=100)
        >>> # Resume later
        >>> study = optimizer.optimize(n_trials=100, resume_from="latest")
    """
    
    def __init__(
        self,
        base_config: Union[str, Dict[str, Any]],
        search_space: Union[Callable[[optuna.Trial], Dict[str, Any]], Any],
        model_class: Type[LightningModule],
        datamodule_class: Optional[Type] = None,
        wandb_project: Optional[str] = None,
        study_name: str = "optuna_study",
        sampler_name: str = "tpe",
        pruner_name: str = "median",
        save_every_n_trials: int = 10,
        enable_pause: bool = True,
        use_reflow: bool = False,  # Option to use LightningReflow
        **optimizer_kwargs
    ):
        """
        Initialize the pausible optimizer.
        
        Args:
            base_config: Base configuration file path or dict
            search_space: Function or OptunaSearchSpace defining parameters to optimize
            model_class: PyTorch Lightning module class
            datamodule_class: Optional PyTorch Lightning datamodule class
            wandb_project: WandB project name for artifact storage (None disables WandB)
            study_name: Name for the study (used in WandB artifacts)
            sampler_name: Name of Optuna sampler to use
            pruner_name: Name of Optuna pruner to use
            save_every_n_trials: Save checkpoint every N trials
            enable_pause: Whether to enable 'p' key pause functionality
            use_reflow: Whether to use LightningReflow for better environment setup and compilation
            **optimizer_kwargs: Additional arguments for OptunaDrivenOptimizer
        """
        self.base_config = base_config
        self.search_space = search_space
        self.model_class = model_class
        self.datamodule_class = datamodule_class
        self.wandb_project = wandb_project
        self.study_name = study_name
        self.sampler_name = sampler_name
        self.pruner_name = pruner_name
        self.save_every_n_trials = save_every_n_trials
        self.enable_pause = enable_pause
        self.use_reflow = use_reflow
        self.optimizer_kwargs = optimizer_kwargs
        # Preserve original CLI argv to build accurate resume commands (Reflow-style)
        try:
            self._original_argv: List[str] = sys.argv.copy()
        except Exception:
            self._original_argv = []
        
        # Track progress
        self.total_trials_completed = 0
        self.should_pause = False
        
        # Setup keyboard handler for 'p' key pause (robust terminal handling)
        self.keyboard_handler = None
        # Backward-compatibility shim for tests expecting `keyboard_monitor`
        # (they patch this attribute; optimize() no longer uses it directly)
        self.keyboard_monitor = None
        self._pause_requested: bool = False
        if enable_pause:
            if create_improved_keyboard_handler is not None:
                self.keyboard_handler = create_improved_keyboard_handler()
            else:
                self.keyboard_handler = None
    
    
    def _verify_study_integrity(self, study: optuna.Study) -> tuple[bool, int, str]:
        """
        Verify study integrity and count finished trials.
        
        Returns:
            (is_valid, finished_count, message)
            
        finished_count includes COMPLETE and PRUNED trials (both are valid outcomes).
        """
        completed_trials = [t for t in study.trials 
                          if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials 
                        if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in study.trials 
                        if t.state == optuna.trial.TrialState.FAIL]
        running_trials = [t for t in study.trials 
                         if t.state == optuna.trial.TrialState.RUNNING]
        waiting_trials = [t for t in study.trials 
                         if t.state == optuna.trial.TrialState.WAITING]
        
        # Both COMPLETE and PRUNED are valid finished trials
        finished_count = len(completed_trials) + len(pruned_trials)
        
        # Check for incomplete trials (RUNNING or WAITING are not acceptable)
        if running_trials or waiting_trials:
            incomplete_count = len(running_trials) + len(waiting_trials)
            message = (f"Study has {incomplete_count} incomplete trial(s) "
                      f"({len(running_trials)} running, {len(waiting_trials)} waiting)")
            return False, finished_count, message
        
        # Study is valid - report statistics
        message = (f"Study has {len(completed_trials)} completed, "
                  f"{len(pruned_trials)} pruned, {len(failed_trials)} failed trials")
        return True, finished_count, message
    
    def save_study_to_wandb(self, study: optuna.Study, expected_trials: int) -> bool:
        """
        Save study state to WandB as an artifact.
        
        Only saves if the study is in a valid state (no incomplete trials).
        
        Args:
            study: Optuna study to save
            expected_trials: Expected number of completed trials
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.wandb_project:
            logger.debug("WandB project not configured, skipping save")
            return False
        
        # Verify study integrity
        is_valid, finished_count, message = self._verify_study_integrity(study)
        
        if not is_valid:
            logger.warning(f"⚠️  Cannot save study: {message}")
            logger.warning("   Incomplete trials must finish before saving.")
            return False
        
        # Verify expected count matches actual
        if finished_count != expected_trials:
            logger.warning(
                f"⚠️  Expected {expected_trials} finished trials but found {finished_count}. "
                f"Saving with actual count."
            )
            trials_completed = finished_count
        else:
            trials_completed = expected_trials
        
        logger.info(f"💾 Saving study: {message}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            session_info = {
                "study": study,
                "total_trials_completed": trials_completed,
                "sampler_name": self.sampler_name,
                "pruner_name": self.pruner_name,
                "study_name": self.study_name,
            }
            
            pickle.dump(session_info, tmp, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.flush()
            os.fsync(tmp.fileno())
            
            # Verify save
            tmp.seek(0)
            try:
                loaded_info = pickle.load(tmp)
                # Double-check the loaded study
                loaded_study = loaded_info["study"]
                loaded_finished = len([t for t in loaded_study.trials 
                                      if t.state in [optuna.trial.TrialState.COMPLETE,
                                                    optuna.trial.TrialState.PRUNED]])
                if loaded_finished != trials_completed:
                    logger.error(f"Verification failed: saved {trials_completed} but loaded {loaded_finished}")
                    return False
            except Exception as e:
                logger.error(f"Failed to verify saved study: {e}")
                return False
            
            # Upload to WandB
            run = wandb.init(
                project=self.wandb_project,
                job_type="hpo_checkpoint",
            )
            artifact = wandb.Artifact(
                f"{self.study_name}_checkpoint",
                type="optuna_study"
            )
            artifact.add_file(tmp.name, name="study.pkl")
            artifact.metadata = {
                "total_finished_trials": trials_completed,
                "completed_trials": len([t for t in study.trials 
                                        if t.state == optuna.trial.TrialState.COMPLETE]),
                "pruned_trials": len([t for t in study.trials 
                                    if t.state == optuna.trial.TrialState.PRUNED]),
                "failed_trials": len([t for t in study.trials 
                                    if t.state == optuna.trial.TrialState.FAIL]),
                "best_value": study.best_value if study.best_trial else None,
                "best_trial_number": study.best_trial.number if study.best_trial else None,
            }
            # Log and wait for artifact to upload
            logged_artifact = run.log_artifact(artifact, aliases=["latest"])
            
            # IMPORTANT: Use wait() to ensure artifact uploads before we exit
            # This blocks until the artifact is fully uploaded to WandB
            logged_artifact.wait()
            
            # Now we can safely finish the run
            run.finish()
            
            logger.info(f"✅ Study saved to WandB: {self.study_name}_checkpoint (v{trials_completed})")
            return True
    
    def load_study_from_wandb(self, version: str = "latest") -> Optional[Dict[str, Any]]:
        """
        Load study state from WandB artifact.
        
        Args:
            version: Artifact version to load (e.g., "latest", "v3")
            
        Returns:
            Session info dict if found, None otherwise
        """
        if not self.wandb_project:
            logger.debug("WandB project not configured, cannot load")
            return None
            
        api = wandb.Api()
        
        try:
            artifact = api.artifact(
                f"{self.wandb_project}/{self.study_name}_checkpoint:{version}"
            )
        except wandb.errors.CommError as e:
            logger.info(f"No existing study found in WandB: {e}")
            return None
        
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact.download(tmpdir)
            file_path = os.path.join(tmpdir, "study.pkl")
            
            if not os.path.exists(file_path):
                logger.error("study.pkl not found in artifact")
                return None
            
            with open(file_path, 'rb') as f:
                try:
                    session_info = pickle.load(f)
                    logger.info(f"✅ Loaded study with {session_info['total_trials_completed']} finished trials")
                    return session_info
                except Exception as e:
                    logger.error(f"Failed to load study: {e}")
                    return None
    
    def optimize(
        self,
        n_trials: int,
        resume_from: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Callback]] = None,
        storage: Optional[str] = None,
        **kwargs
    ) -> optuna.Study:
        """
        Run optimization with periodic saves and resume capability.
        
        Ensures that only finished trials (COMPLETE/PRUNED) are saved to WandB.
        If interrupted mid-trial, the incomplete trial is discarded.
        
        Args:
            n_trials: Number of trials to run
            resume_from: WandB artifact version to resume from (e.g., "latest", "v3")
            config_overrides: Optional config overrides for all trials
            callbacks: Additional Lightning callbacks
            storage: Optional Optuna storage URL (for distributed optimization)
            **kwargs: Additional arguments passed to OptunaDrivenOptimizer
            
        Returns:
            Optuna study with results
        """
        # Try to resume from WandB
        session_info = None
        if resume_from and self.wandb_project:
            session_info = self.load_study_from_wandb(resume_from)
        
        if session_info:
            study = session_info["study"]
            self.total_trials_completed = session_info["total_trials_completed"]
            self.should_pause = False  # Reset pause flag when resuming
            progress_percent = (self.total_trials_completed / n_trials) * 100
            remaining = n_trials - self.total_trials_completed
            logger.info(f"\n{'='*60}")
            logger.info(f"📂 RESUMING OPTIMIZATION")
            logger.info(f"Progress: {self.total_trials_completed}/{n_trials} trials already complete ({progress_percent:.1f}%)")
            logger.info(f"Remaining: {remaining} trials to run")
            logger.info(f"{'='*60}")
            
            # Verify study integrity - count finished trials (COMPLETE + PRUNED)
            finished_count = len([t for t in study.trials 
                                if t.state in [optuna.trial.TrialState.COMPLETE,
                                              optuna.trial.TrialState.PRUNED]])
            if finished_count != self.total_trials_completed:
                logger.warning(
                    f"⚠️  Study has {finished_count} finished trials but "
                    f"expected {self.total_trials_completed}. Using actual count."
                )
                self.total_trials_completed = finished_count
        else:
            # Create new study
            # Seed sampler for reproducible HPO sequences when config has a seed
            # Try to extract seed from base_config if it's a file or dict with 'seed_everything'
            seed_value = None
            try:
                if isinstance(self.base_config, dict):
                    seed_value = self.base_config.get('seed_everything', None)
                else:
                    # If it's a path-like string, attempt to read YAML and pull seed
                    import yaml
                    from pathlib import Path
                    cfg_path = Path(self.base_config)
                    if cfg_path.exists():
                        with cfg_path.open('r') as f:
                            cfg = yaml.safe_load(f)
                            if isinstance(cfg, dict):
                                seed_value = cfg.get('seed_everything', None)
            except Exception:
                seed_value = None

            sampler = create_sampler(self.sampler_name, seed=seed_value)
            pruner = create_pruner(self.pruner_name)
            
            study = optuna.create_study(
                study_name=self.study_name,
                sampler=sampler,
                pruner=pruner,
                direction=self.optimizer_kwargs.get("direction", "minimize"),
                storage=storage
            )
            self.should_pause = False  # Ensure pause flag is reset for new study
            logger.info(f"\n{'='*60}")
            logger.info(f"🆕 STARTING NEW OPTIMIZATION")
            logger.info(f"Study name: {self.study_name}")
            logger.info(f"Total trials to run: {n_trials}")
            logger.info(f"Sampler: {self.sampler_name}")
            logger.info(f"Pruner: {self.pruner_name}")
            logger.info(f"Direction: {self.optimizer_kwargs.get('direction', 'minimize')}")
            if self.wandb_project:
                logger.info(f"WandB project: {self.wandb_project}")
                logger.info(f"Checkpoint frequency: every {self.save_every_n_trials} trials")
            logger.info(f"{'='*60}")
        
        # Merge optimizer kwargs
        opt_kwargs = self.optimizer_kwargs.copy()
        opt_kwargs.update(kwargs)
        
        # Extract direction to avoid duplicate argument
        direction = opt_kwargs.pop("direction", "minimize")
        
        # Create optimizer (use Reflow version if requested)
        OptimizerClass = ReflowOptunaDrivenOptimizer if self.use_reflow else OptunaDrivenOptimizer
        optimizer = OptimizerClass(
            base_config=self.base_config,
            search_space=self.search_space,
            config_overrides=config_overrides,
            model_class=self.model_class,
            datamodule_class=self.datamodule_class,
            sampler=study.sampler,  # Use study's sampler
            pruner=study.pruner,     # Use study's pruner
            study_name=self.study_name,
            direction=direction,
            n_trials=1,  # We'll run one at a time for checkpointing
            callbacks=callbacks,
            wandb_project=self.wandb_project,
            **opt_kwargs
        )
        
        objective = optimizer.create_objective()
        
        # Start keyboard monitoring if available
        if self.keyboard_handler and hasattr(self.keyboard_handler, 'start_monitoring'):
            try:
                self.keyboard_handler.start_monitoring()
            except Exception:
                logger.info("ℹ️  Keyboard monitoring unavailable, pause functionality disabled")
                self.keyboard_handler = None
        # Start background polling for immediate schedule/cancel feedback
        self._pause_poll_thread = None
        self._polling_active = False
        if self.keyboard_handler and hasattr(self.keyboard_handler, 'get_key'):
            self._start_pause_polling_thread()
        
        # Run trials with periodic saves
        trials_in_batch = 0
        last_saved_trial_count = self.total_trials_completed
        
        while self.total_trials_completed < n_trials and not self.should_pause:
            # Record number of finished trials (COMPLETE + PRUNED) before this trial
            trials_before = len([t for t in study.trials 
                                if t.state in [optuna.trial.TrialState.COMPLETE,
                                              optuna.trial.TrialState.PRUNED]])
            
            # Check for keyboard pause request before starting trial
            if self._update_pause_from_keyboard():
                self.should_pause = True
                logger.info("\n⏸️  Executing pause at trial boundary...")
                if self.wandb_project:
                    logger.info("   Study will be saved to WandB for easy resume")
                break
            
            try:
                # Show clear progress before starting trial
                trial_number = self.total_trials_completed + 1
                progress_percent = (self.total_trials_completed / n_trials) * 100
                logger.info(f"\n{'='*60}")
                logger.info(f"📊 Starting Trial {trial_number} of {n_trials} ({progress_percent:.1f}% complete)")
                logger.info(f"{'='*60}")
                
                # Run single trial with automatic garbage collection
                # gc_after_trial=True ensures memory is cleaned between trials
                study.optimize(objective, n_trials=1, show_progress_bar=False, gc_after_trial=True)
                
                # Check if a new trial was actually finished (COMPLETE or PRUNED)
                trials_after = len([t for t in study.trials 
                                   if t.state in [optuna.trial.TrialState.COMPLETE,
                                                  optuna.trial.TrialState.PRUNED]])
                
                if trials_after > trials_before:
                    # Trial finished (either completed or pruned)
                    self.total_trials_completed = trials_after
                    trials_in_batch += 1
                    
                    # Get the latest trial to check if it was pruned
                    latest_trial = study.trials[-1]
                    status = "✅ COMPLETE" if latest_trial.state == optuna.trial.TrialState.COMPLETE else "⏭️ PRUNED"
                    
                    # Calculate updated progress
                    progress_percent = (self.total_trials_completed / n_trials) * 100
                    remaining_trials = n_trials - self.total_trials_completed
                    
                    # Log progress with clearer formatting
                    logger.info(f"\n{'─'*60}")
                    logger.info(f"Trial {trial_number} Result: {status}")
                    logger.info(f"Progress: {self.total_trials_completed}/{n_trials} trials complete ({progress_percent:.1f}%)")
                    logger.info(f"Remaining: {remaining_trials} trials")
                    
                    if study.best_trial:
                        logger.info(
                            f"Current Best: {study.best_value:.6f} (from trial {study.best_trial.number})"
                        )
                    else:
                        logger.info(f"Current Best: No successful trials yet")
                    logger.info(f"{'─'*60}")
                    
                    # Periodic save (only if we have new finished trials and WandB is configured)
                    if self.wandb_project and trials_in_batch >= self.save_every_n_trials:
                        if self.save_study_to_wandb(study, self.total_trials_completed):
                            last_saved_trial_count = self.total_trials_completed
                        trials_in_batch = 0
                    
                    # Check for pause request after trial completes
                    if self._update_pause_from_keyboard():
                        self.should_pause = True
                        logger.info("\n⏸️  Executing pause after trial completion...")
                        if self.wandb_project:
                            logger.info("   Study will be saved to WandB for easy resume")
                        break
                else:
                    # Trial failed (actual error, not pruning)
                    logger.info(f"\n{'─'*60}")
                    logger.info(f"Trial {trial_number} Result: ❌ FAILED")
                    logger.info(f"Progress: {self.total_trials_completed}/{n_trials} trials complete ({progress_percent:.1f}%)")
                    logger.info(f"{'─'*60}")
                    
                    # Check for pause request after failed trial
                    if self._update_pause_from_keyboard():
                        self.should_pause = True
                        logger.info("\n⏸️  Executing pause after failed trial...")
                        if self.wandb_project:
                            logger.info("   Study will be saved to WandB for easy resume")
                        break
                    
            except KeyboardInterrupt:
                # Clean up keyboard handler before terminating
                if self.keyboard_handler and hasattr(self.keyboard_handler, 'stop_monitoring'):
                    try:
                        self.keyboard_handler.stop_monitoring()
                    except Exception:
                        pass
                logger.info("\n❌ Optimization terminated by user (Ctrl+C)")
                # Ensure the KeyboardInterrupt propagates all the way out
                raise
                
            except Exception as e:
                logger.error(f"Error during trial: {e}")
                
                # Check for pause request even after error
                if self.keyboard_monitor and self.keyboard_monitor.is_pause_requested():
                    self.should_pause = True
                    # Don't clear pause here - let keyboard monitor handle the toggle
                    logger.info("\n⏸️  Executing pause after error...")
                    if self.wandb_project:
                        logger.info("   Study will be saved to WandB for easy resume")
                    # Break out of loop to trigger save logic
                    break
                    
                # Continue with next trial if not pausing
                continue
        
        # Stop keyboard monitoring and clear pause state
        if self.keyboard_handler and hasattr(self.keyboard_handler, 'stop_monitoring'):
            try:
                self.keyboard_handler.stop_monitoring()
            except Exception:
                pass
        self._pause_requested = False
        # Stop background polling thread
        self._stop_pause_polling_thread()
        
        # Handle pause save or final save
        study_was_saved = False
        if self.should_pause and self.wandb_project:
            # ALWAYS save when pause is requested, regardless of whether there are new trials
            # This is critical: pause means user wants to stop and resume later
            logger.info(f"💾 Saving study state for pause (with {self.total_trials_completed} finished trials)")
            study_was_saved = self.save_study_to_wandb(study, self.total_trials_completed)
            if study_was_saved:
                last_saved_trial_count = self.total_trials_completed  # Update for consistency
            else:
                logger.error("⚠️  Failed to save study for pause - checkpoint may be incomplete")
        elif self.wandb_project and self.total_trials_completed > last_saved_trial_count:
            # Regular final save - only if we have new finished trials since last save
            logger.info(f"💾 Saving final state with {self.total_trials_completed} finished trials")
            study_was_saved = self.save_study_to_wandb(study, self.total_trials_completed)
        elif self.wandb_project and not self.should_pause:
            logger.info(f"ℹ️  No new finished trials to save since last checkpoint")
        
        if self.should_pause:
            logger.info(f"\n{'='*60}")
            logger.info(f"⏸️  OPTIMIZATION PAUSED")
            logger.info(f"Progress: {self.total_trials_completed}/{n_trials} trials complete ({(self.total_trials_completed/n_trials)*100:.1f}%)")
            logger.info(f"Remaining: {n_trials - self.total_trials_completed} trials")
            if self.wandb_project:
                if study_was_saved:
                    logger.info(f"\n📝 To resume, run:")
                    # Prefer original CLI (includes all user overrides), then append resume flag
                    if self._original_argv:
                        base_cmd = " ".join(self._original_argv)
                    else:
                        # Fallback to a minimal script name if argv is unavailable
                        base_cmd = "scripts/world_model_hpo_optuna.py --wandb {proj}".format(proj=self.wandb_project)
                    resume_cmd = f"python {base_cmd} --resume-from latest"
                    # Ensure study name is present if customized
                    if self.study_name and self.study_name != "optuna_study" and "--study-name" not in resume_cmd:
                        resume_cmd += f" --study-name {self.study_name}"
                    logger.info(f"   {resume_cmd}")
                else:
                    logger.info(f"⚠️  Failed to save study checkpoint - cannot resume from this point")
                    logger.info(f"   Check logs above for save errors")
            else:
                logger.info(f"⚠️  No WandB project configured - checkpoint not saved")
                logger.info(f"   To enable resume, use --wandb <project-name>")
            logger.info(f"{'='*60}")
        else:
            logger.info(f"\n{'='*60}")
            logger.info(f"✨ OPTIMIZATION COMPLETE!")
            logger.info(f"Total trials run: {self.total_trials_completed}/{n_trials} ({100.0:.1f}%)")
            logger.info(f"{'='*60}")
        
        # Print results (only if we have completed trials)
        try:
            if study.best_trial:
                logger.info(f"Best trial: {study.best_trial.number}")
                logger.info(f"Best value: {study.best_value:.6f}")
                logger.info("Best params:")
                for key, value in study.best_params.items():
                    logger.info(f"  {key}: {value}")
        except ValueError:
            # No completed trials yet
            logger.info("No trials completed successfully yet.")
        
        return study

    # --- Internal helpers -------------------------------------------------

    def _update_pause_from_keyboard(self) -> bool:
        """Poll keyboard handler and toggle pause when 'p' is pressed.

        Returns True if pause is currently requested.
        """
        # If background polling is active, just return current flag
        if getattr(self, '_polling_active', False):
            return self._pause_requested
        try:
            if self.keyboard_handler and hasattr(self.keyboard_handler, 'get_key'):
                key = self.keyboard_handler.get_key()
                if key and str(key).lower() == 'p':
                    self._pause_requested = not self._pause_requested
                    if self._pause_requested:
                        logger.info("\n⏸️  Pause SCHEDULED ('p' pressed)")
                    else:
                        logger.info("\n❌ Pause CANCELLED ('p' pressed again)")
        except Exception:
            pass
        return self._pause_requested

    def _build_resume_command(self) -> str:
        """Construct a resume command string based on original argv and study settings."""
        if self._original_argv:
            base_cmd = " ".join(self._original_argv)
        else:
            # Reasonable fallback
            script = "world_model_hpo_optuna.py"
            base_cmd = f"{script} --wandb {self.wandb_project or 'my-project'}"
        cmd = f"python {base_cmd} --resume-from latest"
        if self.study_name and self.study_name != "optuna_study" and "--study-name" not in cmd:
            cmd += f" --study-name {self.study_name}"
        return cmd

    def _start_pause_polling_thread(self) -> None:
        """Start a lightweight background thread to poll keyboard input for 'p'."""
        if self._pause_poll_thread and self._pause_poll_thread.is_alive():
            return
        self._polling_active = True
        self._pause_poll_thread = threading.Thread(target=self._pause_input_loop, daemon=True, name="PauseInputWatcher")
        self._pause_poll_thread.start()

    def _stop_pause_polling_thread(self) -> None:
        """Stop the background polling thread if running."""
        if getattr(self, '_polling_active', False):
            self._polling_active = False
        t = getattr(self, '_pause_poll_thread', None)
        if t and t.is_alive():
            try:
                t.join(timeout=1.0)
            except Exception:
                pass

    def _pause_input_loop(self) -> None:
        """Continuously poll keyboard handler for immediate schedule/cancel feedback."""
        last_state = self._pause_requested
        while getattr(self, '_polling_active', False):
            try:
                if self.keyboard_handler and hasattr(self.keyboard_handler, 'get_key'):
                    key = self.keyboard_handler.get_key()
                    if key and str(key).lower() == 'p':
                        # Toggle pause state
                        self._pause_requested = not self._pause_requested
                        if self._pause_requested and not last_state:
                            logger.info("\n⏸️  Pause SCHEDULED ('p' pressed)")
                        elif (not self._pause_requested) and last_state:
                            logger.info("\n❌ Pause CANCELLED ('p' pressed again)")
                        last_state = self._pause_requested
            except Exception:
                # Ignore read errors
                pass
            time.sleep(0.05)