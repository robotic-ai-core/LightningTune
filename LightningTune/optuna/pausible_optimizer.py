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
import signal
import logging
from typing import Optional, Dict, Any, Callable, Union, Type, List

import optuna
import wandb
from lightning import LightningModule
from lightning.pytorch.callbacks import Callback

from .optimizer import OptunaDrivenOptimizer
from .factories import create_sampler, create_pruner

logger = logging.getLogger(__name__)


class PausibleOptunaOptimizer:
    """
    Wrapper around OptunaDrivenOptimizer with pause/resume via WandB artifacts.
    
    This optimizer adds pausibility and checkpointing to standard Optuna optimization:
    - Press Ctrl+C to pause at the next trial boundary
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
        >>> # Run optimization (press Ctrl+C to pause)
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
            enable_pause: Whether to enable Ctrl+C pause functionality
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
        self.optimizer_kwargs = optimizer_kwargs
        
        # Track progress
        self.total_trials_completed = 0
        self.should_pause = False
        
        # Setup signal handler for graceful pause (if enabled)
        if enable_pause:
            signal.signal(signal.SIGINT, self._handle_pause_signal)
    
    def _handle_pause_signal(self, signum, frame):
        """Handle Ctrl+C to pause optimization."""
        logger.info("\n⏸️  Pause requested. Saving state after current trial...")
        self.should_pause = True
    
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
                reinit=True
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
            run.log_artifact(artifact)
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
            logger.info(f"📂 Resuming from {self.total_trials_completed} finished trials")
            
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
            sampler = create_sampler(self.sampler_name)
            pruner = create_pruner(self.pruner_name)
            
            study = optuna.create_study(
                study_name=self.study_name,
                sampler=sampler,
                pruner=pruner,
                direction=self.optimizer_kwargs.get("direction", "minimize"),
                storage=storage
            )
            logger.info("🆕 Created new study")
        
        # Merge optimizer kwargs
        opt_kwargs = self.optimizer_kwargs.copy()
        opt_kwargs.update(kwargs)
        
        # Create optimizer
        optimizer = OptunaDrivenOptimizer(
            base_config=self.base_config,
            search_space=self.search_space,
            config_overrides=config_overrides,
            model_class=self.model_class,
            datamodule_class=self.datamodule_class,
            sampler=study.sampler,  # Use study's sampler
            pruner=study.pruner,     # Use study's pruner
            study_name=self.study_name,
            direction=opt_kwargs.get("direction", "minimize"),
            n_trials=1,  # We'll run one at a time for checkpointing
            callbacks=callbacks,
            wandb_project=self.wandb_project,
            **opt_kwargs
        )
        
        objective = optimizer.create_objective()
        
        # Run trials with periodic saves
        trials_in_batch = 0
        last_saved_trial_count = self.total_trials_completed
        
        while self.total_trials_completed < n_trials and not self.should_pause:
            # Record number of finished trials (COMPLETE + PRUNED) before this trial
            trials_before = len([t for t in study.trials 
                                if t.state in [optuna.trial.TrialState.COMPLETE,
                                              optuna.trial.TrialState.PRUNED]])
            
            try:
                # Run single trial
                study.optimize(objective, n_trials=1, show_progress_bar=False)
                
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
                    status = "completed" if latest_trial.state == optuna.trial.TrialState.COMPLETE else "pruned"
                    
                    # Log progress
                    if study.best_trial:
                        logger.info(
                            f"Trial {self.total_trials_completed}/{n_trials} ({status}) | "
                            f"Best: {study.best_value:.6f} (trial {study.best_trial.number})"
                        )
                    else:
                        logger.info(f"Trial {self.total_trials_completed}/{n_trials} ({status})")
                    
                    # Periodic save (only if we have new finished trials and WandB is configured)
                    if self.wandb_project and trials_in_batch >= self.save_every_n_trials:
                        if self.save_study_to_wandb(study, self.total_trials_completed):
                            last_saved_trial_count = self.total_trials_completed
                        trials_in_batch = 0
                else:
                    # Trial failed (actual error, not pruning)
                    logger.info(f"Trial failed with error")
                    
            except KeyboardInterrupt:
                if not self.enable_pause:
                    raise  # Re-raise if pause is disabled
                    
                # Handle Ctrl+C gracefully
                logger.info("\n⏸️  Interrupt received. Checking for incomplete trials...")
                self.should_pause = True
                
                # Check if we have any incomplete trials
                incomplete_trials = [t for t in study.trials 
                                   if t.state not in [optuna.trial.TrialState.COMPLETE,
                                                      optuna.trial.TrialState.PRUNED,
                                                      optuna.trial.TrialState.FAIL]]
                if incomplete_trials:
                    logger.warning(f"⚠️  Found {len(incomplete_trials)} incomplete trial(s). These will not be saved.")
                
                # Ensure total_trials_completed reflects only finished trials (COMPLETE + PRUNED)
                self.total_trials_completed = len([t for t in study.trials 
                                                  if t.state in [optuna.trial.TrialState.COMPLETE,
                                                                 optuna.trial.TrialState.PRUNED]])
                break
                
            except Exception as e:
                logger.error(f"Error during trial: {e}")
                # Continue with next trial
                continue
        
        # Final save - only if we have new finished trials since last save
        if self.wandb_project and self.total_trials_completed > last_saved_trial_count:
            logger.info(f"💾 Saving final state with {self.total_trials_completed} finished trials")
            self.save_study_to_wandb(study, self.total_trials_completed)
        elif self.wandb_project:
            logger.info(f"ℹ️  No new finished trials to save since last checkpoint")
        
        if self.should_pause:
            logger.info(f"\n⏸️  Paused after {self.total_trials_completed} finished trials")
            logger.info(f"Resume with: --resume-from latest")
        else:
            logger.info(f"\n✨ Optimization complete!")
        
        # Print results
        if study.best_trial:
            logger.info(f"Best trial: {study.best_trial.number}")
            logger.info(f"Best value: {study.best_value:.6f}")
            logger.info("Best params:")
            for key, value in study.best_params.items():
                logger.info(f"  {key}: {value}")
        
        return study