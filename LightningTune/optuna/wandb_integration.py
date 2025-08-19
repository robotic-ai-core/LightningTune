"""
WandB integration for Optuna optimization.

This module provides seamless integration between Optuna studies and WandB,
enabling persistent optimization sessions that can be paused, resumed, and
tracked across multiple runs.

Based on proven patterns from production ML workflows.
"""

import os
import pickle
import tempfile
import re
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import logging

import optuna
import wandb
from wandb.errors import CommError

logger = logging.getLogger(__name__)


def save_optuna_session(
    session_info: Dict[str, Any],
    project_name: str,
    study_name: str,
    run: Optional[wandb.Run] = None,
) -> None:
    """
    Save Optuna session information to WandB as an artifact.
    
    Args:
        session_info: Dictionary containing study and metadata
        project_name: WandB project name
        study_name: Name of the Optuna study
        run: Optional existing WandB run to use
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        pickle.dump(session_info, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.flush()
        os.fsync(tmp.fileno())
        
        # Verify the file was written correctly
        tmp.seek(0)
        try:
            pickle.load(tmp)
        except Exception as e:
            logger.error(f"Error verifying saved study: {e}")
            return
        
        # Use existing run or create new one
        if run is None:
            run = wandb.init(project=project_name, job_type="optuna-session")
            finish_run = True
        else:
            finish_run = False
        
        artifact = wandb.Artifact(
            f"{study_name}_session_info",
            type="optuna_session",
            description=f"Optuna study session for {study_name}"
        )
        artifact.add_file(tmp.name, name="session_info.pkl")
        run.log_artifact(artifact)
        
        if finish_run:
            run.finish()
        
        # Clean up temp file
        os.unlink(tmp.name)
    
    logger.info(f"Saved Optuna session to WandB: {study_name}")


def load_optuna_session(
    project_name: str,
    study_name: str,
    version: str = "latest"
) -> Optional[Dict[str, Any]]:
    """
    Load Optuna session information from WandB artifact.
    
    Args:
        project_name: WandB project name
        study_name: Name of the Optuna study
        version: Artifact version to load (default: "latest")
        
    Returns:
        Session information dictionary or None if not found
    """
    api = wandb.Api()
    
    try:
        artifact = api.artifact(f"{project_name}/{study_name}_session_info:{version}")
    except CommError as e:
        if re.search(r'project.*not found', str(e).lower()) or \
           re.search(r'artifact.*not found', str(e).lower()):
            logger.info(f"No existing session found for {study_name}")
            return None
        raise e
    except Exception as e:
        logger.error(f"Failed to load artifact: {e}")
        raise e
    
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact.download(tmpdir)
        file_path = os.path.join(tmpdir, "session_info.pkl")
        
        if not os.path.exists(file_path):
            logger.error(f"session_info.pkl not found in the artifact")
            return None
        
        with open(file_path, 'rb') as f:
            try:
                session_info = pickle.load(f)
            except EOFError as e:
                logger.error("Corrupted session file. Unable to load session info.")
                raise e
            except Exception as e:
                logger.error(f"Error loading session info: {e}")
                raise e
    
    logger.info(f"Loaded Optuna session from WandB: {study_name}")
    return session_info


class WandBOptunaOptimizer:
    """
    Optuna optimizer with WandB integration for persistent optimization.
    
    This class combines Optuna's optimization capabilities with WandB's
    artifact storage to enable pause/resume functionality and experiment
    tracking across multiple runs.
    """
    
    def __init__(
        self,
        objective: Callable[[optuna.Trial], float],
        project_name: str,
        study_name: str,
        direction: str = "minimize",
        n_trials: int = 100,
        save_every_n_trials: int = 10,
        resume_from_version: str = "latest",
        prune_patience: int = 12,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        fast_dev_run: bool = False,
        log_to_wandb: bool = True,
        **objective_kwargs
    ):
        """
        Initialize WandB-integrated Optuna optimizer.
        
        Args:
            objective: Objective function to optimize
            project_name: WandB project name
            study_name: Name for the Optuna study
            direction: "minimize" or "maximize"
            n_trials: Total number of trials to run
            save_every_n_trials: Save study to WandB every N trials
            resume_from_version: WandB artifact version to resume from
            prune_patience: Patience for PatientPruner
            sampler: Optuna sampler (default: TPESampler)
            pruner: Optuna pruner (default: PatientPruner with MedianPruner)
            fast_dev_run: Enable fast development mode
            log_to_wandb: Whether to log metrics to WandB
            **objective_kwargs: Additional kwargs to pass to objective
        """
        self.objective = objective
        self.project_name = project_name
        self.study_name = study_name
        self.direction = direction
        self.n_trials = n_trials
        self.save_every_n_trials = save_every_n_trials
        self.resume_from_version = resume_from_version
        self.prune_patience = prune_patience
        self.fast_dev_run = fast_dev_run
        self.log_to_wandb = log_to_wandb
        self.objective_kwargs = objective_kwargs
        
        # Set up sampler and pruner
        self.sampler = sampler or optuna.samplers.TPESampler(seed=42)
        self.pruner = pruner or optuna.pruners.PatientPruner(
            optuna.pruners.MedianPruner(),
            patience=prune_patience
        )
        
        # Adjust for fast dev run
        if fast_dev_run:
            self.save_every_n_trials = 2
            self.study_name = f"{study_name}_fast_dev"
            self.n_trials = 4
        
        # Initialize tracking
        self.total_trials_completed = 0
        self.study = None
        self.wandb_run = None
    
    def _create_or_load_study(self) -> optuna.Study:
        """Create a new study or load existing one from WandB."""
        # Try to load existing session
        session_info = load_optuna_session(
            self.project_name,
            self.study_name,
            version=self.resume_from_version
        )
        
        if session_info is not None:
            logger.info("Resuming existing study session")
            self.study = session_info["study"]
            self.total_trials_completed = session_info.get("total_trials_completed", 0)
            
            # Log resume information
            logger.info(f"Resumed from trial {self.total_trials_completed}")
            logger.info(f"Best value so far: {self.study.best_value}")
            
        else:
            # Create new study
            self.study = optuna.create_study(
                direction=self.direction,
                study_name=self.study_name,
                sampler=self.sampler,
                pruner=self.pruner,
            )
            logger.info("Created new study")
        
        return self.study
    
    def _objective_with_logging(self, trial: optuna.Trial) -> float:
        """Wrapper for objective that logs to WandB."""
        # Create WandB run for this trial if logging is enabled
        if self.log_to_wandb:
            trial_run = wandb.init(
                project=self.project_name,
                name=f"{self.study_name}_trial_{trial.number}",
                config=trial.params,
                group=self.study_name,
                job_type="trial",
                # Let WandB auto-generate unique ID and handle name conflicts
            )
        
        try:
            # Run objective
            value = self.objective(trial, **self.objective_kwargs)
            
            # Log to WandB
            if self.log_to_wandb:
                wandb.log({
                    "trial_number": trial.number,
                    "objective_value": value,
                    **trial.params
                })
            
            return value
            
        finally:
            if self.log_to_wandb:
                wandb.finish()
    
    def run(self) -> optuna.Study:
        """
        Run the optimization with automatic checkpointing to WandB.
        
        Returns:
            The Optuna study object with results
        """
        # Create or load study
        self._create_or_load_study()
        
        # Initialize WandB run for the optimization session
        if self.log_to_wandb:
            self.wandb_run = wandb.init(
                project=self.project_name,
                name=f"{self.study_name}_session",
                job_type="optimization",
                config={
                    "n_trials": self.n_trials,
                    "direction": self.direction,
                    "sampler": self.sampler.__class__.__name__,
                    "pruner": self.pruner.__class__.__name__,
                }
            )
        
        try:
            while self.total_trials_completed < self.n_trials:
                # Calculate trials for this batch
                trials_this_batch = min(
                    self.save_every_n_trials,
                    self.n_trials - self.total_trials_completed
                )
                
                logger.info(f"Running {trials_this_batch} trials "
                          f"({self.total_trials_completed}/{self.n_trials} completed)")
                
                # Run optimization batch
                self.study.optimize(
                    self._objective_with_logging,
                    n_trials=trials_this_batch,
                )
                
                self.total_trials_completed += trials_this_batch
                
                # Save session to WandB
                session_info = {
                    "study": self.study,
                    "total_trials_completed": self.total_trials_completed,
                    "best_params": self.study.best_params,
                    "best_value": self.study.best_value,
                }
                
                save_optuna_session(
                    session_info,
                    self.project_name,
                    self.study_name,
                    run=self.wandb_run if self.log_to_wandb else None
                )
                
                # Log progress
                logger.info(f"Completed {self.total_trials_completed}/{self.n_trials} trials")
                logger.info(f"Current best parameters: {self.study.best_params}")
                logger.info(f"Current best value: {self.study.best_value}")
                
                if self.log_to_wandb:
                    wandb.log({
                        "total_trials_completed": self.total_trials_completed,
                        "best_value": self.study.best_value,
                    })
        
        finally:
            # Clean up WandB run
            if self.log_to_wandb and self.wandb_run:
                # Log final results
                wandb.summary["best_value"] = self.study.best_value
                wandb.summary["best_params"] = self.study.best_params
                wandb.summary["total_trials"] = self.total_trials_completed
                
                wandb.finish()
        
        return self.study
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get the best configuration found."""
        if not self.study:
            raise ValueError("No optimization has been run yet")
        return self.study.best_params
    
    def get_best_value(self) -> float:
        """Get the best objective value found."""
        if not self.study:
            raise ValueError("No optimization has been run yet")
        return self.study.best_value
    
    def visualize(self, save_dir: Optional[Path] = None) -> None:
        """Generate and save visualization plots."""
        if not self.study:
            raise ValueError("No optimization has been run yet")
        
        try:
            import optuna.visualization as vis
            
            save_dir = save_dir or Path("./optuna_visualizations")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate plots
            plots = {
                "optimization_history": vis.plot_optimization_history(self.study),
                "param_importances": vis.plot_param_importances(self.study),
                "parallel_coordinate": vis.plot_parallel_coordinate(self.study),
                "slice": vis.plot_slice(self.study),
            }
            
            # Save plots
            for name, fig in plots.items():
                fig.write_html(save_dir / f"{name}.html")
            
            # Log to WandB if available
            if self.log_to_wandb:
                for name, fig in plots.items():
                    wandb.log({f"plot_{name}": wandb.Html(fig.to_html())})
            
            logger.info(f"Visualizations saved to: {save_dir}")
            
        except ImportError:
            logger.warning("Plotly not installed. Skipping visualizations.")