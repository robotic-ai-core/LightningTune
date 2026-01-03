"""
HPO Configuration dataclasses for clean, typed configuration.

This module provides dataclasses that group related configuration options,
reducing constructor parameter counts and providing clear documentation of defaults.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set, Any, Dict


@dataclass
class HPOPersistenceConfig:
    """Configuration for HPO study persistence.

    Attributes:
        save_every_n_trials: Upload to WandB every N trials. Local saves happen
            every trial when restart_every_trial=True.
        local_checkpoint_dir: Directory for local checkpoints. If None, uses
            checkpoints/<wandb_project>/<study_name> or checkpoints/<study_name>.
        wandb_project: WandB project name for artifact storage. None disables WandB.
        persist_args: Whether to automatically persist command-line arguments.
        args_exclude: Set of argument names to exclude from persistence.
    """
    save_every_n_trials: int = 10
    local_checkpoint_dir: Optional[Path] = None
    wandb_project: Optional[str] = None
    persist_args: bool = True
    args_exclude: Set[str] = field(default_factory=lambda: {'resume_from', 'study_name'})

    def __post_init__(self):
        """Convert string path to Path object if needed."""
        if isinstance(self.local_checkpoint_dir, str):
            self.local_checkpoint_dir = Path(self.local_checkpoint_dir)
        if isinstance(self.args_exclude, (list, tuple)):
            self.args_exclude = set(self.args_exclude)


@dataclass
class HPOPauseConfig:
    """Configuration for pause/resume behavior.

    Attributes:
        enable_pause: Whether to enable 'p' key pause functionality.
        pause_key: Key to trigger pause (default 'p').
        restart_on_save: Whether to exit for process restart after saving.
        restart_every_trial: When restart_on_save=True, restart after every trial
            for complete memory isolation.
    """
    enable_pause: bool = True
    pause_key: str = 'p'
    restart_on_save: bool = False
    restart_every_trial: bool = True


@dataclass
class HPOSamplerConfig:
    """Configuration for Optuna sampler.

    Attributes:
        name: Sampler name ('tpe', 'random', 'cmaes', 'botorch', 'grid').
        seed: Random seed for reproducibility.
        kwargs: Additional arguments passed to sampler constructor.
    """
    name: str = 'tpe'
    seed: Optional[int] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HPOPrunerConfig:
    """Configuration for Optuna pruner.

    Attributes:
        name: Pruner name ('median', 'hyperband', 'successivehalving', 'none').
        kwargs: Additional arguments passed to pruner constructor.
    """
    name: str = 'median'
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HPOTrialConfig:
    """Configuration for trial execution.

    Attributes:
        n_trials: Number of trials to run.
        trial_steps: Maximum steps per trial (None for no limit).
        metric: Metric to optimize.
        direction: Optimization direction ('minimize' or 'maximize').
        checkpoint_top_k: Number of best trial checkpoints to keep. 0 disables.
    """
    n_trials: int = 50
    trial_steps: Optional[int] = None
    metric: str = 'val_loss'
    direction: str = 'minimize'
    checkpoint_top_k: int = 0


@dataclass
class HPOConfig:
    """Complete HPO configuration.

    This dataclass groups all HPO-related configuration into a single,
    well-documented structure. It can be constructed from individual
    parameters or nested config objects.

    Example:
        >>> config = HPOConfig(
        ...     study_name="my_hpo_study",
        ...     persistence=HPOPersistenceConfig(
        ...         wandb_project="my-project",
        ...         save_every_n_trials=5,
        ...     ),
        ...     pause=HPOPauseConfig(enable_pause=True),
        ... )

    Attributes:
        study_name: Name for the Optuna study.
        persistence: Persistence configuration.
        pause: Pause/resume configuration.
        sampler: Sampler configuration.
        pruner: Pruner configuration.
        trial: Trial execution configuration.
    """
    study_name: str = 'optuna_study'
    persistence: HPOPersistenceConfig = field(default_factory=HPOPersistenceConfig)
    pause: HPOPauseConfig = field(default_factory=HPOPauseConfig)
    sampler: HPOSamplerConfig = field(default_factory=HPOSamplerConfig)
    pruner: HPOPrunerConfig = field(default_factory=HPOPrunerConfig)
    trial: HPOTrialConfig = field(default_factory=HPOTrialConfig)

    @classmethod
    def from_flat_params(
        cls,
        study_name: str = 'optuna_study',
        # Persistence params
        save_every_n_trials: int = 10,
        local_checkpoint_dir: Optional[Path] = None,
        wandb_project: Optional[str] = None,
        persist_args: bool = True,
        args_exclude: Optional[Set[str]] = None,
        # Pause params
        enable_pause: bool = True,
        pause_key: str = 'p',
        restart_on_save: bool = False,
        restart_every_trial: bool = True,
        # Sampler params
        sampler_name: str = 'tpe',
        sampler_seed: Optional[int] = None,
        # Pruner params
        pruner_name: str = 'median',
        # Trial params
        n_trials: int = 50,
        trial_steps: Optional[int] = None,
        metric: str = 'val_loss',
        direction: str = 'minimize',
        checkpoint_top_k: int = 0,
        **kwargs,
    ) -> 'HPOConfig':
        """Create HPOConfig from flat parameters (for backward compatibility).

        This factory method allows creating an HPOConfig from the flat parameter
        style used by the old constructors, enabling gradual migration.
        """
        return cls(
            study_name=study_name,
            persistence=HPOPersistenceConfig(
                save_every_n_trials=save_every_n_trials,
                local_checkpoint_dir=local_checkpoint_dir,
                wandb_project=wandb_project,
                persist_args=persist_args,
                args_exclude=args_exclude or {'resume_from', 'study_name'},
            ),
            pause=HPOPauseConfig(
                enable_pause=enable_pause,
                pause_key=pause_key,
                restart_on_save=restart_on_save,
                restart_every_trial=restart_every_trial,
            ),
            sampler=HPOSamplerConfig(
                name=sampler_name,
                seed=sampler_seed,
            ),
            pruner=HPOPrunerConfig(
                name=pruner_name,
            ),
            trial=HPOTrialConfig(
                n_trials=n_trials,
                trial_steps=trial_steps,
                metric=metric,
                direction=direction,
                checkpoint_top_k=checkpoint_top_k,
            ),
        )

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for serialization."""
        return {
            'study_name': self.study_name,
            # Persistence
            'save_every_n_trials': self.persistence.save_every_n_trials,
            'local_checkpoint_dir': str(self.persistence.local_checkpoint_dir) if self.persistence.local_checkpoint_dir else None,
            'wandb_project': self.persistence.wandb_project,
            'persist_args': self.persistence.persist_args,
            'args_exclude': list(self.persistence.args_exclude),
            # Pause
            'enable_pause': self.pause.enable_pause,
            'pause_key': self.pause.pause_key,
            'restart_on_save': self.pause.restart_on_save,
            'restart_every_trial': self.pause.restart_every_trial,
            # Sampler
            'sampler_name': self.sampler.name,
            'sampler_seed': self.sampler.seed,
            # Pruner
            'pruner_name': self.pruner.name,
            # Trial
            'n_trials': self.trial.n_trials,
            'trial_steps': self.trial.trial_steps,
            'metric': self.trial.metric,
            'direction': self.trial.direction,
            'checkpoint_top_k': self.trial.checkpoint_top_k,
        }
