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
from typing import Optional, Dict, Any, Callable, Union, Type, List, Set
import sys
from pathlib import Path
import threading
import time
import yaml

import optuna
import wandb
from lightning import LightningModule
from lightning.pytorch.callbacks import Callback

from .optimizer import OptunaDrivenOptimizer
from .factories import create_sampler, create_pruner
from ..persistence import (
    save_study_to_local as persist_save_study_to_local,
    load_study_from_local as persist_load_study_from_local,
    save_study_to_wandb as persist_save_study_to_wandb,
    load_study_from_wandb as persist_load_study_from_wandb,
    load_saved_session as persist_load_saved_session,
    build_resume_command as persist_build_resume_command,
    build_local_resume_command as persist_build_local_resume_command,
    parse_cli_arg,
)
from ..arg_persistence import (
    merge_args_with_saved,
    normalize_n_trials_in_overrides,
    extend_or_align_n_trials,
)
try:
    # Ensure Reflow package is importable when used as a submodule
    reflow_path = Path(__file__).parent.parent.parent.parent / "LightningReflow"
    if reflow_path.exists():
        sys.path.insert(0, str(reflow_path))
    # Prefer Reflow's robust keyboard handler to improve TTY restore and Ctrl+C behavior
    from lightning_reflow.callbacks.pause.improved_keyboard_handler import (
        create_improved_keyboard_handler,
    )
    # Import KeyboardHandlerService (modern approach, eliminates duplicate threads)
    from lightning_reflow.services import KeyboardHandlerService, KeyboardHandlerStrategy
    HAS_KEYBOARD_SERVICE = True
except Exception:  # Fallback if Reflow is not available
    create_improved_keyboard_handler = None  # type: ignore
    KeyboardHandlerService = None  # type: ignore
    KeyboardHandlerStrategy = None  # type: ignore
    HAS_KEYBOARD_SERVICE = False

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
        restart_on_save: bool = False,
        restart_every_trial: bool = True,  # When restart_on_save=True, restart after every trial
        enable_pause: bool = True,
        use_reflow: bool = True,  # Deprecated - always uses LightningReflow now
        checkpoint_top_k: int = 0,  # Number of best trial checkpoints to keep. 0 disables checkpointing.
        # New enhanced features
        override_config: Optional[Union[str, Dict[str, Any]]] = None,
        persist_args: bool = True,
        args: Optional[Any] = None,
        args_exclude: Optional[Set[str]] = None,
        simplify_param_names: bool = True,
        compile_mode: Optional[str] = None,  # None means default SAFE without persisting
        test_mode: bool = False,  # Enable test keyboard handler for automated testing
        **optimizer_kwargs
    ):
        """
        Initialize the pausible optimizer with enhanced features.

        Args:
            base_config: Base configuration file path or dict
            search_space: Function or OptunaSearchSpace defining parameters to optimize
            model_class: PyTorch Lightning module class
            datamodule_class: Optional PyTorch Lightning datamodule class
            wandb_project: WandB project name for artifact storage (None disables WandB)
            study_name: Name for the study (used in WandB artifacts)
            sampler_name: Name of Optuna sampler to use
            pruner_name: Name of Optuna pruner to use
            save_every_n_trials: Upload to WandB every N trials (local saves happen every trial when restart_every_trial=True)
            restart_on_save: Whether to exit for process restart after saving
            restart_every_trial: When restart_on_save=True, restart after every trial for complete memory isolation
            enable_pause: Whether to enable 'p' key pause functionality
            use_reflow: Deprecated - always uses LightningReflow now (parameter kept for backward compatibility)
            checkpoint_top_k: Number of best trial checkpoints to keep. 0 disables checkpointing (default).
            override_config: Optional override configuration to layer on top of base_config
            persist_args: Whether to automatically persist command-line arguments
            args: Parsed command-line arguments to persist (if persist_args=True)
            args_exclude: Set of argument names to exclude from persistence
            simplify_param_names: Whether to simplify parameter names for cleaner logging
            compile_mode: Torch compile mode - "off", "safe", or "aggressive"
            **optimizer_kwargs: Additional arguments for OptunaDrivenOptimizer
        """
        # Handle layered configs if override_config is provided
        if override_config is not None:
            from ..utils import deep_merge_configs, load_yaml_config
            
            # Load base config
            if isinstance(base_config, str):
                base_dict = load_yaml_config(base_config)
            else:
                base_dict = base_config
                
            # Load override config
            if isinstance(override_config, str):
                override_dict = load_yaml_config(override_config)
            else:
                override_dict = override_config
                
            # Merge configs
            self.base_config = deep_merge_configs(base_dict, override_dict)
            logger.info(f"📑 Merged base and override configs")
        else:
            self.base_config = base_config
            
        self.search_space = search_space
        self.model_class = model_class
        self.datamodule_class = datamodule_class
        self.wandb_project = wandb_project
        self.study_name = study_name
        self.sampler_name = sampler_name
        self.pruner_name = pruner_name
        self.save_every_n_trials = save_every_n_trials
        self.restart_on_save = restart_on_save
        self.restart_every_trial = restart_every_trial
        self.enable_pause = enable_pause
        self.use_reflow = use_reflow
        self.checkpoint_top_k = checkpoint_top_k
        self.test_mode = test_mode
        
        # New enhanced features
        self.persist_args = persist_args
        self.args = args
        # Persist all args by default except resume_from and study_name. n_trials is persisted and auto-restored.
        self.args_exclude = args_exclude or {'resume_from', 'study_name'}
        self.simplify_param_names = simplify_param_names
        # Track whether compile_mode is explicitly provided
        self._compile_mode_explicit = compile_mode is not None
        self.compile_mode = compile_mode or "safe"
        self.optimizer_kwargs = optimizer_kwargs
        
        # Initialize persistent_config_overrides early if we have args
        self.persistent_config_overrides: Optional[Dict[str, Any]] = {}
        if self.persist_args and self.args:
            self._build_args_config_overrides()
        # Optional local checkpoint directory (for mirroring study.pkl)
        self.local_checkpoint_dir: Optional[Path] = None
        try:
            lcd = self.optimizer_kwargs.pop("local_checkpoint_dir", None)
            if lcd:
                self.local_checkpoint_dir = Path(lcd)
                self.local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.local_checkpoint_dir = None
        if self.local_checkpoint_dir is None:
            try:
                base = Path("checkpoints")
                if self.wandb_project:
                    self.local_checkpoint_dir = base / self.wandb_project / self.study_name
                else:
                    self.local_checkpoint_dir = base / self.study_name
                self.local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                self.local_checkpoint_dir = None
        # Preserve original CLI argv to build accurate resume commands (Reflow-style)
        try:
            self._original_argv: List[str] = sys.argv.copy()
        except Exception:
            self._original_argv = []
        
        # Track progress
        self.total_trials_completed = 0
        self.should_pause = False
        self._quit_after_current = False

        # Setup keyboard handler for 'p' key pause (robust terminal handling)
        self.keyboard_handler = None
        self.keyboard_service = None  # KeyboardHandlerService (modern approach)
        self._use_keyboard_service = False  # Flag to determine which handler is active
        # Backward-compatibility shim for tests (deprecated, will be removed)
        self.keyboard_monitor = None
        self._pause_requested: bool = False
        self._pause_lock = threading.Lock()  # Thread-safe state mutation for keyboard callbacks
        # Initialize polling thread attributes (DEPRECATED - will be removed)
        self._pause_poll_thread = None
        self._polling_active = False
        if enable_pause and os.environ.get("LT_CHILD", "0") != "1":
            # Try KeyboardHandlerService first (preferred, eliminates duplicate threads)
            if HAS_KEYBOARD_SERVICE and KeyboardHandlerService is not None:
                try:
                    self.keyboard_service = KeyboardHandlerService.get_instance(
                        strategy=KeyboardHandlerStrategy.IMPROVED_MODE,
                        debounce_interval=0.3
                    )
                    if self.keyboard_service.is_available():
                        self._use_keyboard_service = True
                        logger.info("✅ Using KeyboardHandlerService for HPO pause (eliminates duplicate threads)")
                    else:
                        # Fallback to ImprovedKeyboardHandler
                        if create_improved_keyboard_handler is not None:
                            self.keyboard_handler = create_improved_keyboard_handler(test_mode=test_mode)
                            logger.warning("⚠️  Falling back to ImprovedKeyboardHandler (KeyboardHandlerService not available)")
                except Exception as e:
                    logger.warning(f"KeyboardHandlerService failed ({e}), falling back to ImprovedKeyboardHandler")
                    if create_improved_keyboard_handler is not None:
                        self.keyboard_handler = create_improved_keyboard_handler(test_mode=test_mode)
            elif create_improved_keyboard_handler is not None:
                self.keyboard_handler = create_improved_keyboard_handler(test_mode=test_mode)
                logger.warning("⚠️  KeyboardHandlerService not available, using ImprovedKeyboardHandler")
            else:
                self.keyboard_handler = None
    

    def _handle_key_input(self, key: str) -> bool:
        """Handle a key press with thread-safe state management.

        This is the core key handling logic used by both the KeyboardHandlerService
        callback and the polling thread.

        Returns:
            True if the key was handled, False otherwise.
        """
        skey = key.lower()

        if skey == 'p':
            # Toggle pause state (thread-safe) with debounce to prevent duplicates
            with self._pause_lock:
                # Debounce: ignore if less than 300ms since last 'p' press
                current_time = time.time()
                last_p_time = getattr(self, '_last_p_press_time', 0)
                if current_time - last_p_time < 0.3:
                    return True  # Ignore duplicate
                self._last_p_press_time = current_time

                last_state = self._pause_requested
                self._pause_requested = not self._pause_requested
                current_state = self._pause_requested

            # Print feedback (outside lock) - use print() only, not logger
            # to avoid duplicate output when logger also writes to stdout
            if current_state and not last_state:
                msg = "\n⏸️  Pause SCHEDULED ('p' pressed)"
                print(msg, flush=True)
                self._log_to_pause_file(msg)
            elif not current_state and last_state:
                msg = "\n❌ Pause CANCELLED ('p' pressed again)"
                print(msg, flush=True)
                self._log_to_pause_file(msg)
            return True
        elif skey == 'q':
            with self._pause_lock:
                self._quit_after_current = True
            msg = "\n🛑 Quit requested ('q' pressed). Will stop after current trial."
            print(msg, flush=True)
            return True
        elif key == "\x03":  # Ctrl+C in cbreak mode
            with self._pause_lock:
                self._pause_requested = True
                self.should_pause = True
            msg = "\n⏸️  Ctrl+C detected. Pausing gracefully at trial boundary..."
            print(msg, flush=True)
            return True
        return False

    def _log_to_pause_file(self, msg: str) -> None:
        """Log message to pause log file for visibility."""
        try:
            with open("/tmp/hpo_pause.log", "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {msg.strip()}\n")
                f.flush()
        except:
            pass

    def _on_key_press(self, key: str):
        """Callback from KeyboardHandlerService (invoked from background thread).

        This method is thread-safe using self._pause_lock to protect state mutations.
        """
        self._handle_key_input(key)

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

    @staticmethod
    def load_saved_session(
        resume_from: str,
        wandb_project: Optional[str] = None,
        study_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        return persist_load_saved_session(
            resume_from,
            wandb_project=wandb_project,
            study_name=study_name,
        )
    
    def load_study_from_wandb(self, version: str = "latest") -> Optional[Dict[str, Any]]:
        return persist_load_study_from_wandb(
            self.wandb_project,
            self.study_name,
            version=version,
        )
    
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
        logger.debug(f"optimize() called with n_trials={n_trials}, resume_from={resume_from}")

        # Handle automatic argument persistence
        if self.persist_args and self.args:
            args_dict = self._extract_persistable_args()
            if config_overrides is None:
                config_overrides = {}
            import sys as _sys
            if resume_from:
                for arg_name, arg_value in (args_dict or {}).items():
                    if arg_name in self.args_exclude or arg_value is None:
                        continue
                    cmd = ' '.join(_sys.argv)
                    if f"--{arg_name.replace('_','-')}" in cmd or f"--{arg_name}" in cmd:
                        # Persist only explicitly provided args; include n_trials here
                        config_overrides[f"args.{arg_name}"] = arg_value
            else:
                for arg_name, arg_value in (args_dict or {}).items():
                    if arg_name in self.args_exclude or arg_value is None:
                        continue
                    config_overrides[f"args.{arg_name}"] = arg_value
        
        # Add torch compile settings based on compile_mode (only if model supports it)
        # These are runtime-only and should NOT be persisted
        runtime_overrides: Dict[str, Any] = {}
        if self.compile_mode:
            from ..utils.torch_compile import get_compile_settings_for_mode
            compile_settings = get_compile_settings_for_mode(self.compile_mode)
            if compile_settings:
                # Detect if model_class accepts torch_compile_settings or **kwargs
                safe_to_inject = False
                try:
                    import inspect
                    if self.model_class is not None:
                        signature = inspect.signature(self.model_class)
                        params = signature.parameters
                        if "torch_compile_settings" in params:
                            safe_to_inject = True
                        else:
                            safe_to_inject = any(
                                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                            )
                except Exception:
                    safe_to_inject = False

                if safe_to_inject:
                    runtime_overrides["model.init_args.torch_compile_settings"] = compile_settings
                    if self.compile_mode == "off":
                        logger.info("⚠️  Torch compilation disabled")
                    elif self.compile_mode == "safe":
                        logger.info("🛡️  Using safe torch.compile settings for HPO")
                    elif self.compile_mode == "aggressive":
                        logger.info("🚀 Using aggressive torch.compile settings")
                else:
                    logger.debug("Skipping torch_compile_settings override; model does not accept it")
        
        # Resolve resume automatically (backward compatible call order for tests and existing scripts)
        session_info = None
        if resume_from:
            import os as _os
            logger.info(f"📥 Resume requested: {resume_from}")
            if _os.path.exists(resume_from):
                # Explicit local path preferred
                session_info = self.load_study_from_local(resume_from)
                if session_info is None and self.local_checkpoint_dir and self.local_checkpoint_dir.exists():
                    logger.info(f"💾 Trying local checkpoint fallback: {self.local_checkpoint_dir}")
                    session_info = persist_load_study_from_local(str(self.local_checkpoint_dir))
            else:
                # Alias like 'latest' or 'vN' - try local first, then WandB
                # Note: The parent orchestrator (run_with_auto_restart) now passes explicit local
                # paths for subprocess restarts, so this branch is mainly for manual resume.
                # When users manually use --resume-from latest, prefer local checkpoint since
                # it's more up-to-date (saved every trial vs WandB uploaded every N trials).
                local_session = None
                if self.local_checkpoint_dir and self.local_checkpoint_dir.exists():
                    local_session = persist_load_study_from_local(str(self.local_checkpoint_dir))

                if local_session:
                    local_trials = local_session.get("total_trials_completed", 0)
                    logger.info(f"💾 Using local checkpoint ({local_trials} trials)")
                    session_info = local_session
                else:
                    # No local checkpoint, try WandB
                    session_info = self.load_study_from_wandb(resume_from)

                # Final generic fallback
                if session_info is None:
                    session_info = persist_load_saved_session(
                        resume_from,
                        wandb_project=self.wandb_project,
                        study_name=self.study_name,
                    )
            
            if session_info is None:
                logger.error(f"\n{'='*60}")
                logger.error(f"❌ FATAL: Failed to resume from '{resume_from}'")
                logger.error(f"{'='*60}")
                logger.error(f"Possible reasons:")
                logger.error(f"  • No checkpoint exists with this name")
                logger.error(f"  • WandB project/study name mismatch")
                logger.error(f"  • Network/authentication issues with WandB")
                logger.error(f"  • File is corrupted or unreadable")
                logger.error(f"\n💡 To start a fresh study, remove the --resume-from flag")
                logger.error(f"{'='*60}\n")
                import sys
                sys.exit(1)  # Exit with error - DO NOT start fresh study!
        
        if session_info:
            study = session_info["study"]
            self.total_trials_completed = session_info["total_trials_completed"]
            self.should_pause = False  # Reset pause flag when resuming

            # Log resume info (useful for verifying correct checkpoint was loaded)
            loaded_trial_count = len(study.trials)
            logger.debug(f"Loaded study with {loaded_trial_count} trials, "
                        f"total_trials_completed={self.total_trials_completed}")
            if loaded_trial_count > 0:
                last_trial = study.trials[-1]
                logger.debug(f"Last trial: number={last_trial.number}, state={last_trial.state}")

            # Handle persistent config overrides
            saved_config_overrides = session_info.get("config_overrides", {}) or {}
            current_config_overrides = config_overrides or {}
            normalize_n_trials_in_overrides(saved_config_overrides)
            if (
                self.persist_args and self.args and saved_config_overrides and
                not getattr(self.args, '_restored_by_hporunner', False)
            ):
                import sys as _sys
                merge_args_with_saved(
                    self.args,
                    saved_config_overrides,
                    non_persistent=self.args_exclude,
                    argv=_sys.argv,
                )
            elif getattr(self.args, '_restored_by_hporunner', False):
                logger.debug("  Skipping arg restoration - handled by HPORunner")
            
            # Check if n_trials was overridden
            saved_n_trials = saved_config_overrides.get('args.n_trials')
            n_trials_extended = False
            original_n_trials = n_trials  # Keep the original value for comparison

            # Check if n_trials was explicitly specified on command line
            # But if we're being called from HPORunner, it has already handled n_trials restoration
            # So we should skip this logic if HPORunner is in charge
            should_handle_n_trials = not getattr(self.args, '_restored_by_hporunner', False) if self.args else True

            if not should_handle_n_trials:
                # HPORunner is managing argument restoration, don't interfere
                if saved_n_trials and n_trials != saved_n_trials:
                    # Just log what HPORunner decided
                    if n_trials > saved_n_trials:
                        n_trials_extended = True
                        logger.debug(f"HPORunner extended n_trials from {saved_n_trials} to {n_trials}")
                    else:
                        logger.debug(f"HPORunner set n_trials to {n_trials} (saved was {saved_n_trials})")
            else:
                import sys as _sys
                cli_specified = ('--n-trials' in ' '.join(_sys.argv)) or ('--n_trials' in ' '.join(_sys.argv))
                n_trials, n_trials_extended = extend_or_align_n_trials(
                    n_trials,
                    saved_n_trials,
                    cli_specified=bool(cli_specified),
                )
                if n_trials_extended:
                    logger.info(f"📈 n_trials extended from {saved_n_trials} to {n_trials}")

            # Merge saved and current overrides (current takes precedence)
            merged_config_overrides = {**saved_config_overrides, **current_config_overrides}

            # Always update n_trials in persistent config with the current value
            merged_config_overrides['args.n_trials'] = n_trials

            # Create a copy for passing to the optimizer (without n_trials since it's a separate param)
            optimizer_config_overrides = merged_config_overrides.copy()
            if 'args.n_trials' in optimizer_config_overrides:
                del optimizer_config_overrides['args.n_trials']
            if 'n_trials' in optimizer_config_overrides:
                del optimizer_config_overrides['n_trials']

            # Display resume information (simplified for Lightning progress bar compatibility)
            logger.info(f"\n📂 Resuming: {self.total_trials_completed}/{n_trials} trials complete")

            # Show n_trials extension if applicable
            if n_trials_extended:
                logger.info(f"   Extending trials: {saved_n_trials} → {n_trials}")
            
            # Store merged overrides for future saves (includes n_trials if extended)
            self.persistent_config_overrides = merged_config_overrides
            # Use the version without n_trials for the optimizer
            config_overrides = optimizer_config_overrides
            
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
                storage=storage,
                load_if_exists=True if storage else False
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
            
            # Store config overrides for new study (merge with args-based overrides)
            if config_overrides:
                self.persistent_config_overrides.update(config_overrides)
            
            # Display initial config overrides if any (simplified)
            if self.persistent_config_overrides:
                # Filter out torch compile defaults when they are the only items
                items = {
                    k: v for k, v in self.persistent_config_overrides.items()
                    if v is not None and k != "model.init_args.torch_compile_settings"
                }
                if items:
                    logger.info(f"\n📋 Config overrides: {len(items)} parameter(s)")
        
        # Merge optimizer kwargs
        opt_kwargs = self.optimizer_kwargs.copy()
        opt_kwargs.update(kwargs)

        # Extract direction to avoid duplicate argument
        direction = opt_kwargs.pop("direction", "minimize")

        # Remove use_reflow if present (deprecated, always use LightningReflow now)
        opt_kwargs.pop("use_reflow", None)
        # Merge persistent overrides and runtime-only overrides for the optimizer
        _config_overrides_for_optimizer = dict(config_overrides or {})
        if runtime_overrides:
            _config_overrides_for_optimizer.update(runtime_overrides)

        pre_injected_optimizer = getattr(self, 'underlying_optimizer', None)
        if pre_injected_optimizer is not None:
            optimizer = pre_injected_optimizer
        else:
            optimizer = OptunaDrivenOptimizer(
                base_config=self.base_config,
                search_space=self.search_space,
                config_overrides=_config_overrides_for_optimizer,
                model_class=self.model_class,
                datamodule_class=self.datamodule_class,
                sampler=study.sampler,  # Use study's sampler
                pruner=study.pruner,     # Use study's pruner
                study_name=self.study_name,
                direction=direction,
                n_trials=1,  # We'll run one at a time for checkpointing
                callbacks=callbacks,
                wandb_project=self.wandb_project,
                checkpoint_top_k=self.checkpoint_top_k,
                **opt_kwargs
            )

        # Allow tests to inject a custom objective by patching self.create_objective
        # Expose underlying optimizer for test injection
        try:
            self.underlying_optimizer = optimizer
        except Exception:
            pass

        try:
            custom_obj_factory = getattr(self, 'create_objective', None)
            objective = None
            if callable(custom_obj_factory):
                try:
                    candidate = custom_obj_factory()
                    if callable(candidate):
                        objective = candidate
                except Exception:
                    objective = None
            if objective is None:
                under = getattr(self, 'underlying_optimizer', optimizer)
                objective = under.create_objective()
        except Exception:
            objective = optimizer.create_objective()
        
        # Start keyboard monitoring if available
        if self._use_keyboard_service and self.keyboard_service:
            # Register callback with KeyboardHandlerService (MODERN APPROACH - no polling thread!)
            try:
                self.keyboard_service.register_subscriber("hpo_pause", self._on_key_press)
                logger.info("⌨️  Keyboard monitoring active via KeyboardHandlerService - press 'p' to pause between trials")
                logger.info("   ✅ Using centralized service (eliminates duplicate threads)")
                logger.info("   Pause events will be logged to /tmp/hpo_pause.log")
            except Exception as e:
                logger.warning(f"⚠️  Failed to register with KeyboardHandlerService: {e}")
                logger.warning("⚠️  Pause functionality disabled")
                self.keyboard_service = None
                self._use_keyboard_service = False
        elif self.keyboard_handler and hasattr(self.keyboard_handler, 'start_monitoring'):
            # Legacy ImprovedKeyboardHandler path (for backward compatibility)
            try:
                # Check if it's actually available before starting
                if hasattr(self.keyboard_handler, 'is_available'):
                    if not self.keyboard_handler.is_available():
                        logger.warning("⚠️  Keyboard monitoring unavailable (no TTY)")
                        logger.warning("⚠️  stdin.isatty() returned False - are you running in a pipe/redirect?")
                        logger.warning("⚠️  Pause functionality will be disabled")
                        self.keyboard_handler = None
                    else:
                        self.keyboard_handler.start_monitoring()
                        logger.info("⌨️  Keyboard monitoring active - press 'p' to pause between trials")
                        logger.info("   Pause events will be logged to /tmp/hpo_pause.log")
                else:
                    self.keyboard_handler.start_monitoring()
            except Exception as e:
                logger.warning(f"⚠️  Keyboard monitoring failed to start: {e}")
                logger.warning("⚠️  Pause functionality disabled")
                self.keyboard_handler = None
            # Start background polling for immediate schedule/cancel feedback
            # NOTE: This polling thread is ONLY used with legacy ImprovedKeyboardHandler
            # KeyboardHandlerService uses callbacks - no polling thread needed!
            self._pause_poll_thread = None
            self._polling_active = False
            if self.keyboard_handler and hasattr(self.keyboard_handler, 'get_key'):
                self._start_pause_polling_thread()
                logger.warning("⚠️  Using legacy polling thread (upgrade to KeyboardHandlerService to eliminate)")
        
        # Run trials with periodic saves
        # If resuming, restore the counter state from checkpoint
        # Note: When we load checkpoint, self.total_trials_completed is already set from it
        # That value IS the last saved count, so we can use it directly
        # Note: last_checkpoint_trial_count is a LOCAL loop variable (not persisted to disk)
        if session_info:
            # Checkpoint's total_trials_completed IS the last saved count
            checkpoint_trial_count = session_info.get('total_trials_completed', 0)
            trials_in_batch = self.total_trials_completed - checkpoint_trial_count
            logger.info(f"📊 Restored save counter: {trials_in_batch}/{self.save_every_n_trials} trials since last save")
            last_checkpoint_trial_count = checkpoint_trial_count
        else:
            trials_in_batch = 0
            last_checkpoint_trial_count = self.total_trials_completed
        
        logger.debug(f"Starting optimization loop: total_trials_completed={self.total_trials_completed}, n_trials={n_trials}")
        logger.debug(f"Study has {len(study.trials)} trials, next trial will be number {len(study.trials)}")

        while self.total_trials_completed < n_trials and not self.should_pause:
            # Record number of finished trials (COMPLETE + PRUNED) before this trial
            trials_before = len([t for t in study.trials 
                                if t.state in [optuna.trial.TrialState.COMPLETE,
                                              optuna.trial.TrialState.PRUNED]])
            
            # Check for keyboard pause request before starting trial
            if self._update_pause_from_keyboard():
                self.should_pause = True
                msg = "\n⏸️  Executing pause at trial boundary..."
                print(msg)  # Print directly so it shows up even with progress bar
                logger.info(msg)
                # Log to file for visibility
                try:
                    with open("/tmp/hpo_pause.log", "a") as f:
                        f.write(f"[{time.strftime('%H:%M:%S')}] {msg.strip()}\n")
                        f.flush()
                except:
                    pass
                if self.wandb_project:
                    logger.info("   Study will be saved to WandB for easy resume")
                break
            
            try:
                # Show trial start (simplified for Lightning progress bar compatibility)
                trial_number = self.total_trials_completed + 1
                logger.info(f"📊 Trial {trial_number}/{n_trials}")
                
                # Run single trial with automatic garbage collection
                # gc_after_trial=True ensures memory is cleaned between trials
                study.optimize(objective, n_trials=1, show_progress_bar=False, gc_after_trial=True)

                # Additional memory cleanup to prevent accumulation
                from .memory_cleanup import cleanup_trial_resources, aggressive_cleanup
                cleanup_trial_resources()
                aggressive_cleanup()  # Force release of any lingering references
                
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

                    # Simplify param names for logging if enabled
                    trial_params = latest_trial.params
                    if self.simplify_param_names and trial_params:
                        from ..utils.param_utils import simplify_param_names
                        trial_params = simplify_param_names(trial_params)

                    # Log trial result (simplified for Lightning progress bar compatibility)
                    save_info = f"[{trials_in_batch}/{self.save_every_n_trials} until save]" if self.save_every_n_trials else ""
                    logger.info(f"✓ Trial {trial_number}: {status} | {self.total_trials_completed}/{n_trials} complete {save_info}")
                    
                    try:
                        # study.best_trial raises an exception if no COMPLETE trials exist
                        best_trial = study.best_trial
                        if best_trial:
                            logger.info(
                                f"   Best: {study.best_value:.6f} (trial {best_trial.number})"
                            )
                    except (ValueError, RuntimeError):
                        # This happens when there are no COMPLETE trials (only PRUNED)
                        pass  # Skip logging if no successful trials yet

                    # Always mirror local checkpoint if configured
                    local_save_success = False
                    if self.local_checkpoint_dir:
                        local_save_success = persist_save_study_to_local(
                            self.local_checkpoint_dir,
                            study,
                            self.total_trials_completed,
                            sampler_name=self.sampler_name,
                            pruner_name=self.pruner_name,
                            study_name=self.study_name,
                            config_overrides=self.persistent_config_overrides,
                        )

                    # Periodic WandB upload (when save_every_n_trials is reached)
                    wandb_save_succeeded = False
                    if trials_in_batch >= self.save_every_n_trials:
                        # Try WandB save if configured
                        if self.wandb_project:
                            wandb_save_succeeded = persist_save_study_to_wandb(
                                self.wandb_project,
                                study_name=self.study_name,
                                study=study,
                                total_trials_completed=self.total_trials_completed,
                                sampler_name=self.sampler_name,
                                pruner_name=self.pruner_name,
                                config_overrides=self.persistent_config_overrides,
                            )
                            if wandb_save_succeeded:
                                logger.info(f"☁️  Uploaded to WandB (trial {self.total_trials_completed})")
                            else:
                                logger.warning("⚠️  WandB save failed")

                        # Reset batch counter after WandB upload attempt
                        if wandb_save_succeeded or not self.wandb_project:
                            last_checkpoint_trial_count = self.total_trials_completed
                            trials_in_batch = 0

                    # Check for pause or quit request BEFORE restart logic
                    # This ensures 'p' works even with restart_every_trial=True
                    pause_check_result = self._update_pause_from_keyboard()
                    # ALWAYS log pause state (not just debug) to diagnose issues
                    logger.info(f"[PAUSE] After trial {trial_number}: pause_check={pause_check_result}, "
                               f"_pause_requested={self._pause_requested}, "
                               f"restart_on_save={self.restart_on_save}, "
                               f"restart_every_trial={self.restart_every_trial}")
                    if pause_check_result:
                        self.should_pause = True
                        logger.info("\n⏸️  Executing pause after trial completion...")
                        logger.info(f"   Breaking out of trial loop (will NOT call sys.exit(42))")
                        if self.wandb_project:
                            logger.info("   Study will be saved to WandB for easy resume")
                        break
                    if self._quit_after_current:
                        logger.info("\n🛑 Quit requested. Stopping after current trial.")
                        self.should_pause = True
                        break

                    # Handle restart after trial (only if not pausing)
                    # When restart_every_trial=True, restart after EVERY trial for complete memory isolation
                    # When restart_every_trial=False, only restart when WandB upload happens
                    if self.restart_on_save:
                        should_restart = False
                        if self.restart_every_trial:
                            # Restart after every trial if local save succeeded
                            should_restart = local_save_success
                        else:
                            # Legacy behavior: only restart when save_every_n_trials is reached
                            should_restart = (wandb_save_succeeded or (not self.wandb_project and local_save_success and trials_in_batch == 0))

                        if should_restart:
                            resume_path = "latest" if self.wandb_project else str(self.local_checkpoint_dir)
                            if self.restart_every_trial:
                                logger.info(f"\n🔄 Per-trial restart: Exiting for process restart (memory isolation)")
                            else:
                                logger.info(f"\n🔄 restart_on_save enabled: Exiting after save for process restart")
                            logger.info(f"   Study saved with {self.total_trials_completed} trials")
                            logger.info(f"   Resume with: --resume-from {resume_path}")
                            import sys
                            sys.exit(42)  # Exit code 42 signals successful save + restart
                else:
                    # Trial failed (actual error, not pruning)
                    logger.info(f"✗ Trial {trial_number}: ❌ FAILED | {self.total_trials_completed}/{n_trials} complete")
                    
                    # Check for pause or quit request after failed trial
                    if self._update_pause_from_keyboard():
                        self.should_pause = True
                        logger.info("\n⏸️  Executing pause after failed trial...")
                        if self.wandb_project:
                            logger.info("   Study will be saved to WandB for easy resume")
                        break
                    if self._quit_after_current:
                        logger.info("\n🛑 Quit requested. Stopping after current trial.")
                        self.should_pause = True
                        break
                    
            except KeyboardInterrupt:
                # Clean up keyboard handler before terminating
                if self._use_keyboard_service and self.keyboard_service:
                    try:
                        self.keyboard_service.unregister_subscriber("hpo_pause")
                        logger.info("✅ Unregistered from KeyboardHandlerService")
                    except Exception:
                        pass
                elif self.keyboard_handler and hasattr(self.keyboard_handler, 'stop_monitoring'):
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
                if self._update_pause_from_keyboard():
                    self.should_pause = True
                    logger.info("\n⏸️  Executing pause after error...")
                    if self.wandb_project:
                        logger.info("   Study will be saved to WandB for easy resume")
                    # Break out of loop to trigger save logic
                    break
                    
                # Continue with next trial if not pausing
                continue
        
        # Stop keyboard monitoring and clear pause state
        if self._use_keyboard_service and self.keyboard_service:
            # Unregister from KeyboardHandlerService
            try:
                self.keyboard_service.unregister_subscriber("hpo_pause")
                logger.info("✅ Unregistered from KeyboardHandlerService")
            except Exception:
                pass
        elif self.keyboard_handler and hasattr(self.keyboard_handler, 'stop_monitoring'):
            # Stop legacy ImprovedKeyboardHandler
            try:
                self.keyboard_handler.stop_monitoring()
            except Exception:
                pass
            # Stop background polling thread (only used with legacy handler)
            self._stop_pause_polling_thread()
        self._pause_requested = False
        
        # Handle pause save or final save
        study_was_saved = False
        if self.should_pause:
            # Always save local
            if self.local_checkpoint_dir:
                self.save_study_to_local(study, self.total_trials_completed)
            if self.wandb_project:
                logger.info(f"💾 Saving study state for pause (with {self.total_trials_completed} finished trials)")
                study_was_saved = self.save_study_to_wandb(study, self.total_trials_completed)
                if study_was_saved:
                    last_checkpoint_trial_count = self.total_trials_completed
                    # NOTE: For pause, we do NOT call sys.exit(42) because that would
                    # trigger a restart. We want to exit cleanly. Exit code 43 signals
                    # "pause requested - do not restart"
                else:
                    logger.error("⚠️  Failed to save study for pause - checkpoint may be incomplete")
        elif self.wandb_project and self.total_trials_completed > last_checkpoint_trial_count:
            # Regular final save - only if we have new finished trials since last save
            logger.info(f"💾 Saving final state with {self.total_trials_completed} finished trials")
            study_was_saved = self.save_study_to_wandb(study, self.total_trials_completed)
        elif self.wandb_project and not self.should_pause:
            logger.info(f"ℹ️  No new finished trials to save since last checkpoint")

        # Debug logging to diagnose pause detection issues
        logger.debug(f"[PAUSE DEBUG] Loop exited: should_pause={self.should_pause}, "
                    f"total_trials_completed={self.total_trials_completed}/{n_trials}, "
                    f"_pause_requested={self._pause_requested}")

        if self.should_pause:
            logger.info(f"\n⏸️  OPTIMIZATION PAUSED | {self.total_trials_completed}/{n_trials} trials complete")
            if self.wandb_project:
                if study_was_saved:
                    logger.info(f"\n📝 To resume, run:")
                    resume_cmd = self._build_resume_command()
                    logger.info(f"   {resume_cmd}")
                else:
                    logger.info(f"⚠️  Failed to save study to WandB - cannot resume from WandB")
                    logger.info(f"   Check logs above for save errors")
            else:
                # No WandB, but local checkpoint should be available
                if self.local_checkpoint_dir:
                    logger.info(f"💾 Study saved to local checkpoint: {self.local_checkpoint_dir}")
                    logger.info(f"\n📝 To resume, run:")
                    try:
                        local_path = str(self.local_checkpoint_dir)
                        local_resume_cmd = persist_build_local_resume_command(self._original_argv, "scripts/world_model_hpo_optuna.py", local_path)
                        logger.info(f"   {local_resume_cmd}")
                    except Exception as e:
                        logger.warning(f"   Could not generate resume command: {e}")
                    logger.info(f"\n💡 For WandB cloud storage, add: --wandb <project-name>")
                else:
                    logger.warning(f"⚠️  No checkpoint saved (no WandB or local checkpoint configured)")
                    logger.info(f"   To enable resume, use --wandb <project-name>")
            # Also show local resume path for WandB users (backup option)
            if self.wandb_project and self.local_checkpoint_dir:
                try:
                    local_path = str(self.local_checkpoint_dir)
                    local_resume_cmd = persist_build_local_resume_command(self._original_argv, "scripts/world_model_hpo_optuna.py", local_path)
                    logger.info(f"   Local backup: {local_resume_cmd}")
                except Exception:
                    pass
            logger.info(f"{'='*60}")
        else:
            logger.info(f"\n{'='*60}")
            logger.info(f"✨ OPTIMIZATION COMPLETE!")
            percentage = (self.total_trials_completed / n_trials * 100) if n_trials > 0 else 0
            logger.info(f"Total trials run: {self.total_trials_completed}/{n_trials} ({percentage:.1f}%)")
            logger.info(f"{'='*60}")

        # Always mirror a local checkpoint at the end (even if 0 finished trials)
        if self.local_checkpoint_dir:
            try:
                persist_save_study_to_local(
                    self.local_checkpoint_dir,
                    study,
                    self.total_trials_completed,
                    sampler_name=self.sampler_name,
                    pruner_name=self.pruner_name,
                    study_name=self.study_name,
                    config_overrides=self.persistent_config_overrides,
                )
            except Exception:
                pass
        
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
    def create_objective(self):
        """Stub method to allow tests to patch a custom objective on the instance."""
        return None

    def _extract_persistable_args(self) -> Dict[str, Any]:
        """Minimal blacklist-based extractor for CLI args.

        - Source from dict or argparse.Namespace via vars()
        - Drop a tiny blacklist of routing/infra args
        - Persist only simple types (str/int/float/bool/Path->str)
        - Skip private keys (start with "_")
        """
        blacklist = {
            "resume_from",
            "study_name",
            "experiment_dir",
            "local_checkpoint_dir",
            "wandb_run_id",
            # keep optional tiny extras to reduce noise
            "config",
            "config_override",
            "config_overrides",
        }

        def simple(v: Any):
            from pathlib import Path
            if isinstance(v, (str, int, float, bool)):
                return v
            if isinstance(v, Path):
                return str(v)
            return None

        try:
            raw: Dict[str, Any]
            if isinstance(self.args, dict):
                raw = dict(self.args)
            elif hasattr(self.args, "__dict__"):
                raw = dict(vars(self.args))  # type: ignore[arg-type]
                # Fallback: include class-level attributes if instance __dict__ is empty
                if not raw:
                    raw = {}
                    for name in dir(self.args):
                        if name.startswith('_'):
                            continue
                        try:
                            val = getattr(self.args, name)
                        except Exception:
                            continue
                        if callable(val):
                            continue
                        sv = simple(val)
                        if sv is not None:
                            raw[name] = sv
            else:
                # Final fallback: reflect attributes directly
                raw = {}
                for name in dir(self.args):
                    if name.startswith('_'):
                        continue
                    try:
                        val = getattr(self.args, name)
                    except Exception:
                        continue
                    if callable(val):
                        continue
                    sv = simple(val)
                    if sv is not None:
                        raw[name] = sv
        except Exception:
            return {}

        out: Dict[str, Any] = {}
        for key, val in raw.items():
            if not key or key.startswith("_") or key in blacklist:
                continue
            sv = simple(val)
            if sv is not None:
                out[key] = sv
        return out

    def _update_pause_from_keyboard(self) -> bool:
        """Poll keyboard handler and handle pause/quit keys.

        - 'p' toggles a scheduled pause at the next trial boundary
        - 'q' requests immediate quit (sets flag to stop loop)
        - Ctrl+C is handled separately by KeyboardInterrupt

        Returns True if pause is currently requested.
        """
        # If using KeyboardHandlerService, state is managed by callback (thread-safe)
        if self._use_keyboard_service:
            with self._pause_lock:
                result = self._pause_requested
                if result:
                    logger.debug(f"[PAUSE DEBUG] KeyboardService path: returning {result}")
                return result

        # If background polling is active, use lock for thread-safe access
        if getattr(self, '_polling_active', False):
            with self._pause_lock:
                result = self._pause_requested
            logger.debug(f"[PAUSE DEBUG] Polling path: _pause_requested={result}")
            return result

        # If we reach here without _use_keyboard_service or _polling_active,
        # log a warning as this is unexpected with ImprovedKeyboardHandler
        if not getattr(self, '_manual_poll_warning_logged', False):
            logger.debug(f"[PAUSE DEBUG] Using manual polling path (use_keyboard_service={self._use_keyboard_service}, polling_active={getattr(self, '_polling_active', False)})")
            self._manual_poll_warning_logged = True
        try:
            if self.keyboard_handler and hasattr(self.keyboard_handler, 'get_key'):
                key = self.keyboard_handler.get_key()
                if key:
                    # Normalize key; handle control characters
                    raw = str(key)
                    skey = raw.lower()
                    if skey == 'p':
                        self._pause_requested = not self._pause_requested
                        if self._pause_requested:
                            print("\n⏸️  Pause SCHEDULED ('p' pressed)", flush=True)
                        else:
                            print("\n❌ Pause CANCELLED ('p' pressed again)", flush=True)
                    elif skey == 'q':
                        # Request quit after current trial ends
                        self._quit_after_current = True
                        print("\n🛑 Quit requested ('q' pressed). Will stop after current trial.", flush=True)
                    elif raw == "\x03":  # Ctrl+C in cbreak mode
                        # Treat as a graceful pause request so we save state
                        self._pause_requested = True
                        self.should_pause = True
                        print("\n⏸️  Ctrl+C detected. Pausing gracefully at trial boundary...", flush=True)
        except Exception as e:
            logger.debug(f"[PAUSE DEBUG] Exception in manual polling: {e}")
        result = self._pause_requested
        if result:
            logger.debug(f"[PAUSE DEBUG] Manual polling returning: {result}")
        return result

    def _build_resume_command(self) -> str:
        """Construct a minimal resume command that preserves only what's necessary.

        Minimal W&B resume requires:
        - script path
        - --wandb <project> (so artifact namespace is correct)
        - --study-name <name> (to select the exact study artifact)
        - --resume-from latest
        """
        script = self._original_argv[0] if (self._original_argv and self._original_argv[0]) else "scripts/world_model_hpo_optuna.py"
        parts: List[str] = ["python", script]

        wandb_proj = parse_cli_arg(self._original_argv, "wandb") or (str(self.wandb_project) if self.wandb_project else None)
        study_name = parse_cli_arg(self._original_argv, "study-name") or (str(self.study_name) if self.study_name else None)
        if wandb_proj:
            parts += ["--wandb", wandb_proj]
        if study_name:
            parts += ["--study-name", study_name]

        ts = parse_cli_arg(self._original_argv, "trial-steps")
        if not ts:
            try:
                ts = self.persistent_config_overrides.get("args.trial_steps") or self.persistent_config_overrides.get("trial_steps")
            except Exception:
                ts = None
        if ts:
            parts += ["--trial-steps", str(ts)]

        parts += ["--resume-from", "latest"]
        return " ".join(parts)

    def _build_local_resume_command(self, local_path: str) -> str:
        """Construct a minimal local resume command using filesystem path only."""
        script = self._original_argv[0] if (self._original_argv and self._original_argv[0]) else "scripts/world_model_hpo_optuna.py"
        # Include trial steps if known (prefer original argv, else persistent overrides)
        ts = parse_cli_arg(self._original_argv, "trial-steps")
        if not ts:
            try:
                ts = self.persistent_config_overrides.get("args.trial_steps") or self.persistent_config_overrides.get("trial_steps")
            except Exception:
                ts = None

        ts_arg = f" --trial-steps {ts}" if ts else ""
        return f"python {script}{ts_arg} --resume-from {local_path}"

    # --- Local checkpoint helpers ----------------------------------------
    def _build_args_config_overrides(self):
        """Build config overrides from args."""
        if not self.args:
            return
        # Use safe extractor to avoid persisting MagicMock internals
        args_dict = self._extract_persistable_args()

        for arg_name, arg_value in (args_dict or {}).items():
            # Skip excluded args (n_trials should NOT be persisted)
            if arg_name in self.args_exclude:
                continue
            # Skip None values only (but keep False boolean values)
            if arg_value is None:
                continue

            config_key = f"args.{arg_name}"
            self.persistent_config_overrides[config_key] = arg_value

    def save_study_to_local(self, study: optuna.Study, total_trials_completed: int) -> bool:
        """Save study to local checkpoint. Delegates to persist_save_study_to_local."""
        if not self.local_checkpoint_dir:
            return False
        return persist_save_study_to_local(
            self.local_checkpoint_dir,
            study,
            total_trials_completed,
            sampler_name=self.sampler_name,
            pruner_name=self.pruner_name,
            study_name=self.study_name,
            config_overrides=self.persistent_config_overrides,
        )

    def save_study_to_wandb(self, study: optuna.Study, total_trials_completed: int) -> bool:
        """Save study to WandB. Delegates to persist_save_study_to_wandb."""
        if not self.wandb_project:
            return False
        return persist_save_study_to_wandb(
            self.wandb_project,
            study_name=self.study_name,
            study=study,
            total_trials_completed=total_trials_completed,
            sampler_name=self.sampler_name,
            pruner_name=self.pruner_name,
            config_overrides=self.persistent_config_overrides,
        )

    def load_study_from_local(self, path_or_dir: Optional[str]) -> Optional[Dict[str, Any]]:
        candidate: Optional[Path] = None
        try:
            if path_or_dir and os.path.exists(path_or_dir):
                p = Path(path_or_dir)
                candidate = p if p.is_file() else (p / "study.pkl")
            elif self.local_checkpoint_dir:
                candidate = self.local_checkpoint_dir / "study.pkl"
            if not candidate or not candidate.exists():
                return None
            with open(candidate, 'rb') as f:
                session_info = pickle.load(f)
            logger.info(f"✅ Loaded local study: {candidate}")
            return session_info
        except Exception as e:
            logger.error(f"Failed to load local study: {e}")
            return None

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
        while getattr(self, '_polling_active', False):
            try:
                if self.keyboard_handler and hasattr(self.keyboard_handler, 'get_key'):
                    key = self.keyboard_handler.get_key()
                    if key:
                        self._handle_key_input(str(key))
            except Exception:
                # Ignore read errors
                pass
            time.sleep(0.05)