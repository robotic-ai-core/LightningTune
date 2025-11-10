"""
High-level HPO Runner that encapsulates all CLI handling, persistence, and resume logic.

This provides a simple interface for running HPO experiments without dealing with
argument parsing, checkpoint restoration, or persistence details.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Type, Union, Callable, List
import pickle

from lightning import LightningModule
from lightning.pytorch.callbacks import Callback

from .optuna.pausible_optimizer import PausibleOptunaOptimizer
from .utils import load_yaml_config, deep_merge_configs

logger = logging.getLogger(__name__)

# Import PauseCallback - graceful fallback if LightningReflow not available
try:
    from lightning_reflow.callbacks.pause.pause_callback import PauseCallback
    HAS_PAUSE_CALLBACK = True
except ImportError:
    HAS_PAUSE_CALLBACK = False
    logger.debug("LightningReflow not available, pause functionality will be limited")


class HPORunner:
    """
    High-level runner for HPO experiments with automatic CLI handling.

    This class encapsulates all the boilerplate for:
    - Command-line argument parsing
    - Checkpoint loading and saving
    - Argument persistence and restoration
    - Resume logic

    Example:
        >>> runner = HPORunner(
        ...     model_class=MyModel,
        ...     datamodule_class=MyDataModule,
        ...     search_space=my_search_space,
        ...     base_config="config.yaml"
        ... )
        >>> runner.run_from_cli()
    """

    # Default CLI arguments for all HPO experiments
    DEFAULT_CLI_ARGS = {
        'config': {'type': str, 'default': None, 'help': 'Path to base configuration file (YAML)'},
        'n_trials': {'type': int, 'default': 50, 'help': 'Number of trials to run'},
        'sampler': {'type': str, 'default': 'tpe', 'choices': ['tpe', 'random', 'cmaes', 'botorch']},
        'pruner': {'type': str, 'default': 'hyperband', 'choices': ['median', 'hyperband', 'successivehalving', 'none']},
        'trial_steps': {'type': int, 'default': None, 'help': 'Max steps per trial'},
        # Align with reference script: allow overriding validation interval via CLI
        'val_interval': {'type': int, 'default': None, 'help': 'Validation check interval (steps)'},
        'save_every': {'type': int, 'default': 10, 'help': 'Save checkpoint every N trials'},
        'restart_on_save': {'type': bool, 'default': False, 'action': 'store_true', 'help': 'Exit after saving (for auto-restart wrapper)'},
        'wandb': {'type': str, 'default': None, 'help': 'WandB project name'},
        'study_name': {'type': str, 'default': None, 'help': 'Study name'},
        'resume_from': {'type': str, 'default': None, 'help': 'Resume from checkpoint'},
        'experiment_dir': {'type': str, 'default': None, 'help': 'Directory for results'},
        'use_reflow': {'type': bool, 'default': True, 'action': 'store_true'},
        'no_reflow': {'type': bool, 'default': False, 'action': 'store_true'},
        'test_mode': {'type': bool, 'default': False, 'action': 'store_true'},
        'enable_pause': {'type': bool, 'default': None, 'action': 'store_true', 'help': 'Enable interactive pause (press p to pause)'},
        'disable_pause': {'type': bool, 'default': False, 'action': 'store_true', 'help': 'Disable interactive pause'},
        'pause_key': {'type': str, 'default': None, 'help': 'Key to trigger pause (default: p)'},
    }

    # Arguments that should not be persisted
    # - resume_from: Should not auto-resume on fresh runs
    # - study_name: Allow forking studies with new names
    # Note: n_trials IS persistent so resume continues to original target
    NON_PERSISTENT_ARGS = {'resume_from', 'study_name'}

    def __init__(
        self,
        model_class: Type[LightningModule],
        datamodule_class: Optional[Type] = None,
        search_space: Union[Callable, Any] = None,
        base_config: Union[str, Dict[str, Any]] = None,
        override_config: Optional[Union[str, Dict[str, Any]]] = None,
        additional_cli_args: Optional[Dict[str, Dict]] = None,
        callbacks: Optional[List[Callback]] = None,
        default_study_name: Optional[str] = None,
        enable_pause: bool = True,
        pause_key: str = 'p',
    ):
        """
        Initialize HPO runner.

        Args:
            model_class: PyTorch Lightning model class
            datamodule_class: PyTorch Lightning datamodule class
            search_space: Function or OptunaSearchSpace defining parameters to optimize
            base_config: Base configuration file or dict
            override_config: Optional override configuration
            additional_cli_args: Additional CLI arguments specific to this experiment
            callbacks: Lightning callbacks to use
            default_study_name: Default study name if not specified via CLI
            enable_pause: Enable interactive pause functionality (press 'p' to pause)
            pause_key: Key to trigger pause (default 'p')
        """
        self.model_class = model_class
        self.datamodule_class = datamodule_class
        self.search_space = search_space
        self.base_config = base_config
        self.override_config = override_config
        self.callbacks = callbacks or []
        self.default_study_name = default_study_name
        self.enable_pause = enable_pause
        self.pause_key = pause_key

        # Merge additional CLI args with defaults
        self.cli_args = self.DEFAULT_CLI_ARGS.copy()
        if additional_cli_args:
            self.cli_args.update(additional_cli_args)

        # Parsed arguments will be stored here
        self.args = None
        self.config_overrides = {}

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with all defined CLI arguments."""
        parser = argparse.ArgumentParser(
            description="HPO experiment runner",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        for arg_name, arg_spec in self.cli_args.items():
            arg_type = arg_spec.get('type', str)
            default = arg_spec.get('default')
            help_text = arg_spec.get('help', '')
            choices = arg_spec.get('choices')
            action = arg_spec.get('action')

            # Convert underscore to hyphen for CLI
            cli_name = '--' + arg_name.replace('_', '-')

            kwargs = {
                'default': default,
                'help': help_text,
            }

            if action:
                kwargs['action'] = action
            else:
                kwargs['type'] = arg_type

            if choices:
                kwargs['choices'] = choices

            parser.add_argument(cli_name, **kwargs)

        return parser

    def _was_arg_specified(self, arg_name: str, argv: Optional[List[str]] = None) -> bool:
        """Check if an argument was explicitly specified on command line."""
        arg_patterns = [
            f'--{arg_name.replace("_", "-")}',
            f'--{arg_name}',
        ]
        # Use provided argv or fall back to sys.argv
        cmd_args = ' '.join(argv if argv is not None else sys.argv)
        return any(pattern in cmd_args for pattern in arg_patterns)

    def _parse_dot_notation_args(self, unknown_args: List[str]) -> Dict[str, Any]:
        """
        Parse Lightning CLI-style dot-notation arguments from unknown args.

        Supports arguments like:
            --data.batch_size 512
            --model.learning_rate 1e-4
            --trainer.max_epochs 100

        Args:
            unknown_args: List of unknown arguments from argparse

        Returns:
            Dictionary of config overrides with dot-notation keys
        """
        config_overrides = {}
        i = 0

        while i < len(unknown_args):
            arg = unknown_args[i]

            # Check if this looks like a config argument (starts with -- and contains .)
            if arg.startswith('--') and '.' in arg:
                key = arg[2:]  # Remove --

                # Get the value (next argument)
                if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                    value_str = unknown_args[i + 1]

                    # Auto-detect type
                    value = self._auto_convert_type(value_str)

                    config_overrides[key] = value
                    i += 2  # Skip both arg and value
                else:
                    # Flag without value (treat as True)
                    config_overrides[key] = True
                    i += 1
            else:
                i += 1

        return config_overrides

    def _auto_convert_type(self, value_str: str) -> Any:
        """
        Automatically convert string value to appropriate Python type.

        Args:
            value_str: String value from command line

        Returns:
            Converted value (int, float, bool, or str)
        """
        # Try boolean
        if value_str.lower() in ('true', 'false'):
            return value_str.lower() == 'true'

        # Try int
        try:
            return int(value_str)
        except ValueError:
            pass

        # Try float
        try:
            return float(value_str)
        except ValueError:
            pass

        # Return as string
        return value_str

    def _load_checkpoint(self, resume_from: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint from file or WandB."""
        # First try local file
        if os.path.exists(resume_from):
            with open(resume_from, 'rb') as f:
                checkpoint = pickle.load(f)
                return checkpoint

        # Try WandB if project is configured
        if self.args.wandb:
            # Create temporary optimizer just to load from WandB
            temp_optimizer = PausibleOptunaOptimizer(
                base_config={"dummy": "config"},
                search_space=lambda trial: {},
                model_class=None,
                datamodule_class=None,
                wandb_project=self.args.wandb,
                study_name=self.args.study_name or self.default_study_name,
            )
            checkpoint = temp_optimizer.load_study_from_wandb(resume_from)
            return checkpoint

        # Try local checkpoint directory for "latest" (no WandB)
        if resume_from == "latest":
            from pathlib import Path
            # Try to find local checkpoint in checkpoints/<study_name>/study.pkl
            study_name = self.args.study_name or self.default_study_name
            local_checkpoint_path = Path("checkpoints") / study_name / "study.pkl"
            if local_checkpoint_path.exists():
                logger.info(f"📂 Found local checkpoint: {local_checkpoint_path}")
                with open(local_checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                    return checkpoint

        logger.error(f"❌ Could not load checkpoint: {resume_from}")
        return None

    def _restore_args_from_checkpoint(self, checkpoint: Dict[str, Any], argv: Optional[List[str]] = None) -> None:
        """Restore arguments from checkpoint, respecting explicit overrides."""
        if not checkpoint or "config_overrides" not in checkpoint:
            logger.warning("  ⚠️ No config_overrides in checkpoint")
            return

        saved_overrides = checkpoint["config_overrides"]
        restored_count = 0
        overridden_count = 0

        # Normalize checkpoint format: convert bare 'n_trials' to 'args.n_trials'
        # This handles legacy checkpoints that may have both formats
        if 'n_trials' in saved_overrides:
            bare_n_trials = saved_overrides.get('n_trials')
            args_n_trials = saved_overrides.get('args.n_trials')

            if args_n_trials is None:
                # Only bare exists, convert it
                saved_overrides['args.n_trials'] = bare_n_trials
                logger.debug(f"Normalized: n_trials={bare_n_trials} → args.n_trials")
            elif bare_n_trials and args_n_trials and bare_n_trials != args_n_trials:
                # Both exist with different values, use the larger (likely extended)
                saved_overrides['args.n_trials'] = max(bare_n_trials, args_n_trials)
                logger.debug(f"Resolved conflict: using n_trials={saved_overrides['args.n_trials']}")

            # Remove bare version to avoid confusion
            del saved_overrides['n_trials']

        # Now process all args uniformly
        for key, value in saved_overrides.items():
            if key.startswith("args."):
                arg_name = key[5:]  # Remove "args." prefix

                # Skip non-persistent args
                if arg_name in self.NON_PERSISTENT_ARGS:
                    continue

                if hasattr(self.args, arg_name):
                    # Check if explicitly specified
                    was_specified = self._was_arg_specified(arg_name, argv)
                    current_value = getattr(self.args, arg_name)

                    if was_specified:
                        # Keep the specified value
                        if current_value != value:
                            overridden_count += 1
                    else:
                        # Restore saved value
                        setattr(self.args, arg_name, value)
                        restored_count += 1

        if restored_count > 0:
            logger.info(f"  ✓ Restored {restored_count} saved arguments")
        if overridden_count > 0:
            logger.info(f"  📈 Overrode {overridden_count} arguments with new values")

    def _build_config_overrides(self) -> Dict[str, Any]:
        """Build configuration overrides from parsed arguments."""
        config_overrides = {}

        # Add all non-None arguments
        for arg_name, arg_value in vars(self.args).items():
            # Skip non-persistent args
            if arg_name in self.NON_PERSISTENT_ARGS:
                continue

            # Skip None values
            if arg_value is None:
                continue

            # Store as config override
            config_key = f"args.{arg_name}"
            config_overrides[config_key] = arg_value

        # Add any specific trainer overrides
        if self.args.test_mode:
            config_overrides["trainer.limit_val_batches"] = 5
            config_overrides["trainer.limit_train_batches"] = 10

        return config_overrides

    def _merge_configs(self) -> Union[str, Dict[str, Any]]:
        """Merge base and override configs if needed."""
        if not self.override_config:
            return self.base_config

        # Load configs
        if isinstance(self.base_config, str):
            base_dict = load_yaml_config(self.base_config)
        else:
            base_dict = self.base_config

        if isinstance(self.override_config, str):
            override_dict = load_yaml_config(self.override_config)
        else:
            override_dict = self.override_config

        # Merge
        return deep_merge_configs(base_dict, override_dict)

    def run_with_auto_restart(self, argv: Optional[List[str]] = None) -> int:
        """
        Run HPO with automatic subprocess restarts for complete resource cleanup.

        This orchestrator spawns child processes to run the actual optimization.
        Each save triggers a clean process restart, ensuring complete resource
        cleanup (GPU memory, DataLoaders, thread pools, file handles, etc.).

        Edge Cases Handled:
        - User's initial --resume-from path is overridden on first restart (intentional)
        - Subprocess crashes propagate exit code to parent
        - Ctrl+C is caught both in child (returns 130) and parent (exception handler)
        - Environment flag prevents infinite nested restarts

        Args:
            argv: Optional command line arguments (for testing)

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        import subprocess

        restart_count = 0
        max_restarts = 100  # Safety limit - if we hit this, something is wrong

        # Build command for subprocess
        if argv is None:
            cmd = [sys.executable] + sys.argv
        else:
            # For testing: validate sys.argv[0] is a valid script path
            script_path = sys.argv[0]
            if not script_path or not Path(script_path).exists():
                raise RuntimeError(
                    f"Cannot construct subprocess command: sys.argv[0]='{script_path}' is not a valid script path. "
                    "This usually means run_with_auto_restart() was called from an unexpected context (e.g., pytest module mode)."
                )
            cmd = [sys.executable, script_path] + argv

        # Ensure restart-on-save is enabled (for child processes)
        # Note: Check both --restart-on-save and --restart-on-save=... forms
        has_restart_flag = any(
            arg == "--restart-on-save" or arg.startswith("--restart-on-save=")
            for arg in cmd
        )
        if not has_restart_flag:
            cmd.append("--restart-on-save")

        # Add environment flag to prevent nested auto-restart
        # Note: This copies all environment variables, which is standard practice
        # for subprocess.run(). Be aware credentials are propagated.
        env = os.environ.copy()
        env["LIGHTNING_TUNE_NO_AUTO_RESTART"] = "1"

        logger.info("="*60)
        logger.info("🔄 HPO Auto-Restart Mode")
        logger.info("="*60)
        logger.info("This will automatically restart the process after each save")
        logger.info("for complete resource cleanup (GPU memory, DataLoaders, etc.)")
        logger.info("Press Ctrl+C to stop gracefully")
        logger.info("="*60)

        try:
            while restart_count < max_restarts:
                # Add --resume-from after first run
                if restart_count > 0:
                    # Remove any existing --resume-from flag and its argument
                    # This handles both --resume-from value and --resume-from=value forms
                    cleaned_cmd = []
                    skip_next = False
                    for i, arg in enumerate(cmd):
                        if skip_next:
                            skip_next = False
                            continue
                        if arg == "--resume-from":
                            # Skip both the flag and the next argument
                            skip_next = True
                            continue
                        if arg.startswith("--resume-from="):
                            # Skip --resume-from=value form
                            continue
                        cleaned_cmd.append(arg)

                    # Add new resume flag
                    cmd = cleaned_cmd + ["--resume-from", "latest"]

                    logger.info(f"\n{'='*60}")
                    logger.info(f"🔄 Restart #{restart_count}: Resuming from latest checkpoint")
                    logger.info(f"{'='*60}\n")
                else:
                    logger.info(f"\n🚀 Starting initial HPO run\n")

                # Run subprocess with generous timeout (24 hours)
                # Note: subprocess.run() without capture_output=True inherits parent's
                # stdout/stderr, so child output is visible to user
                timeout_seconds = 24 * 60 * 60  # 24 hours
                try:
                    result = subprocess.run(cmd, env=env, timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    logger.error(f"\n{'='*60}")
                    logger.error(f"❌ Child process timed out after {timeout_seconds}s")
                    logger.error(f"   Restart count: {restart_count}")
                    logger.error(f"{'='*60}")
                    return 1

                # Handle exit codes
                if result.returncode == 42:
                    # Successful save, restart
                    restart_count += 1
                    logger.info(f"\n✅ Save successful! Restarting process...")
                    continue

                elif result.returncode == 0:
                    # Normal completion
                    logger.info(f"\n{'='*60}")
                    logger.info(f"🎉 HPO Complete! Total restarts: {restart_count}")
                    logger.info(f"{'='*60}")
                    return 0

                elif result.returncode == 130:
                    # Ctrl+C (SIGINT)
                    logger.info(f"\n{'='*60}")
                    logger.info(f"⚠️  Interrupted by user")
                    logger.info(f"{'='*60}")
                    return 130

                else:
                    # Error - provide context for debugging
                    logger.error(f"\n{'='*60}")
                    logger.error(f"❌ HPO failed with exit code: {result.returncode}")
                    logger.error(f"   Restart count: {restart_count}")
                    logger.error(f"   Command: {' '.join(cmd)}")
                    logger.error(f"{'='*60}")
                    return result.returncode

            # Exceeded max restarts - something is wrong with save logic
            logger.error(f"\n{'='*60}")
            logger.error(f"❌ Exceeded maximum restarts ({max_restarts})")
            logger.error(f"   This usually indicates a problem with checkpoint saving.")
            logger.error(f"   Check save_every setting and checkpoint storage.")
            logger.error(f"{'='*60}")
            return 1

        except KeyboardInterrupt:
            # This catches Ctrl+C pressed in the brief moment between subprocess runs
            # (Child process Ctrl+C is caught by the returncode == 130 branch above)
            logger.info(f"\n⚠️  Ctrl+C received in parent. Exiting gracefully...")
            return 130

    def run_from_cli(self, argv: Optional[List[str]] = None) -> Any:
        """
        Run HPO from command line arguments.

        This is the main entry point that handles everything:
        - Parsing arguments (including dot-notation config args)
        - Loading checkpoints if resuming
        - Restoring saved arguments
        - Running optimization

        If --restart-on-save is enabled and this is the parent process,
        automatically runs in subprocess mode for resource cleanup.

        Args:
            argv: Optional command line arguments (for testing)

        Returns:
            Optuna study object (if running directly) or exit code (if orchestrating)
        """
        # Quick parse to check for restart-on-save flag
        # Use a temporary parser to avoid side effects
        temp_parser = self._create_parser()
        temp_args, _ = temp_parser.parse_known_args(argv)

        # Check if we should use auto-restart orchestration
        restart_on_save = getattr(temp_args, 'restart_on_save', False)
        # Robust check for environment flag (handles multiple truthy values)
        env_value = os.environ.get("LIGHTNING_TUNE_NO_AUTO_RESTART", "")
        is_child_process = env_value in ("1", "true", "TRUE", "True", "yes", "YES")

        if restart_on_save and not is_child_process:
            # Parent orchestrator mode - spawn child processes
            exit_code = self.run_with_auto_restart(argv)
            sys.exit(exit_code)
        else:
            # Child worker mode or normal mode - run actual optimization
            return self._run_optimization(argv)

    def _run_optimization(self, argv: Optional[List[str]] = None) -> Any:
        """
        Run the actual optimization (called by child process or directly if no auto-restart).

        This is the core optimization logic that was previously in run_from_cli.

        Args:
            argv: Optional command line arguments (for testing)

        Returns:
            Optuna study object
        """
        # Parse command line - use parse_known_args to capture dot-notation args
        parser = self._create_parser()
        self.args, unknown_args = parser.parse_known_args(argv)

        # Handle --config CLI argument (overrides __init__ base_config)
        if self.args.config is not None:
            config_path = Path(self.args.config)
            if not config_path.exists():
                logger.error(f"❌ Config file not found: {self.args.config}")
                sys.exit(1)
            logger.info(f"📄 Using config from CLI: {self.args.config}")
            self.base_config = self.args.config

        # Parse Lightning CLI-style dot-notation arguments
        dot_notation_overrides = self._parse_dot_notation_args(unknown_args)

        # Log parsed dot-notation args
        if dot_notation_overrides:
            logger.info(f"📝 Parsed {len(dot_notation_overrides)} dot-notation arguments:")
            for key, value in dot_notation_overrides.items():
                logger.info(f"   {key} = {value} ({type(value).__name__})")

        # Set default study name if not specified
        if not self.args.study_name:
            if self.default_study_name:
                self.args.study_name = self.default_study_name
            else:
                self.args.study_name = f"{self.args.sampler}_{self.args.pruner}_study"

        # Handle resume
        checkpoint_config_overrides = {}  # Initialize first to avoid UnboundLocalError
        if self.args.resume_from:
            logger.info(f"\n{'='*60}")
            logger.info(f"📂 RESUMING FROM CHECKPOINT")
            logger.info(f"{'='*60}")
            logger.info(f"Loading from: {self.args.resume_from}")

            checkpoint = self._load_checkpoint(self.args.resume_from)
            if checkpoint:
                # Pass sys.argv if argv is None so _was_arg_specified works correctly
                actual_argv = argv if argv is not None else sys.argv
                self._restore_args_from_checkpoint(checkpoint, actual_argv)

                completed = checkpoint.get('total_trials_completed', 0)

                logger.info(f"\n📊 Resume Status:")
                logger.info(f"{'─'*60}")
                logger.info(f"{'Trials completed':<35} {completed:<25}")
                logger.info(f"{'Target n_trials':<35} {self.args.n_trials:<25}")
                remaining = max(0, self.args.n_trials - completed)
                logger.info(f"{'Trials remaining':<35} {remaining:<25}")
                logger.info(f"{'─'*60}")

                if remaining == 0:
                    logger.info(f"\n✅ All {self.args.n_trials} trials already complete!")
                    logger.info(f"💡 To run more trials, use --n-trials with a value > {completed}")
                    logger.info(f"{'='*60}")

                # New: persist the loaded checkpoint to a local temp file and switch resume_from
                try:
                    import tempfile as _tempfile
                    with _tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as _tmpf:
                        pickle.dump(checkpoint, _tmpf)
                        _tmpf.flush()
                        os.fsync(_tmpf.fileno())
                        self.args.resume_from = _tmpf.name
                        logger.info(f"🔁 Using local checkpoint copy for resume: {self.args.resume_from}")
                except Exception:
                    # If this fails, keep original behavior
                    pass

                # Save checkpoint config_overrides for merging later
                # This preserves non-"args.*" config overrides (like data.batch_size, data.num_workers)
                # that were set in the original run
                checkpoint_config_overrides = checkpoint.get('config_overrides', {})
            else:
                # Resume was explicitly requested but failed - DO NOT start fresh!
                logger.error(f"\n{'='*60}")
                logger.error(f"❌ FATAL: Failed to resume from checkpoint")
                logger.error(f"{'='*60}")
                logger.error(f"Checkpoint path: {self.args.resume_from}")
                logger.error(f"\nPossible causes:")
                logger.error(f"  • Checkpoint file/artifact does not exist")
                logger.error(f"  • WandB project/study name mismatch")
                logger.error(f"  • File is corrupted or unreadable")
                logger.error(f"  • Network/authentication issues with WandB")
                logger.error(f"\n💡 To start a fresh study, remove the --resume-from flag")
                logger.error(f"{'='*60}\n")
                sys.exit(1)  # Exit with error - DO NOT start fresh study!

        # Build config overrides from final args
        self.config_overrides = self._build_config_overrides()

        # Merge checkpoint config overrides (for resume)
        # Restore config overrides from checkpoint, but skip:
        # 1. Non-persistent args (like n_trials which is extensible)
        # 2. Keys already set in self.config_overrides (new CLI takes precedence)
        for key, value in checkpoint_config_overrides.items():
            # Skip non-persistent args (e.g., args.n_trials, args.resume_from)
            if key.startswith("args."):
                arg_name = key[5:]  # Remove "args." prefix
                if arg_name in self.NON_PERSISTENT_ARGS:
                    continue

            # Restore if not already set by new CLI
            if key not in self.config_overrides:
                self.config_overrides[key] = value

        # Merge dot-notation overrides
        # These take precedence over defaults but are lower priority than explicit config overrides
        for key, value in dot_notation_overrides.items():
            if key not in self.config_overrides:
                self.config_overrides[key] = value

        # Align trainer overrides with reference world_model_hpo_optuna.py
        # These ensure consistent validation cadence and UI behavior during HPO
        try:
            # Base trainer behavior during HPO
            self.config_overrides["trainer.check_val_every_n_epoch"] = None
            self.config_overrides["trainer.enable_model_summary"] = False
            self.config_overrides["trainer.enable_progress_bar"] = True
            # Validation interval (steps)
            if getattr(self.args, 'val_interval', None) is not None:
                self.config_overrides["trainer.val_check_interval"] = self.args.val_interval
            else:
                self.config_overrides["trainer.val_check_interval"] = 1000
            # Trial steps (early stopping without affecting LR schedule)
            # DO NOT set trainer.max_steps here - that would break the LR schedule!
            # Instead, we'll add EarlyStoppingSteps callback below.
            # Extra limits for tests (speed up integration tests) via config override
            # Prefer config-driven approach instead of code-side trainer mutations.
            import os as _os
            if _os.environ.get('PYTEST_CURRENT_TEST') or _os.environ.get('FAST_HPO_TESTS'):
                print("🚀 FAST_HPO_TESTS mode: applying minimal test config", flush=True)
                self.config_overrides["trainer.limit_train_batches"] = 1
                # Use 2 val batches to ensure epoch aggregation works properly
                self.config_overrides["trainer.limit_val_batches"] = 2
                print(f"   Set limit_val_batches = 2", flush=True)
                self.config_overrides["trainer.num_sanity_val_steps"] = 0
                self.config_overrides["trainer.max_epochs"] = 1
                # Also ensure validation runs by setting check interval
                self.config_overrides["trainer.val_check_interval"] = 1
                print(f"   Set val_check_interval = 1", flush=True)
                # Disable expensive callbacks by overriding them to empty list
                # This happens BEFORE merge, so callbacks won't be instantiated
                self.config_overrides["trainer.callbacks"] = []
                print("   ✅ Disabled all callbacks for fast testing", flush=True)
        except Exception as e:
            print(f"⚠️  Error in FAST_HPO_TESTS config: {e}", flush=True)
            import traceback
            traceback.print_exc()

        # Merge configs
        final_config = self._merge_configs()

        # Determine whether to use reflow
        use_reflow = self.args.use_reflow and not self.args.no_reflow

        # Display configuration override table
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Starting HPO: {self.args.study_name}")
        logger.info(f"{'='*60}")

        # Build display table of all configuration
        logger.info(f"\n📋 Configuration:")
        logger.info(f"{'─'*60}")
        logger.info(f"{'Parameter':<35} {'Value':<25}")
        logger.info(f"{'─'*60}")

        # Display key arguments
        logger.info(f"{'n_trials':<35} {self.args.n_trials:<25}")
        logger.info(f"{'sampler':<35} {self.args.sampler:<25}")
        logger.info(f"{'pruner':<35} {self.args.pruner:<25}")

        if self.args.wandb:
            logger.info(f"{'wandb_project':<35} {self.args.wandb:<25}")

        if self.args.trial_steps:
            logger.info(f"{'trial_steps':<35} {self.args.trial_steps:<25}")

        logger.info(f"{'save_every':<35} {self.args.save_every:<25}")

        # Display all config overrides
        if self.config_overrides:
            for key, value in sorted(self.config_overrides.items()):
                # Skip args that we already displayed above
                if not key.startswith('args.'):
                    display_value = str(value)
                    if len(display_value) > 25:
                        display_value = display_value[:22] + "..."
                    logger.info(f"{key:<35} {display_value:<25}")

        logger.info(f"{'─'*60}")
        logger.info(f"{'='*60}\n")

        # Mark that args have been pre-restored by HPORunner
        if hasattr(self.args, '__dict__'):
            self.args._restored_by_hporunner = True

        # Add EarlyStoppingSteps callback if trial_steps is specified
        # This stops trials early WITHOUT affecting the LR schedule (preserves trainer.max_steps)
        if getattr(self.args, 'trial_steps', None) is not None:
            from LightningTune.callbacks import EarlyStoppingSteps
            early_stop_callback = EarlyStoppingSteps(
                stopping_steps=self.args.trial_steps,
                verbose=True
            )
            self.callbacks.append(early_stop_callback)
            logger.info(f"📍 Added EarlyStoppingSteps callback: will stop trials at {self.args.trial_steps} steps")
            logger.info(f"   (Preserves LR schedule by not modifying trainer.max_steps)")

        # Resolve pause settings from CLI args
        # CLI args override __init__ parameters
        final_enable_pause = self.enable_pause  # Default from __init__
        if hasattr(self.args, 'enable_pause') and self.args.enable_pause is not None:
            final_enable_pause = self.args.enable_pause
        if hasattr(self.args, 'disable_pause') and self.args.disable_pause:
            final_enable_pause = False

        final_pause_key = self.pause_key  # Default from __init__
        if hasattr(self.args, 'pause_key') and self.args.pause_key is not None:
            final_pause_key = self.args.pause_key

        # Note: We do NOT add PauseCallback during HPO runs
        # HPO uses PausibleOptunaOptimizer which implements trial-boundary pause
        # (pauses AFTER completing a trial, not during training at validation boundaries)
        # Adding PauseCallback here would cause validation-boundary pauses which:
        #   - Interrupt trials mid-training (corrupts trial metrics)
        #   - Prevents fair trial comparison in Optuna
        #   - Wastes GPU time on incomplete trials
        # For single training runs (not HPO), use train_world_model.py which includes PauseCallback

        if final_enable_pause:
            logger.info(f"⏸️  HPO pause enabled: press '{final_pause_key}' to pause at TRIAL boundary")
            logger.info(f"   (Trials will complete before pausing - controlled by PausibleOptunaOptimizer)")
        else:
            logger.info("⏸️  HPO pause disabled")

        # Determine an absolute local checkpoint directory so local resume paths are reliable
        # Use current working directory as the anchor to avoid module-relative saves
        try:
            base_ckpt_dir = Path.cwd() / "checkpoints"
            if self.args.wandb:
                _local_ckpt_dir = base_ckpt_dir / str(self.args.wandb) / str(self.args.study_name)
            else:
                _local_ckpt_dir = base_ckpt_dir / str(self.args.study_name)
        except Exception:
            _local_ckpt_dir = None

        optimizer = PausibleOptunaOptimizer(
            base_config=final_config,
            search_space=self.search_space,
            model_class=self.model_class,
            datamodule_class=self.datamodule_class,
            wandb_project=self.args.wandb,
            study_name=self.args.study_name,
            sampler_name=self.args.sampler,
            pruner_name=self.args.pruner,
            save_every_n_trials=self.args.save_every,
            restart_on_save=self.args.restart_on_save,
            enable_pause=final_enable_pause,
            use_reflow=use_reflow,
            experiment_dir=self.args.experiment_dir,
            args=self.args,
            persist_args=True,
            args_exclude=self.NON_PERSISTENT_ARGS,  # Pass non-persistent args to optimizer
            # Critical: disable Lightning checkpoints during HPO to avoid conflicts
            # with enable_checkpointing=False in HPO configs
            save_checkpoints=False,
            # Mirror study checkpoints locally under an absolute path for reliable resume
            local_checkpoint_dir=(str(_local_ckpt_dir) if _local_ckpt_dir is not None else None),
        )

        # Run optimization
        study = optimizer.optimize(
            n_trials=self.args.n_trials,
            resume_from=self.args.resume_from,
            config_overrides=self.config_overrides,
            callbacks=self.callbacks,
        )

        logger.info("\n✨ Optimization Complete!")

        return study

    def run(self, **kwargs) -> Any:
        """
        Run HPO programmatically (without CLI).

        Args:
            **kwargs: Arguments that would normally come from CLI
                Special handling for 'config_overrides': dict of dotted-path overrides
                e.g., {"data.init_args.num_workers": 4}

        Returns:
            Optuna study object
        """
        # Extract config_overrides dict before converting to CLI
        config_overrides = kwargs.pop('config_overrides', None)

        # Convert kwargs to CLI-style arguments
        argv = []

        # First, convert config_overrides dict to individual dot-notation CLI args
        # This preserves the dotted paths like "data.init_args.num_workers"
        if config_overrides and isinstance(config_overrides, dict):
            for key, value in config_overrides.items():
                # Add as --key value (dot notation is preserved in key)
                argv.extend([f'--{key}', str(value)])

        # Then convert remaining kwargs to CLI-style arguments
        for key, value in kwargs.items():
            if value is not None:
                cli_key = '--' + key.replace('_', '-')
                if isinstance(value, bool):
                    if value:
                        argv.append(cli_key)
                else:
                    argv.extend([cli_key, str(value)])

        return self.run_from_cli(argv)