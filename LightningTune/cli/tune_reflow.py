"""
TuneReflowCLI v2 - Properly handles state serialization for resume.

This version saves all necessary configuration to enable true resume capability.
"""

import sys
import signal
import logging
import threading
import time
import pickle
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Union
from dataclasses import dataclass, asdict, field
import hashlib

import ray
from ray import tune
from ray.tune import Tuner

try:
    from lightning_reflow.callbacks.pause.unified_keyboard_handler import UnifiedKeyboardHandler
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    UnifiedKeyboardHandler = None

try:
    from ..core.config import SearchSpace
    from ..core.strategies import OptimizationStrategy
except ImportError:
    # Fallback for direct imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.config import SearchSpace
    from core.strategies import OptimizationStrategy

logger = logging.getLogger(__name__)


@dataclass
class TuneSessionState:
    """Complete state needed to resume a Tune session."""
    
    # Experiment identification
    experiment_name: str
    experiment_dir: str
    session_id: str
    
    # Optimizer configuration
    base_config_path: str
    search_space_pickle: bytes  # Pickled SearchSpace object
    strategy_pickle: bytes  # Pickled Strategy object
    optimization_config_pickle: bytes  # Pickled OptimizationConfig
    
    # Runtime state
    pause_requested: bool = False
    pause_completed: bool = False
    trials_paused: int = 0
    total_trials: int = 0
    
    # Resume information
    checkpoint_dir: str = ""
    resume_command: str = ""
    original_argv: list = field(default_factory=list)
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    paused_at: Optional[float] = None
    
    def save(self, path: Path):
        """Save state to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle for complete object serialization
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        # Also save human-readable summary
        summary_path = path.with_suffix('.summary.yaml')
        summary = {
            'experiment_name': self.experiment_name,
            'experiment_dir': self.experiment_dir,
            'session_id': self.session_id,
            'base_config_path': self.base_config_path,
            'pause_requested': self.pause_requested,
            'trials_paused': self.trials_paused,
            'total_trials': self.total_trials,
            'checkpoint_dir': self.checkpoint_dir,
            'created_at': self.created_at,
            'paused_at': self.paused_at,
        }
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f)
    
    @classmethod
    def load(cls, path: Path) -> 'TuneSessionState':
        """Load state from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class TuneReflowCLI:
    """
    Enhanced CLI that properly saves and restores complete session state.
    
    This version saves:
    - The search space object
    - The strategy configuration
    - The optimization config
    - All necessary paths and parameters
    
    Example:
        ```python
        # First run - everything saved automatically
        cli = TuneReflowCLI(experiment_name="my_opt")
        results = cli.run(optimizer)
        
        # Resume - everything restored from checkpoint
        cli = TuneReflowCLI.resume(checkpoint_dir="./experiments/my_opt/session_xxx")
        results = cli.run()  # No need to pass optimizer!
        ```
    """
    
    def __init__(
        self,
        experiment_name: str,
        experiment_dir: str = "./experiments",
        session_id: Optional[str] = None,
        enable_pause: bool = True,
        verbose: bool = True,
    ):
        """Initialize CLI with session management."""
        self.experiment_name = experiment_name
        self.experiment_dir = Path(experiment_dir)
        self.enable_pause = enable_pause
        self.verbose = verbose
        
        # Generate session ID
        if session_id is None:
            # Create unique session ID based on timestamp and random component
            import uuid
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            self.session_id = f"{timestamp}_{unique_id}"
        else:
            self.session_id = session_id
        
        # Options
        self.pause_key = 'p'
        self.quit_key = 'q'
        
        # Session directory
        self.session_dir = self.experiment_dir / experiment_name / f"session_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # State file
        self.state_file = self.session_dir / "session_state.pkl"
        
        # Initialize state
        self.state: Optional[TuneSessionState] = None
        self._optimizer = None
        self._monitoring = False
        
        # Keyboard handler
        self._keyboard_handler: Optional[UnifiedKeyboardHandler] = None
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Pause signal file for communication with trials
        self.pause_signal_file = Path(f"/tmp/tune_pause_{experiment_name}_{self.session_id}.signal")
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def save_optimizer_config(self, optimizer: Any) -> TuneSessionState:
        """
        Save all optimizer configuration for resume.
        
        This is the KEY method that enables proper resume!
        """
        import pickle
        
        # Extract all necessary components
        base_config_path = str(optimizer.base_config_path)
        
        # Pickle the search space
        search_space_pickle = pickle.dumps(optimizer.search_space)
        
        # Pickle the strategy  
        strategy_pickle = pickle.dumps(optimizer.strategy)
        
        # Pickle the optimization config
        optimization_config_pickle = pickle.dumps(optimizer.optimization_config)
        
        # Create state
        state = TuneSessionState(
            experiment_name=self.experiment_name,
            experiment_dir=str(self.experiment_dir),
            session_id=self.session_id,
            base_config_path=base_config_path,
            search_space_pickle=search_space_pickle,
            strategy_pickle=strategy_pickle,
            optimization_config_pickle=optimization_config_pickle,
            checkpoint_dir=str(self.session_dir),
            original_argv=sys.argv.copy(),
        )
        
        # Save to disk
        state.save(self.state_file)
        
        if self.verbose:
            print(f"💾 Session state saved to: {self.state_file}")
        
        return state
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful pause/shutdown."""
        def signal_handler(signum, frame):
            if signum == signal.SIGINT:
                if not self.state or not self.state.pause_requested:
                    print("\n⏸️  Pause requested. Waiting for trials to reach validation boundaries...")
                    self._request_pause()
                else:
                    print("\n⛔ Force quit requested. Saving checkpoint and exiting...")
                    self._force_quit()
            elif signum == signal.SIGUSR1:
                # Alternative signal for remote processes
                self._request_pause()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGUSR1, signal_handler)
    
    def _start_keyboard_monitoring(self):
        """Start monitoring keyboard input."""
        if not self.enable_pause or not KEYBOARD_AVAILABLE:
            if not KEYBOARD_AVAILABLE and self.verbose:
                print("⚠️  Keyboard monitoring unavailable. Install lightning_reflow for interactive features.")
            return
        
        try:
            self._keyboard_handler = UnifiedKeyboardHandler()
            if self._keyboard_handler.is_available():
                self._keyboard_handler.start_monitoring()
                self._monitoring = True
                
                # Start monitor thread
                self._monitor_thread = threading.Thread(
                    target=self._keyboard_monitor_loop,
                    daemon=True
                )
                self._monitor_thread.start()
                
                if self.verbose:
                    print(f"🎮 Interactive mode enabled:")
                    print(f"   Press '{self.pause_key}' to pause at next validation")
                    print(f"   Press '{self.quit_key}' to quit immediately")
                    print(f"   Press Ctrl+C to pause gracefully\n")
            else:
                self.enable_pause = False
                logger.warning("Keyboard not available. Disabling interactive features.")
        except Exception as e:
            self.enable_pause = False
            logger.warning(f"Could not initialize keyboard monitoring: {e}")
    
    def _keyboard_monitor_loop(self):
        """Monitor keyboard input in background thread."""
        while self._monitoring:
            try:
                key = self._keyboard_handler.get_key()
                if key == self.pause_key:
                    if not self.state or not self.state.pause_requested:
                        print(f"\n⏸️  Pause key '{self.pause_key}' pressed. Pausing at next validation...")
                        self._request_pause()
                elif key == self.quit_key:
                    print(f"\n⛔ Quit key '{self.quit_key}' pressed. Saving and exiting...")
                    self._force_quit()
                
                time.sleep(0.1)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                logger.debug(f"Keyboard monitor error: {e}")
    
    def _request_pause(self):
        """Request pause for all trials."""
        if self.state:
            self.state.pause_requested = True
        
        # Write pause signal file for trials to detect
        try:
            self.pause_signal_file.parent.mkdir(parents=True, exist_ok=True)
            pause_data = {
                "pause_requested": True,
                "timestamp": time.time(),
                "experiment_name": self.experiment_name,
                "session_id": self.session_id,
            }
            self.pause_signal_file.write_text(json.dumps(pause_data))
            logger.info(f"Pause signal written to {self.pause_signal_file}")
        except Exception as e:
            logger.error(f"Failed to write pause signal: {e}")
    
    def _force_quit(self):
        """Force quit with checkpoint saving."""
        # Ray Tune handles SIGINT gracefully, so we can just exit
        sys.exit(0)
    
    def restore_optimizer(self, state: TuneSessionState) -> Any:
        """
        Restore optimizer from saved state.
        
        This recreates the EXACT optimizer configuration!
        """
        import pickle
        try:
            from ..core.optimizer import ConfigDrivenOptimizer
        except ImportError:
            from core.optimizer import ConfigDrivenOptimizer
        
        # Unpickle components
        search_space = pickle.loads(state.search_space_pickle)
        strategy = pickle.loads(state.strategy_pickle)
        optimization_config = pickle.loads(state.optimization_config_pickle)
        
        # Recreate optimizer with exact same configuration
        optimizer = ConfigDrivenOptimizer(
            base_config_path=state.base_config_path,
            search_space=search_space,
            strategy=strategy,
            optimization_config=optimization_config,
        )
        
        if self.verbose:
            print(f"✅ Optimizer restored from session state")
            print(f"   Search space: {search_space.__class__.__name__}")
            print(f"   Strategy: {strategy.get_strategy_name()}")
            print(f"   Base config: {state.base_config_path}")
        
        return optimizer
    
    def run(
        self,
        optimizer: Optional[Any] = None,
        resume: bool = False,
        **kwargs
    ) -> tune.ResultGrid:
        """
        Run optimization with proper state management.
        
        Args:
            optimizer: Optimizer instance (not needed for resume!)
            resume: Whether to resume from checkpoint
            **kwargs: Additional arguments
        """
        # Handle resume case
        if resume or (optimizer is None and self.state_file.exists()):
            if self.state_file.exists():
                # Load saved state
                self.state = TuneSessionState.load(self.state_file)
                
                # Restore optimizer from state
                self._optimizer = self.restore_optimizer(self.state)
                
                print(f"\n♻️  Resuming session: {self.state.session_id}")
                print(f"📊 Experiment: {self.state.experiment_name}")
                print(f"⏰ Originally started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.state.created_at))}")
                
                # Actually resume the Ray Tune experiment
                kwargs['resume'] = True
                
            else:
                raise FileNotFoundError(
                    f"Cannot resume: state file not found at {self.state_file}\n"
                    f"Make sure you're using the correct session directory."
                )
        
        # Handle new run case
        elif optimizer is not None:
            # Save optimizer configuration
            self.state = self.save_optimizer_config(optimizer)
            self._optimizer = optimizer
        
        else:
            raise ValueError(
                "Either provide an optimizer for new run, "
                "or ensure state file exists for resume"
            )
        
        # Start keyboard monitoring
        self._start_keyboard_monitoring()
        
        # Inject pause callback into optimizer
        if hasattr(self._optimizer, 'additional_callbacks'):
            try:
                from ..callbacks.tune_pause_callback import TunePauseCallback
            except ImportError:
                from callbacks.tune_pause_callback import TunePauseCallback
            pause_callback = TunePauseCallback(
                pause_signal_file=self.pause_signal_file,
                cli_instance=self,
            )
            self._optimizer.additional_callbacks.append(pause_callback)
        
        # Now run with the optimizer (either new or restored)
        try:
            results = self._optimizer.run(**kwargs)
            return results
            
        except KeyboardInterrupt:
            self._handle_pause()
            raise
        
        finally:
            # Cleanup
            self._cleanup()
    
    def _handle_pause(self):
        """Handle pause and print proper resume instructions."""
        if self.state:
            self.state.pause_requested = True
            self.state.paused_at = time.time()
            self.state.save(self.state_file)
        
        print("\n" + "="*70)
        print("✅ OPTIMIZATION PAUSED")
        print("="*70)
        print(f"\n📊 Session: {self.session_id}")
        print(f"💾 State saved to: {self.state_file}")
        
        # Generate resume command
        resume_command = self._generate_resume_command()
        print("\n📝 To resume this EXACT session with all settings, run:")
        print(f"\n   {resume_command}\n")
        
        print("ℹ️  This will restore:")
        print("   • The exact search space")
        print("   • The strategy configuration")
        print("   • The optimization progress")
        print("   • All hyperparameters and settings")
        print("="*70)
    
    def _generate_resume_command(self) -> str:
        """Generate accurate resume command."""
        # The resume command just needs the session directory!
        script = sys.argv[0]
        return f"python {script} --resume-session {self.session_dir}"
    
    def _cleanup(self):
        """Clean up resources."""
        # Stop keyboard monitoring
        if self._monitoring:
            self._monitoring = False
            if self._keyboard_handler:
                self._keyboard_handler.stop_monitoring()
        
        # Remove pause signal file
        if self.pause_signal_file.exists():
            try:
                self.pause_signal_file.unlink()
            except Exception as e:
                logger.debug(f"Could not remove pause signal file: {e}")
    
    @classmethod
    def resume(cls, session_dir: Union[str, Path]) -> 'TuneReflowCLI':
        """
        Resume from a saved session.
        
        Example:
            ```python
            # Resume from checkpoint
            cli = TuneReflowCLI.resume(session_dir="./experiments/my_opt/session_xxx")
            results = cli.run()  # No optimizer needed!
            ```
        """
        session_dir = Path(session_dir)
        state_file = session_dir / "session_state.pkl"
        
        if not state_file.exists():
            raise FileNotFoundError(f"Session state not found: {state_file}")
        
        # Load state
        state = TuneSessionState.load(state_file)
        
        # Create CLI with loaded state
        cli = cls(
            experiment_name=state.experiment_name,
            experiment_dir=state.experiment_dir,
            session_id=state.session_id,
        )
        cli.state = state
        cli.state_file = state_file
        cli.session_dir = session_dir
        
        return cli


def main():
    """Example with proper resume capability."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--base-config", type=str)
    parser.add_argument("--resume-session", type=str, help="Resume from session directory")
    
    args = parser.parse_args()
    
    if args.resume_session:
        # RESUME case - no need for any configuration!
        cli = TuneReflowCLI.resume(session_dir=args.resume_session)
        results = cli.run()  # Everything restored from checkpoint!
        
    else:
        # NEW run - need full configuration
        try:
            from ..core.optimizer import ConfigDrivenOptimizer
            from ..core.strategies import BOHBStrategy
        except ImportError:
            from core.optimizer import ConfigDrivenOptimizer
            from core.strategies import BOHBStrategy
        
        # Create optimizer (this defines everything)
        strategy = BOHBStrategy(grace_period=10)
        optimizer = ConfigDrivenOptimizer(
            base_config_path=args.base_config,
            search_space=create_search_space(),  # Your search space
            strategy=strategy,
        )
        
        # Create CLI and run
        cli = TuneReflowCLI(experiment_name=args.experiment_name)
        results = cli.run(optimizer)  # Everything saved automatically!
    
    print(f"Best result: {results.get_best_result().config}")


if __name__ == "__main__":
    main()