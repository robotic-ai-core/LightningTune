"""
Shared persistence utilities for saving/loading Optuna study sessions and
building resume commands. Used by both HPORunner and PausibleOptunaOptimizer.

This module provides:
1. StudyPersistence protocol - interface for custom storage backends
2. LocalPersistence - local filesystem storage
3. WandBPersistence - WandB artifact storage
4. Legacy functions - for backward compatibility
"""

from __future__ import annotations

import os
import tempfile
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Protocol, runtime_checkable
import pickle

logger = logging.getLogger(__name__)


# =============================================================================
# Persistence Protocol and Base Classes
# =============================================================================

@dataclass
class StudyMetadata:
    """Metadata associated with a saved study.

    Attributes:
        study_name: Name of the Optuna study.
        total_trials_completed: Number of finished (COMPLETE + PRUNED) trials.
        sampler_name: Name of the Optuna sampler used.
        pruner_name: Name of the Optuna pruner used.
        config_overrides: Persistent configuration overrides.
        last_wandb_upload_trial_count: Trial count at last WandB upload.
    """
    study_name: str
    total_trials_completed: int
    sampler_name: str
    pruner_name: str
    config_overrides: Dict[str, Any]
    last_wandb_upload_trial_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "study_name": self.study_name,
            "total_trials_completed": self.total_trials_completed,
            "sampler_name": self.sampler_name,
            "pruner_name": self.pruner_name,
            "config_overrides": self.config_overrides,
            "last_wandb_upload_trial_count": self.last_wandb_upload_trial_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StudyMetadata":
        """Create from dictionary."""
        return cls(
            study_name=data.get("study_name", "unknown"),
            total_trials_completed=data.get("total_trials_completed", 0),
            sampler_name=data.get("sampler_name", "tpe"),
            pruner_name=data.get("pruner_name", "median"),
            config_overrides=data.get("config_overrides", {}),
            last_wandb_upload_trial_count=data.get("last_wandb_upload_trial_count"),
        )


@runtime_checkable
class StudyPersistence(Protocol):
    """Protocol for HPO study persistence backends.

    This protocol defines the interface that all persistence backends must implement.
    It enables custom storage backends (S3, GCS, databases) while maintaining
    a consistent interface.

    Example:
        >>> class S3Persistence:
        ...     def save_study(self, study, metadata: StudyMetadata) -> bool:
        ...         # Upload to S3
        ...         return True
        ...     def load_study(self, identifier: str) -> Optional[Dict[str, Any]]:
        ...         # Download from S3
        ...         return session_info
        ...     def list_checkpoints(self) -> List[str]:
        ...         return ["checkpoint_1", "checkpoint_2"]
    """

    def save_study(self, study, metadata: StudyMetadata) -> bool:
        """Save an Optuna study with metadata.

        Args:
            study: Optuna study object to save.
            metadata: Associated metadata.

        Returns:
            True if save succeeded, False otherwise.
        """
        ...

    def load_study(self, identifier: str = "latest") -> Optional[Dict[str, Any]]:
        """Load a saved study session.

        Args:
            identifier: Checkpoint identifier (e.g., "latest", version number, path).

        Returns:
            Session info dict with 'study', 'total_trials_completed', etc.,
            or None if not found.
        """
        ...

    def list_checkpoints(self) -> List[str]:
        """List available checkpoints.

        Returns:
            List of checkpoint identifiers.
        """
        ...


class BasePersistence(ABC):
    """Abstract base class for persistence implementations.

    Provides common functionality and enforces the interface.
    """

    @abstractmethod
    def save_study(self, study, metadata: StudyMetadata) -> bool:
        """Save study with metadata."""
        pass

    @abstractmethod
    def load_study(self, identifier: str = "latest") -> Optional[Dict[str, Any]]:
        """Load study by identifier."""
        pass

    @abstractmethod
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        pass

    def _build_session_info(self, study, metadata: StudyMetadata) -> Dict[str, Any]:
        """Build the session info dictionary."""
        return {
            "study": study,
            "total_trials_completed": metadata.total_trials_completed,
            "sampler_name": metadata.sampler_name,
            "pruner_name": metadata.pruner_name,
            "study_name": metadata.study_name,
            "config_overrides": metadata.config_overrides,
            "last_wandb_upload_trial_count": metadata.last_wandb_upload_trial_count,
        }


class LocalPersistence(BasePersistence):
    """Local filesystem persistence backend.

    Saves studies to a local directory as pickle files with YAML metadata.

    Example:
        >>> persistence = LocalPersistence(Path("checkpoints/my_study"))
        >>> metadata = StudyMetadata(
        ...     study_name="my_study",
        ...     total_trials_completed=10,
        ...     sampler_name="tpe",
        ...     pruner_name="median",
        ...     config_overrides={},
        ... )
        >>> persistence.save_study(study, metadata)
        True
    """

    def __init__(self, checkpoint_dir: Path):
        """Initialize local persistence.

        Args:
            checkpoint_dir: Directory for storing checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir)

    def save_study(self, study, metadata: StudyMetadata) -> bool:
        """Save study to local filesystem."""
        import yaml

        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            local_path = self.checkpoint_dir / "study.pkl"

            session_info = self._build_session_info(study, metadata)
            with open(local_path, 'wb') as f:
                pickle.dump(session_info, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"💾 Saved local study checkpoint: {local_path}")

            # Also save minimal YAML metadata for quick inspection
            session_args = {
                "n_trials": metadata.config_overrides.get("args.n_trials"),
                "save_every": metadata.config_overrides.get("args.save_every"),
                "isolate_trials": metadata.config_overrides.get("args.isolate_trials", True),
                "sampler_name": metadata.sampler_name,
                "pruner_name": metadata.pruner_name,
                "study_name": metadata.study_name,
                "total_trials_completed": metadata.total_trials_completed,
            }
            with open(self.checkpoint_dir / "session_args.yaml", 'w') as f:
                yaml.dump(session_args, f, default_flow_style=False)

            return True
        except Exception as e:
            logger.error(f"Failed to save local study: {e}")
            return False

    def load_study(self, identifier: str = "latest") -> Optional[Dict[str, Any]]:
        """Load study from local filesystem.

        Args:
            identifier: Path to file/directory or "latest" for default location.
        """
        try:
            if identifier == "latest":
                candidate = self.checkpoint_dir / "study.pkl"
            elif os.path.exists(identifier):
                p = Path(identifier)
                candidate = p if p.is_file() else (p / "study.pkl")
            else:
                candidate = self.checkpoint_dir / "study.pkl"

            if not candidate.exists():
                return None

            with open(candidate, 'rb') as f:
                session_info = pickle.load(f)
            logger.info(f"✅ Loaded local study: {candidate}")
            return session_info
        except Exception as e:
            logger.error(f"Failed to load local study: {e}")
            return None

    def list_checkpoints(self) -> List[str]:
        """List checkpoint files in the directory."""
        if not self.checkpoint_dir.exists():
            return []
        return [str(p) for p in self.checkpoint_dir.glob("*.pkl")]


class WandBPersistence(BasePersistence):
    """WandB artifact persistence backend.

    Saves studies as WandB artifacts with versioning and 'latest' alias.

    Example:
        >>> persistence = WandBPersistence("my-project", "my_study")
        >>> persistence.save_study(study, metadata)
        True
        >>> session = persistence.load_study("latest")
    """

    def __init__(self, project: str, study_name: str):
        """Initialize WandB persistence.

        Args:
            project: WandB project name.
            study_name: Study name (used for artifact naming).
        """
        self.project = project
        self.study_name = study_name

    def save_study(self, study, metadata: StudyMetadata) -> bool:
        """Save study to WandB as an artifact."""
        import optuna
        import wandb

        # Verify finished trials integrity
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
        waiting = [t for t in study.trials if t.state == optuna.trial.TrialState.WAITING]
        finished_count = len(completed) + len(pruned)

        if running or waiting:
            logger.warning(
                f"⚠️  Cannot save study: {len(running)} running, {len(waiting)} waiting trials"
            )
            return False

        trials_completed = metadata.total_trials_completed
        if finished_count != metadata.total_trials_completed:
            logger.warning(
                f"⚠️  Expected {metadata.total_trials_completed} finished trials "
                f"but found {finished_count}. Saving with actual count."
            )
            trials_completed = finished_count

        logger.info(
            f"💾 Saving study: {len(completed)} completed, {len(pruned)} pruned, "
            f"{len(study.trials) - finished_count} failed"
        )

        # Update metadata with actual count
        updated_metadata = StudyMetadata(
            study_name=metadata.study_name,
            total_trials_completed=trials_completed,
            sampler_name=metadata.sampler_name,
            pruner_name=metadata.pruner_name,
            config_overrides=metadata.config_overrides,
            last_wandb_upload_trial_count=metadata.last_wandb_upload_trial_count,
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            session_info = self._build_session_info(study, updated_metadata)
            pickle.dump(session_info, tmp, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.flush()
            os.fsync(tmp.fileno())

            # Verify
            tmp.seek(0)
            try:
                _ = pickle.load(tmp)
            except Exception as e:
                logger.error(f"Failed to verify saved study: {e}")
                return False

        run = None
        try:
            logger.info(f"🌐 Uploading checkpoint to WandB project '{self.project}'...")
            run = wandb.init(project=self.project, job_type="hpo_checkpoint")
            artifact = wandb.Artifact(f"{self.study_name}_checkpoint", type="optuna_study")
            artifact.add_file(tmp.name, name="study.pkl")
            artifact.metadata = {"total_finished_trials": trials_completed}
            logged_artifact = run.log_artifact(artifact, aliases=["latest"])
            logged_artifact.wait()
            run.finish()
            logger.info(f"✅ Study saved to WandB: {self.study_name}_checkpoint (v{trials_completed})")
            return True
        except Exception as e:
            logger.error(f"❌ WandB upload failed: {e}")
            if run is not None:
                try:
                    run.finish()
                except Exception:
                    pass
            return False

    def load_study(self, identifier: str = "latest") -> Optional[Dict[str, Any]]:
        """Load study from WandB artifact."""
        import wandb

        # First try: Api path (no run created)
        try:
            api = wandb.Api()
            artifact_path = f"{self.project}/{self.study_name}_checkpoint:{identifier}"
            artifact = api.artifact(artifact_path)
            with tempfile.TemporaryDirectory() as tmpdir:
                downloaded_path = artifact.download(tmpdir)
                candidate = os.path.join(downloaded_path, "study.pkl")
                if not os.path.exists(candidate):
                    for dirpath, _, filenames in os.walk(downloaded_path):
                        for fname in filenames:
                            if fname.endswith('.pkl'):
                                candidate = os.path.join(dirpath, fname)
                                break
                        if os.path.exists(candidate):
                            break
                if not os.path.exists(candidate):
                    logger.error("study.pkl not found in artifact (Api)")
                    return None
                with open(candidate, 'rb') as f:
                    session_info = pickle.load(f)
            logger.info(
                f"✅ Loaded study with {session_info.get('total_trials_completed', 0)} finished trials"
            )
            return session_info
        except wandb.errors.CommError as e:
            logger.warning(f"❌ No WandB artifact found via Api: {e}")
            return None
        except Exception as e:
            logger.debug(f"wandb.Api() failed, trying run.use_artifact: {e}")

        # Fallback: use_artifact
        try:
            run = wandb.init(project=self.project, job_type="hpo_resume")
            artifact_name = f"{self.study_name}_checkpoint:{identifier}"
            artifact = run.use_artifact(artifact_name)
            with tempfile.TemporaryDirectory() as tmpdir:
                downloaded_path = artifact.download(root=tmpdir)
                expected_path = os.path.join(downloaded_path, "study.pkl")
                file_path = expected_path if os.path.exists(expected_path) else None
                if not file_path:
                    for dirpath, _, filenames in os.walk(downloaded_path):
                        for fname in filenames:
                            if fname.endswith(".pkl"):
                                file_path = os.path.join(dirpath, fname)
                                break
                        if file_path:
                            break
                if not file_path:
                    logger.error("study.pkl not found in artifact")
                    run.finish()
                    return None
                with open(file_path, 'rb') as f:
                    session_info = pickle.load(f)
            run.finish()
            logger.info(
                f"✅ Loaded study with {session_info.get('total_trials_completed', 0)} finished trials"
            )
            return session_info
        except wandb.errors.CommError as e:
            logger.warning(f"❌ No WandB artifact found: {e}")
            return None
        except Exception as e:
            logger.warning(f"❌ Unexpected WandB error: {e}")
            return None

    def list_checkpoints(self) -> List[str]:
        """List WandB artifact versions."""
        try:
            import wandb
            api = wandb.Api()
            artifact_type = f"{self.project}/{self.study_name}_checkpoint"
            versions = api.artifact_versions(type_name="optuna_study", name=artifact_type)
            return [v.version for v in versions]
        except Exception as e:
            logger.debug(f"Failed to list WandB checkpoints: {e}")
            return []


class CompositePersistence(BasePersistence):
    """Persistence backend that saves to multiple backends.

    Useful for saving to both local and WandB simultaneously.

    Example:
        >>> local = LocalPersistence(Path("checkpoints"))
        >>> wandb = WandBPersistence("my-project", "my_study")
        >>> composite = CompositePersistence([local, wandb])
        >>> composite.save_study(study, metadata)  # Saves to both
    """

    def __init__(self, backends: List[BasePersistence], primary_index: int = 0):
        """Initialize composite persistence.

        Args:
            backends: List of persistence backends.
            primary_index: Index of primary backend for load operations.
        """
        self.backends = backends
        self.primary_index = primary_index

    def save_study(self, study, metadata: StudyMetadata) -> bool:
        """Save to all backends, return True if primary succeeds."""
        results = []
        for backend in self.backends:
            try:
                results.append(backend.save_study(study, metadata))
            except Exception as e:
                logger.warning(f"Backend {type(backend).__name__} failed: {e}")
                results.append(False)

        # Return success of primary backend
        return results[self.primary_index] if results else False

    def load_study(self, identifier: str = "latest") -> Optional[Dict[str, Any]]:
        """Load from primary backend, fall back to others if needed."""
        for i, backend in enumerate(self.backends):
            # Start with primary, then try others
            idx = (self.primary_index + i) % len(self.backends)
            try:
                result = self.backends[idx].load_study(identifier)
                if result is not None:
                    return result
            except Exception as e:
                logger.debug(f"Backend {type(self.backends[idx]).__name__} load failed: {e}")
        return None

    def list_checkpoints(self) -> List[str]:
        """List checkpoints from primary backend."""
        return self.backends[self.primary_index].list_checkpoints()


# =============================================================================
# Legacy Functions (for backward compatibility)
# =============================================================================


def save_study_to_local(
    local_checkpoint_dir: Path,
    study,
    total_trials_completed: int,
    *,
    sampler_name: str,
    pruner_name: str,
    study_name: str,
    config_overrides: Optional[Dict[str, Any]] = None,
    last_wandb_upload_trial_count: Optional[int] = None,
) -> bool:
    """Save study session to local filesystem.

    Writes study.pkl and a small session_args.yaml for quick inspection.

    Args:
        last_wandb_upload_trial_count: Trial count at last WandB upload. Used to track
            how many trials since last upload when restart_every_trial=True.
    """
    import yaml

    try:
        local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Always write study.pkl for simplicity and predictable resumes
        local_path = local_checkpoint_dir / "study.pkl"

        session_info = {
            "study": study,
            "total_trials_completed": total_trials_completed,
            "sampler_name": sampler_name,
            "pruner_name": pruner_name,
            "study_name": study_name,
            "config_overrides": config_overrides or {},
            "last_wandb_upload_trial_count": last_wandb_upload_trial_count,
        }
        with open(local_path, 'wb') as f:
            pickle.dump(session_info, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"💾 Saved local study checkpoint: {local_path}")

        # Also save minimal YAML metadata
        # Derive n_trials and isolate_trials with sensible defaults for resume UX
        saved_args = config_overrides or {}
        session_args = {
            "config": saved_args.get("args.config", None),
            "trial_steps": saved_args.get("args.trial_steps", None),
            "n_trials": saved_args.get("args.n_trials", None),
            "save_every": saved_args.get("args.save_every", None),
            "isolate_trials": saved_args.get("args.isolate_trials", True),
            "sampler_name": sampler_name,
            "pruner_name": pruner_name,
            "study_name": study_name,
            "total_trials_completed": total_trials_completed,
        }
        with open(local_checkpoint_dir / "session_args.yaml", 'w') as f:
            yaml.dump(session_args, f, default_flow_style=False)

        return True
    except Exception as e:
        logger.error(f"Failed to save local study: {e}")
        return False


def load_study_from_local(path_or_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load study session from a local file or directory.

    Accepts either a direct path to a pkl, or a directory containing study.pkl.
    """
    try:
        candidate: Optional[Path] = None
        if path_or_dir and os.path.exists(path_or_dir):
            p = Path(path_or_dir)
            candidate = p if p.is_file() else (p / "study.pkl")
        if not candidate or not candidate.exists():
            return None
        with open(candidate, 'rb') as f:
            session_info = pickle.load(f)
        logger.info(f"✅ Loaded local study: {candidate}")
        return session_info
    except Exception as e:
        logger.error(f"Failed to load local study: {e}")
        return None


def save_study_to_wandb(
    wandb_project: Optional[str],
    *,
    study_name: str,
    study,
    total_trials_completed: int,
    sampler_name: str,
    pruner_name: str,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> bool:
    """Save study session to WandB artifact (with alias 'latest')."""
    if not wandb_project:
        logger.debug("WandB project not configured, skipping save")
        return False

    import optuna
    import wandb

    # Verify finished trials integrity
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    waiting = [t for t in study.trials if t.state == optuna.trial.TrialState.WAITING]
    finished_count = len(completed) + len(pruned)

    if running or waiting:
        logger.warning(
            f"⚠️  Cannot save study: {len(running)} running, {len(waiting)} waiting trials"
        )
        return False

    trials_completed = total_trials_completed
    if finished_count != total_trials_completed:
        logger.warning(
            f"⚠️  Expected {total_trials_completed} finished trials but found {finished_count}. Saving with actual count."
        )
        trials_completed = finished_count

    logger.info(
        f"💾 Saving study: {len(completed)} completed, {len(pruned)} pruned, {len(study.trials) - finished_count} failed"
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        session_info = {
            "study": study,
            "total_trials_completed": trials_completed,
            "sampler_name": sampler_name,
            "pruner_name": pruner_name,
            "study_name": study_name,
            "config_overrides": config_overrides or {},
        }
        pickle.dump(session_info, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.flush()
        os.fsync(tmp.fileno())

        # Simple verification
        tmp.seek(0)
        try:
            _ = pickle.load(tmp)
        except Exception as e:
            logger.error(f"Failed to verify saved study: {e}")
            return False

    run = None
    try:
        logger.info(f"🌐 Uploading checkpoint to WandB project '{wandb_project}'...")
        run = wandb.init(project=wandb_project, job_type="hpo_checkpoint")
        artifact = wandb.Artifact(f"{study_name}_checkpoint", type="optuna_study")
        # Backward compatibility: some consumers expect different internal names
        # Always add as study.pkl, but also add a duplicate with legacy name if needed
        artifact.add_file(tmp.name, name="study.pkl")
        artifact.metadata = {
            "total_finished_trials": trials_completed,
        }
        logged_artifact = run.log_artifact(artifact, aliases=["latest"])
        logged_artifact.wait()
        run.finish()
        logger.info(f"✅ Study saved to WandB: {study_name}_checkpoint (v{trials_completed})")
        return True
    except Exception as e:
        logger.error(f"❌ WandB upload failed: {e}")
        logger.error(f"   Check your WandB API key and network connection")
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass
        return False


def load_study_from_wandb(
    wandb_project: Optional[str],
    study_name: Optional[str],
    version: str = "latest",
) -> Optional[Dict[str, Any]]:
    """Load a study session from WandB artifact without creating a run when possible.

    Strategy:
    1) Prefer no-run client: wandb.Api().artifact(...).download(...)
    2) Fallback to run.use_artifact(...) if Api path fails

    Also scans for any *.pkl if study.pkl isn't at the root for backward compatibility.
    """
    if not wandb_project or not study_name:
        return None
    import wandb

    # First try: Api path (no run created)
    try:
        api = wandb.Api()
        artifact_path = f"{wandb_project}/{study_name}_checkpoint:{version}"
        artifact = api.artifact(artifact_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded_path = artifact.download(tmpdir)
            candidate = os.path.join(downloaded_path, "study.pkl")
            if not os.path.exists(candidate):
                # scan for any pkl
                for dirpath, _dirnames, filenames in os.walk(downloaded_path):
                    for fname in filenames:
                        if fname.endswith('.pkl'):
                            candidate = os.path.join(dirpath, fname)
                            break
                    if os.path.exists(candidate):
                        break
            if not os.path.exists(candidate):
                logger.error("study.pkl not found in artifact (Api)")
                return None
            with open(candidate, 'rb') as f:
                session_info = pickle.load(f)
        logger.info(
            f"✅ Loaded study with {session_info.get('total_trials_completed', 0)} finished trials"
        )
        return session_info
    except wandb.errors.CommError as e_api:
        logger.warning(f"❌ No WandB artifact found via Api: {e_api}")
        return None
    except Exception as e_api:
        logger.debug(f"wandb.Api() artifact load failed, falling back to run.use_artifact: {e_api}")

    # Fallback: use_artifact (creates a short-lived run)
    try:
        run = wandb.init(project=wandb_project, job_type="hpo_resume")
        artifact_name = f"{study_name}_checkpoint:{version}"
        artifact = run.use_artifact(artifact_name)
        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded_path = artifact.download(root=tmpdir)
            expected_path = os.path.join(downloaded_path, "study.pkl")
            file_path: Optional[str] = expected_path if os.path.exists(expected_path) else None
            if not file_path:
                found: Optional[str] = None
                for dirpath, _dirnames, filenames in os.walk(downloaded_path):
                    for fname in filenames:
                        if fname.endswith(".pkl"):
                            found = os.path.join(dirpath, fname)
                            break
                    if found:
                        break
                file_path = found
            if not file_path:
                logger.error("study.pkl not found in artifact and no .pkl fallback available")
                run.finish()
                return None
            with open(file_path, 'rb') as f:
                session_info = pickle.load(f)
        run.finish()
        logger.info(
            f"✅ Loaded study with {session_info.get('total_trials_completed', 0)} finished trials"
        )
        return session_info
    except wandb.errors.CommError as e:
        logger.warning(f"❌ No WandB artifact found via use_artifact: {e}")
        return None
    except Exception as e:
        logger.warning(f"❌ Unexpected WandB error: {e}")
        return None


def load_saved_session(
    resume_from: str,
    *,
    wandb_project: Optional[str] = None,
    study_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Attempt to load a saved session from local path, else WandB."""
    if os.path.exists(resume_from):
        logger.info(f"📁 Loading from local file/dir: {resume_from}")
        try:
            return load_study_from_local(resume_from)
        except Exception as e:
            logger.warning(f"Failed to load from {resume_from}: {e}")
            return None
    if wandb_project and study_name:
        logger.info("☁️  Attempting to load from WandB...")
        return load_study_from_wandb(wandb_project, study_name, version=resume_from)
    logger.warning(f"Could not load session from {resume_from}")
    return None


def parse_cli_arg(argv: List[str], name: str) -> Optional[str]:
    """Parse a CLI argument value from argv.

    Handles both '--name value' and '--name=value' formats.

    Args:
        argv: List of command line arguments
        name: Name of the argument (without leading --)

    Returns:
        The argument value if found, None otherwise
    """
    flag = f"--{name}"
    for i, tok in enumerate(argv or []):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


# Alias for backward compatibility
_parse_arg = parse_cli_arg


def build_resume_command(original_argv: List[str], default_script: str, *, fallback_wandb: Optional[str] = None, fallback_study: Optional[str] = None) -> str:
    """Construct a minimal WandB resume command preserving trial-steps if present."""
    script = original_argv[0] if (original_argv and original_argv[0]) else default_script
    parts: List[str] = ["python", script]
    wandb_proj = _parse_arg(original_argv, "wandb") or fallback_wandb
    study_name = _parse_arg(original_argv, "study-name") or fallback_study
    if wandb_proj:
        parts += ["--wandb", wandb_proj]
    if study_name:
        parts += ["--study-name", study_name]
    ts = _parse_arg(original_argv, "trial-steps")
    if ts:
        parts += ["--trial-steps", str(ts)]
    parts += ["--resume-from", "latest"]
    return " ".join(parts)


def build_local_resume_command(original_argv: List[str], default_script: str, local_path: str) -> str:
    script = original_argv[0] if (original_argv and original_argv[0]) else default_script
    ts = _parse_arg(original_argv, "trial-steps")
    ts_arg = f" --trial-steps {ts}" if ts else ""
    return f"python {script}{ts_arg} --resume-from {local_path}"


