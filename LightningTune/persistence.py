"""
Shared persistence utilities for saving/loading Optuna study sessions and
building resume commands. Used by both HPORunner and PausibleOptunaOptimizer.
"""

from __future__ import annotations

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import pickle

logger = logging.getLogger(__name__)


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


