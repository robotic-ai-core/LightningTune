#!/usr/bin/env python
"""
Unit tests for WandB artifact loading paths and session_args.yaml content.

Covers:
- Prefer wandb.Api (no run) when available
- Return None on Api CommError (no fallback)
- Fallback to run.use_artifact when Api path raises non-CommError
- session_args.yaml contains n_trials/save_every/isolate_trials keys
"""

import sys
import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import optuna
import yaml
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from LightningTune.persistence import (
    load_study_from_wandb,
    save_study_to_local,
)


def _make_study(n: int = 3) -> optuna.Study:
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=n)
    return study


def test_wandb_api_load_success_no_run(monkeypatch):
    """API path should load without creating a run."""
    study = _make_study(5)
    session_info = {
        "study": study,
        "total_trials_completed": 5,
        "sampler_name": "tpe",
        "pruner_name": "median",
        "study_name": "test-study",
    }

    # Mock Api().artifact().download() to write study.pkl
    mock_api = Mock()
    mock_artifact = Mock()

    def _download(tmpdir):
        p = Path(tmpdir) / "study.pkl"
        with open(p, "wb") as f:
            pickle.dump(session_info, f)
        return tmpdir

    mock_artifact.download.side_effect = _download
    mock_api.artifact.return_value = mock_artifact

    with patch.object(wandb, "Api", return_value=mock_api) as _api_patch, \
         patch.object(wandb, "init") as mock_init:
        loaded = load_study_from_wandb("proj", "test-study", version="latest")
        assert loaded is not None
        assert loaded["total_trials_completed"] == 5
        # Ensure no run was created
        mock_init.assert_not_called()


def test_wandb_api_commerror_returns_none(monkeypatch):
    """On Api CommError, return None and do not create a run."""
    mock_api = Mock()
    mock_api.artifact.side_effect = wandb.errors.CommError("not found")

    with patch.object(wandb, "Api", return_value=mock_api), \
         patch.object(wandb, "init") as mock_init:
        loaded = load_study_from_wandb("proj", "test-study", version="latest")
        assert loaded is None
        mock_init.assert_not_called()


def test_wandb_fallback_use_artifact_on_generic_error(monkeypatch):
    """If Api path raises a non-CommError, fallback to run.use_artifact."""
    # Force generic exception from Api path to trigger fallback
    mock_api = Mock()
    mock_api.artifact.side_effect = RuntimeError("generic failure")

    study = _make_study(2)
    session_info = {
        "study": study,
        "total_trials_completed": 2,
        "sampler_name": "tpe",
        "pruner_name": "median",
        "study_name": "test-study",
    }

    # Mock run.use_artifact().download()
    mock_run = Mock()
    mock_artifact = Mock()

    def _dl(root=None):
        # Respect the provided root (loader passes one)
        target = Path(root) if root else Path(tempfile.mkdtemp())
        p = target / "study.pkl"
        with open(p, "wb") as f:
            pickle.dump(session_info, f)
        # Return the directory path so the loader finds study.pkl
        return str(target)

    mock_artifact.download.side_effect = _dl
    mock_run.use_artifact.return_value = mock_artifact

    with patch.object(wandb, "Api", return_value=mock_api), \
         patch.object(wandb, "init", return_value=mock_run) as mock_init:
        loaded = load_study_from_wandb("proj", "test-study", version="latest")
        assert loaded is not None
        assert loaded["total_trials_completed"] == 2
        mock_init.assert_called_once()
        mock_run.finish.assert_called_once()


def test_session_args_yaml_contains_expected_keys(tmp_path: Path):
    """session_args.yaml should include n_trials/save_every/isolate_trials keys."""
    ckpt_dir = tmp_path / "checkpoints" / "proj" / "study"
    study = _make_study(3)
    overrides = {
        "args.n_trials": 10,
        "args.save_every": 2,
        # Leave args.isolate_trials unspecified to check default True
    }
    ok = save_study_to_local(
        ckpt_dir,
        study,
        total_trials_completed=3,
        sampler_name="tpe",
        pruner_name="hyperband",
        study_name="study",
        config_overrides=overrides,
    )
    assert ok
    sess_yaml = ckpt_dir / "session_args.yaml"
    assert sess_yaml.exists()
    with open(sess_yaml, "r") as f:
        sess = yaml.safe_load(f) or {}
    assert "n_trials" in sess and sess["n_trials"] == 10
    assert "save_every" in sess and sess["save_every"] == 2
    assert "isolate_trials" in sess and sess["isolate_trials"] is True


