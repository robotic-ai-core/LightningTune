#!/usr/bin/env python
"""
Tests for local checkpoint mirroring and storage-backed resume in PausibleOptunaOptimizer.
"""

import sys
from pathlib import Path
from unittest.mock import patch, Mock
import tempfile
import pickle

import optuna

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external" / "LightningTune"))

from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer


def _simple_objective(trial: optuna.Trial) -> float:
    # Pure optuna; no Lightning dependencies
    return trial.suggest_float('x', 0.0, 1.0)


def _patch_underlying_optimizer():
    """Patch Optimizer classes inside pausible_optimizer to return a simple objective.

    Returns a context manager that patches both OptunaDrivenOptimizer and
    ReflowOptunaDrivenOptimizer in the module under test.
    """
    mock_instance = Mock()
    mock_instance.create_objective.return_value = _simple_objective

    patch_opt = patch(
        'LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer',
        autospec=True
    )
    patch_opt_reflow = patch(
        'LightningTune.optuna.pausible_optimizer.ReflowOptunaDrivenOptimizer',
        autospec=True
    )

    cm_opt = patch_opt.start()
    cm_opt.return_value = mock_instance

    cm_opt_rf = patch_opt_reflow.start()
    cm_opt_rf.return_value = mock_instance

    return mock_instance, patch_opt, patch_opt_reflow


def _stop_patches(*patchers):
    for p in patchers:
        try:
            p.stop()
        except Exception:
            pass


def test_local_checkpoint_and_resume_from_dir(tmp_path: Path):
    # Arrange
    local_dir = tmp_path / "local_ckpt"
    mock_instance, p1, p2 = _patch_underlying_optimizer()
    try:
        optimizer = PausibleOptunaOptimizer(
            base_config={'dummy': 'config'},
            search_space=lambda trial: {},
            model_class=Mock,  # unused due to patch
            datamodule_class=Mock,  # unused due to patch
            wandb_project=None,
            study_name="test_local_resume_dir",
            save_every_n_trials=10,
            enable_pause=False,
            use_reflow=False,
            local_checkpoint_dir=str(local_dir),
        )

        # Act: run 3 trials and ensure local study.pkl exists with correct count
        study = optimizer.optimize(
            n_trials=3,
            config_overrides={},
            callbacks=[],
        )

        assert (local_dir / "study.pkl").exists(), "Local study.pkl not saved"
        with open(local_dir / "study.pkl", 'rb') as f:
            session = pickle.load(f)
        assert session["total_trials_completed"] == 3
        # Verify study object in session aligns
        finished = len([t for t in session["study"].trials
                        if t.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]])
        assert finished == 3

        # Act: resume from local directory to reach total of 5
        optimizer2 = PausibleOptunaOptimizer(
            base_config={'dummy': 'config'},
            search_space=lambda trial: {},
            model_class=Mock,
            datamodule_class=Mock,
            wandb_project=None,
            study_name="test_local_resume_dir",
            save_every_n_trials=10,
            enable_pause=False,
            use_reflow=False,
            local_checkpoint_dir=str(local_dir),
        )
        study2 = optimizer2.optimize(
            n_trials=5,
            resume_from=str(local_dir),
            config_overrides={},
            callbacks=[],
        )

        # Assert: 5 finished trials in the resumed study
        finished2 = len([t for t in study2.trials
                         if t.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]])
        assert finished2 == 5
    finally:
        _stop_patches(p1, p2)


def test_local_resume_from_file(tmp_path: Path):
    local_dir = tmp_path / "local_ckpt_file"
    mock_instance, p1, p2 = _patch_underlying_optimizer()
    try:
        opt1 = PausibleOptunaOptimizer(
            base_config={'dummy': 'config'},
            search_space=lambda trial: {},
            model_class=Mock,
            datamodule_class=Mock,
            wandb_project=None,
            study_name="test_local_resume_file",
            save_every_n_trials=10,
            enable_pause=False,
            use_reflow=False,
            local_checkpoint_dir=str(local_dir),
        )
        _ = opt1.optimize(n_trials=2, config_overrides={}, callbacks=[])

        ckpt_file = local_dir / "study.pkl"
        assert ckpt_file.exists()

        opt2 = PausibleOptunaOptimizer(
            base_config={'dummy': 'config'},
            search_space=lambda trial: {},
            model_class=Mock,
            datamodule_class=Mock,
            wandb_project=None,
            study_name="test_local_resume_file",
            save_every_n_trials=10,
            enable_pause=False,
            use_reflow=False,
            local_checkpoint_dir=str(local_dir),
        )
        study2 = opt2.optimize(
            n_trials=4,
            resume_from=str(ckpt_file),
            config_overrides={},
            callbacks=[],
        )

        finished2 = len([t for t in study2.trials
                         if t.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]])
        assert finished2 == 4
    finally:
        _stop_patches(p1, p2)


def test_storage_resume_sqlite(tmp_path: Path):
    # Arrange SQLite storage URL
    db_path = tmp_path / "study.db"
    storage_url = f"sqlite:///{db_path}"
    mock_instance, p1, p2 = _patch_underlying_optimizer()
    try:
        study_name = "test_storage_sqlite"
        # First session: run 2 trials and persist to DB
        opt_a = PausibleOptunaOptimizer(
            base_config={'dummy': 'config'},
            search_space=lambda trial: {},
            model_class=Mock,
            datamodule_class=Mock,
            wandb_project=None,
            study_name=study_name,
            save_every_n_trials=10,
            enable_pause=False,
            use_reflow=False,
        )
        study_a = opt_a.optimize(
            n_trials=2,
            storage=storage_url,
            config_overrides={},
            callbacks=[],
        )
        finished_a = len([t for t in study_a.trials
                          if t.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]])
        assert finished_a == 2

        # Second session: resume automatically from DB to reach 5 total
        opt_b = PausibleOptunaOptimizer(
            base_config={'dummy': 'config'},
            search_space=lambda trial: {},
            model_class=Mock,
            datamodule_class=Mock,
            wandb_project=None,
            study_name=study_name,
            save_every_n_trials=10,
            enable_pause=False,
            use_reflow=False,
        )
        study_b = opt_b.optimize(
            n_trials=5,  # absolute total desired
            storage=storage_url,
            config_overrides={},
            callbacks=[],
        )
        finished_b = len([t for t in study_b.trials
                          if t.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]])
        assert finished_b == 5
    finally:
        _stop_patches(p1, p2)


