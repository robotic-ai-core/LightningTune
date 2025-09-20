import sys
from pathlib import Path
import pickle

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from LightningTune.persistence import (
    save_study_to_local,
    load_study_from_local,
    build_local_resume_command,
)


def _make_study(name: str, n: int) -> optuna.Study:
    study = optuna.create_study(study_name=name)
    for _ in range(n):
        study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)
    return study


def test_local_roundtrip_and_resume_cmd(tmp_path: Path, monkeypatch):
    study_name = "persist_test"
    study = _make_study(study_name, 3)
    ckpt_dir = tmp_path / "checkpoints" / "proj" / study_name

    ok = save_study_to_local(
        ckpt_dir,
        study,
        total_trials_completed=3,
        sampler_name="tpe",
        pruner_name="hyperband",
        study_name=study_name,
        config_overrides={"args.n_trials": 5},
    )
    assert ok
    assert (ckpt_dir / "study.pkl").exists()

    # Load from dir
    session = load_study_from_local(str(ckpt_dir))
    assert session is not None
    assert session["total_trials_completed"] == 3
    # And from file
    session2 = load_study_from_local(str(ckpt_dir / "study.pkl"))
    assert session2 is not None

    # Resume command uses original argv shape
    argv = [
        "scripts/world_model_hpo_optuna.py",
        "--wandb", "proj",
        "--study-name", study_name,
        "--trial-steps", "40000",
    ]
    cmd = build_local_resume_command(argv, "scripts/world_model_hpo_optuna.py", str(ckpt_dir))
    assert cmd.startswith("python ")
    assert "--resume-from" in cmd
    assert str(ckpt_dir) in cmd

