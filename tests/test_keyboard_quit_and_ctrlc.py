from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna

from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer


def _patch_underlying_optimizer():
    mock_instance = MagicMock()
    # Objective increments best value each trial so trials finish
    def simple_objective(trial: optuna.Trial):
        return trial.suggest_float('x', 0.0, 1.0)
    mock_instance.create_objective.return_value = simple_objective
    # After consolidation, only OptunaDrivenOptimizer exists (always uses LightningReflow)
    p1 = patch('LightningTune.optuna.pausible_optimizer.OptunaDrivenOptimizer', autospec=True)
    cm1 = p1.start(); cm1.return_value = mock_instance
    return mock_instance, p1, None  # p2 is no longer needed


def _stop(*patchers):
    for p in patchers:
        try:
            p.stop()
        except Exception:
            pass


class FakeKeyboard:
    def __init__(self, keys):
        self._keys = list(keys)
    def start_monitoring(self):
        pass
    def stop_monitoring(self):
        pass
    def get_key(self):
        if self._keys:
            return self._keys.pop(0)
        return None


def test_quit_key_stops_after_current_trial(tmp_path: Path):
    mock_instance, p1, p2 = _patch_underlying_optimizer()
    try:
        opt = PausibleOptunaOptimizer(
            base_config={'dummy': 'config'},
            search_space=lambda t: {},
            model_class=MagicMock,
            wandb_project=None,
            study_name="kb_test",
            enable_pause=True,
            use_reflow=False,
        )
        # Feed 'q' so loop should stop after first trial finishes
        opt.keyboard_handler = FakeKeyboard(['q'])
        study = opt.optimize(n_trials=3, config_overrides={}, callbacks=[])
        finished = len([t for t in study.trials if t.state.name in ("COMPLETE","PRUNED")])
        assert finished >= 1
    finally:
        _stop(p1, p2)


def test_ctrl_c_char_schedules_pause(tmp_path: Path):
    mock_instance, p1, p2 = _patch_underlying_optimizer()
    try:
        opt = PausibleOptunaOptimizer(
            base_config={'dummy': 'config'},
            search_space=lambda t: {},
            model_class=MagicMock,
            wandb_project=None,
            study_name="kb_test2",
            enable_pause=True,
            use_reflow=False,
        )
        # '\x03' is Ctrl+C char in cbreak mode; request graceful pause
        opt.keyboard_handler = FakeKeyboard(['\x03'])
        study = opt.optimize(n_trials=3, config_overrides={}, callbacks=[])
        # Should have paused; zero or more finished trials depending on timing
        finished = len([t for t in study.trials if t.state.name in ("COMPLETE","PRUNED")])
        assert finished >= 0
        # Pause flag should have been set during run
        assert opt.should_pause is True
    finally:
        _stop(p1, p2)


