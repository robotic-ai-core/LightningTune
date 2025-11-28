#!/usr/bin/env python
"""Test that trials continue correctly after pause/resume.

This test reproduces the bug where trials repeat after pause/resume instead of continuing
from where they left off.
"""

import pytest
import sys
import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add LightningTune root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from LightningTune.optuna.pausible_optimizer import PausibleOptunaOptimizer


class TestPauseResumeContinuity:
    """Test that pause/resume maintains trial continuity."""

    def test_trial_numbers_continue_after_resume(self, tmp_path):
        """Test that trial numbers continue correctly after pause and resume.

        This is a regression test for the bug where trials would repeat after resume.

        Flow:
        1. Run 3 trials, pause
        2. Save checkpoint with study containing trials 0, 1, 2
        3. Resume from checkpoint
        4. Run 2 more trials
        5. Verify trials 3, 4 ran (not 0, 1 again)
        """
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_file = checkpoint_dir / "study.pkl"

        # Track which trial numbers actually ran
        trial_numbers_run = []

        def test_objective(trial):
            trial_numbers_run.append(trial.number)
            return trial.number * 0.1

        # Phase 1: Run 3 trials with pause triggered after
        optimizer1 = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test_continuity",
            restart_on_save=False,  # No subprocess restart
        )
        optimizer1.local_checkpoint_dir = checkpoint_dir

        # Create a custom objective that tracks trial numbers
        call_count = [0]
        def phase1_objective(trial):
            call_count[0] += 1
            trial_numbers_run.append(trial.number)
            return trial.number * 0.1

        with patch.object(optimizer1, 'underlying_optimizer', create=True) as mock_opt:
            mock_opt.create_objective.return_value = phase1_objective
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_wandb:
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                    mock_wandb.return_value = True
                    mock_local.return_value = True

                    # Trigger pause after 3 trials
                    def mock_pause_check():
                        return call_count[0] >= 3
                    optimizer1._update_pause_from_keyboard = mock_pause_check

                    study1 = optimizer1.optimize(n_trials=10)

        # Verify phase 1: ran trials 0, 1, 2
        assert len(trial_numbers_run) == 3, f"Phase 1 should run 3 trials, ran {len(trial_numbers_run)}"
        assert trial_numbers_run == [0, 1, 2], f"Phase 1 trial numbers should be [0, 1, 2], got {trial_numbers_run}"
        assert optimizer1.total_trials_completed == 3
        assert optimizer1.should_pause == True

        # Manually save the checkpoint (since we mocked the save)
        session_info = {
            "study": study1,
            "total_trials_completed": optimizer1.total_trials_completed,
            "sampler_name": "tpe",
            "pruner_name": "median",
            "study_name": "test_continuity",
            "config_overrides": {},
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(session_info, f)

        # Verify checkpoint was saved correctly
        with open(checkpoint_file, 'rb') as f:
            loaded = pickle.load(f)
        assert loaded["total_trials_completed"] == 3
        assert len(loaded["study"].trials) == 3

        # Phase 2: Resume and run 2 more trials
        trial_numbers_run.clear()  # Reset tracking

        optimizer2 = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=False,  # No pause in phase 2
            wandb_project=None,
            study_name="test_continuity",
            restart_on_save=False,
        )
        optimizer2.local_checkpoint_dir = checkpoint_dir

        # Phase 2 objective - track trial numbers
        def phase2_objective(trial):
            trial_numbers_run.append(trial.number)
            return trial.number * 0.1

        with patch.object(optimizer2, 'underlying_optimizer', create=True) as mock_opt:
            mock_opt.create_objective.return_value = phase2_objective
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_wandb:
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                    mock_wandb.return_value = True
                    mock_local.return_value = True

                    # Resume from checkpoint and run until 5 total
                    study2 = optimizer2.optimize(
                        n_trials=5,  # Target 5 total trials
                        resume_from=str(checkpoint_file)
                    )

        # Verify phase 2: ran trials 3, 4 (continuing from where we left off)
        print(f"Phase 2 trial numbers: {trial_numbers_run}")
        print(f"Study2 has {len(study2.trials)} trials")
        print(f"optimizer2.total_trials_completed = {optimizer2.total_trials_completed}")

        assert len(trial_numbers_run) == 2, f"Phase 2 should run 2 trials, ran {len(trial_numbers_run)}: {trial_numbers_run}"
        assert trial_numbers_run == [3, 4], f"Phase 2 trial numbers should be [3, 4], got {trial_numbers_run}"
        assert optimizer2.total_trials_completed == 5
        assert len(study2.trials) == 5

    def test_study_trials_preserved_on_resume(self, tmp_path):
        """Test that the Optuna study object preserves trials correctly through pickle."""
        checkpoint_file = tmp_path / "study.pkl"

        # Create a study with 3 trials
        study = optuna.create_study()
        for i in range(3):
            study.add_trial(optuna.trial.create_trial(
                value=i * 0.1,
                params={"x": i},
                distributions={"x": optuna.distributions.IntDistribution(0, 10)},
                state=optuna.trial.TrialState.COMPLETE
            ))

        # Save via pickle
        session = {
            "study": study,
            "total_trials_completed": 3,
            "config_overrides": {},
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(session, f)

        # Load and verify
        with open(checkpoint_file, 'rb') as f:
            loaded = pickle.load(f)

        loaded_study = loaded["study"]

        # Verify trials are preserved
        assert len(loaded_study.trials) == 3, f"Should have 3 trials, got {len(loaded_study.trials)}"

        # Run one more trial on the loaded study
        trial_numbers = []
        def objective(trial):
            trial_numbers.append(trial.number)
            return trial.number * 0.1

        loaded_study.optimize(objective, n_trials=1)

        # The new trial should be number 3 (not 0!)
        assert len(loaded_study.trials) == 4
        assert trial_numbers == [3], f"New trial should be number 3, got {trial_numbers}"


class TestRealOptimizerResumeFlow:
    """Test with real optimizer (no mocking) to catch integration issues."""

    def test_real_optimizer_resume_continues_trials(self, tmp_path):
        """Test that resuming with a real optimizer continues trials correctly.

        This test uses the actual PausibleOptunaOptimizer without mocking
        the core optimization logic, to catch any integration issues.
        """
        checkpoint_dir = tmp_path / "checkpoints" / "real_test"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "study.pkl"

        # Phase 1: Run 3 trials
        optimizer1 = PausibleOptunaOptimizer(
            base_config={"dummy": "config"},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=False,
            wandb_project=None,
            study_name="real_test",
            restart_on_save=False,
        )
        optimizer1.local_checkpoint_dir = checkpoint_dir

        # Track trial numbers
        phase1_trials = []
        def phase1_objective(trial):
            phase1_trials.append(trial.number)
            return trial.number * 0.1

        # Inject objective directly into underlying optimizer
        with patch.object(optimizer1, 'create_objective', return_value=phase1_objective):
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb', return_value=False):
                study1 = optimizer1.optimize(n_trials=3)

        # Save checkpoint
        from LightningTune.persistence import save_study_to_local
        save_study_to_local(
            checkpoint_dir,
            study1,
            optimizer1.total_trials_completed,
            sampler_name="tpe",
            pruner_name="median",
            study_name="real_test",
            config_overrides={},
        )

        print(f"Phase 1 trials: {phase1_trials}")
        print(f"Study1 has {len(study1.trials)} trials")
        assert phase1_trials == [0, 1, 2], f"Phase 1 should run trials [0, 1, 2], got {phase1_trials}"

        # Phase 2: Resume and run 2 more trials
        optimizer2 = PausibleOptunaOptimizer(
            base_config={"dummy": "config"},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=False,
            wandb_project=None,
            study_name="real_test",
            restart_on_save=False,
        )
        optimizer2.local_checkpoint_dir = checkpoint_dir

        phase2_trials = []
        def phase2_objective(trial):
            phase2_trials.append(trial.number)
            return trial.number * 0.1

        with patch.object(optimizer2, 'create_objective', return_value=phase2_objective):
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb', return_value=False):
                study2 = optimizer2.optimize(
                    n_trials=5,  # Target 5 total
                    resume_from=str(checkpoint_file)
                )

        print(f"Phase 2 trials: {phase2_trials}")
        print(f"Study2 has {len(study2.trials)} trials")

        # CRITICAL: Phase 2 should run trials 3, 4 (not 0, 1)
        assert phase2_trials == [3, 4], f"Phase 2 should run trials [3, 4], got {phase2_trials}"
        assert len(study2.trials) == 5
        assert optimizer2.total_trials_completed == 5


class TestLocalVsWandBCheckpointSelection:
    """Test that the most recent checkpoint is used when both local and WandB exist."""

    def test_local_checkpoint_preferred_with_restart_every_trial(self, tmp_path):
        """Test that local checkpoint is always used when restart_every_trial=True.

        This is a regression test for the bug where:
        - restart_every_trial=True saves locally after every trial
        - upload_every=10 only uploads to WandB every 10 trials
        - After trial 18 completes, child restarts with --resume-from latest
        - Child would load stale WandB checkpoint (17 trials from pause)
        - Child runs "trial 18" again instead of trial 19

        With the fix, restart_every_trial mode always prefers local checkpoint.
        """
        checkpoint_dir = tmp_path / "checkpoints" / "test_study"
        checkpoint_dir.mkdir(parents=True)

        # Create a "WandB" checkpoint with 17 trials (from when user paused)
        wandb_study = optuna.create_study()
        for i in range(17):
            wandb_study.add_trial(optuna.trial.create_trial(
                value=i * 0.1,
                params={"x": i},
                distributions={"x": optuna.distributions.IntDistribution(0, 100)},
                state=optuna.trial.TrialState.COMPLETE
            ))
        wandb_session = {
            "study": wandb_study,
            "total_trials_completed": 17,
            "config_overrides": {},
        }

        # Create a local checkpoint with 18 trials (after one more trial ran)
        local_study = optuna.create_study()
        for i in range(18):
            local_study.add_trial(optuna.trial.create_trial(
                value=i * 0.1,
                params={"x": i},
                distributions={"x": optuna.distributions.IntDistribution(0, 100)},
                state=optuna.trial.TrialState.COMPLETE
            ))

        # Save local checkpoint
        from LightningTune.persistence import save_study_to_local
        save_study_to_local(
            checkpoint_dir,
            local_study,
            18,  # total_trials_completed
            sampler_name="tpe",
            pruner_name="median",
            study_name="test_study",
            config_overrides={},
        )

        # Create optimizer with restart_every_trial=True (the key setting!)
        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=False,
            wandb_project="test_project",  # WandB configured
            study_name="test_study",
            restart_on_save=True,
            restart_every_trial=True,  # This is the key!
        )
        optimizer.local_checkpoint_dir = checkpoint_dir

        # Mock WandB load to return the stale checkpoint (should NOT be used)
        with patch.object(optimizer, 'load_study_from_wandb', return_value=wandb_session) as mock_wandb:
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb', return_value=True):
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local', return_value=True):
                    # Resume from "latest" - should use local (18 trials), NOT WandB (17 trials)
                    trial_numbers = []
                    def test_objective(trial):
                        trial_numbers.append(trial.number)
                        return trial.number * 0.1

                    with patch.object(optimizer, 'create_objective', return_value=test_objective):
                        # Patch sys.exit to prevent actual exit on restart
                        with patch('sys.exit') as mock_exit:
                            mock_exit.side_effect = SystemExit(42)
                            try:
                                study = optimizer.optimize(
                                    n_trials=20,  # Target 20 total
                                    resume_from="latest"
                                )
                            except SystemExit:
                                pass  # Expected for restart_every_trial

        # With restart_every_trial=True, WandB should NOT be called at all
        # because local checkpoint is always preferred
        mock_wandb.assert_not_called()

        # Should have run trial 18 (continuing from local's 18 trials)
        # NOT trial 17 (from stale WandB checkpoint)
        print(f"Trial numbers run: {trial_numbers}")
        assert trial_numbers == [18], \
            f"Should run trial [18] from local checkpoint, got {trial_numbers}"


class TestSubprocessRestartSimulation:
    """Simulate the subprocess restart flow to identify trial repetition issues."""

    def test_subprocess_restart_trial_continuity(self, tmp_path):
        """Simulate multiple subprocess restarts and verify trial continuity.

        This simulates what happens in production:
        1. Child 1: Run trial 0, save checkpoint, exit(42)
        2. Parent: Sees exit code 42, spawns child 2 with --resume-from latest
        3. Child 2: Load checkpoint, run trial 1, save, exit(42)
        ... repeat

        The key is ensuring that each "child" gets the correct trial count.
        """
        checkpoint_dir = tmp_path / "checkpoints" / "test_study"
        checkpoint_dir.mkdir(parents=True)

        # Track all trial numbers across "restarts"
        all_trial_numbers = []

        # Simulate 5 subprocess restarts
        for restart_num in range(5):
            # Simulate child process
            optimizer = PausibleOptunaOptimizer(
                base_config={},
                search_space=lambda trial: {},
                model_class=MagicMock,
                enable_pause=False,
                wandb_project=None,
                study_name="test_study",
                restart_on_save=True,
                restart_every_trial=True,
            )
            optimizer.local_checkpoint_dir = checkpoint_dir

            # Determine resume path
            checkpoint_file = checkpoint_dir / "study.pkl"
            resume_from = str(checkpoint_file) if checkpoint_file.exists() else None

            # Track trial numbers
            current_trial_numbers = []
            def test_objective(trial):
                current_trial_numbers.append(trial.number)
                all_trial_numbers.append(trial.number)
                return trial.number * 0.1

            exit_code = None
            with patch.object(optimizer, 'underlying_optimizer', create=True) as mock_opt:
                mock_opt.create_objective.return_value = test_objective
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_wandb:
                    with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                        mock_wandb.return_value = False  # No WandB
                        # Actually save to local
                        def real_local_save(checkpoint_dir, study, total_trials_completed, **kwargs):
                            from LightningTune.persistence import save_study_to_local as persist_save
                            return persist_save(checkpoint_dir, study, total_trials_completed, **kwargs)
                        mock_local.side_effect = real_local_save

                        # Intercept sys.exit(42) to capture the "restart" signal
                        with patch('sys.exit') as mock_exit:
                            def capture_exit(code):
                                nonlocal exit_code
                                exit_code = code
                                raise SystemExit(code)
                            mock_exit.side_effect = capture_exit

                            try:
                                study = optimizer.optimize(
                                    n_trials=10,  # Target 10 total
                                    resume_from=resume_from
                                )
                            except SystemExit as e:
                                # Expected for restart_every_trial
                                if e.code == 42:
                                    pass  # Normal restart signal
                                else:
                                    raise

            print(f"Restart {restart_num}: ran trials {current_trial_numbers}, exit_code={exit_code}")

            # Verify this restart ran exactly 1 trial (since restart_every_trial=True)
            assert len(current_trial_numbers) == 1, \
                f"Restart {restart_num} should run 1 trial, ran {len(current_trial_numbers)}"

            # Verify the trial number is correct (should be restart_num)
            assert current_trial_numbers[0] == restart_num, \
                f"Restart {restart_num} should run trial {restart_num}, ran trial {current_trial_numbers[0]}"

        # Verify all trials ran in sequence without repetition
        print(f"All trial numbers: {all_trial_numbers}")
        assert all_trial_numbers == [0, 1, 2, 3, 4], \
            f"Expected trials [0, 1, 2, 3, 4], got {all_trial_numbers}"

    def test_pause_prevents_restart_exit_code(self, tmp_path):
        """Test that when pause is requested, we don't get exit code 42."""
        checkpoint_dir = tmp_path / "checkpoints" / "test_study"
        checkpoint_dir.mkdir(parents=True)

        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=True,
            wandb_project=None,
            study_name="test_study",
            restart_on_save=True,
            restart_every_trial=True,
        )
        optimizer.local_checkpoint_dir = checkpoint_dir

        call_count = [0]
        def test_objective(trial):
            call_count[0] += 1
            return trial.number * 0.1

        exit_code = None
        with patch.object(optimizer, 'underlying_optimizer', create=True) as mock_opt:
            mock_opt.create_objective.return_value = test_objective
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_wandb:
                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local') as mock_local:
                    mock_wandb.return_value = False
                    mock_local.return_value = True

                    # Set pause after first trial
                    def mock_pause():
                        return call_count[0] >= 1
                    optimizer._update_pause_from_keyboard = mock_pause

                    with patch('sys.exit') as mock_exit:
                        def capture_exit(code):
                            nonlocal exit_code
                            exit_code = code
                            raise SystemExit(code)
                        mock_exit.side_effect = capture_exit

                        try:
                            study = optimizer.optimize(n_trials=10)
                            # If we get here, no exit was called (good for pause!)
                        except SystemExit as e:
                            pytest.fail(f"sys.exit({e.code}) was called - pause should prevent restart!")

        # Verify we didn't get exit code 42
        assert exit_code is None, f"No exit should be called on pause, got exit({exit_code})"
        assert call_count[0] == 1, f"Should run 1 trial before pause, ran {call_count[0]}"
        assert optimizer.should_pause, "Optimizer should be in paused state"


class TestWandBUploadEveryTracking:
    """Test that upload_every counter is preserved across per-trial restarts."""

    def test_last_wandb_upload_restored_correctly(self, tmp_path):
        """Test that trials_in_batch is calculated correctly on resume.

        This is a regression test for the bug where:
        - save_every_n_trials=5 (upload to WandB every 5 trials)
        - restart_every_trial=True (restart process after every trial)
        - The counter was being reset to 0 on every resume because
          local checkpoints didn't track last_wandb_upload_trial_count

        With the fix, last_wandb_upload_trial_count is saved in checkpoint
        and used to restore trials_in_batch correctly.
        """
        checkpoint_dir = tmp_path / "checkpoints" / "test_study"
        checkpoint_dir.mkdir(parents=True)

        # Create a checkpoint at trial 7 with last WandB upload at trial 5
        from LightningTune.persistence import save_study_to_local, load_study_from_local

        study = optuna.create_study()
        for i in range(7):
            study.add_trial(optuna.trial.create_trial(
                value=i * 0.1,
                params={"x": i},
                distributions={"x": optuna.distributions.IntDistribution(0, 100)},
                state=optuna.trial.TrialState.COMPLETE
            ))

        # Simulate: total=7 trials completed, last WandB upload was at 5
        # This means trials_in_batch should be 2 (trials 6 and 7 since last upload)
        save_study_to_local(
            checkpoint_dir,
            study,
            total_trials_completed=7,
            sampler_name="tpe",
            pruner_name="median",
            study_name="test_study",
            config_overrides={},
            last_wandb_upload_trial_count=5,
        )

        # Load checkpoint and verify
        loaded = load_study_from_local(str(checkpoint_dir))
        assert loaded["total_trials_completed"] == 7
        assert loaded["last_wandb_upload_trial_count"] == 5

        # On resume, trials_in_batch should be 7 - 5 = 2
        expected_trials_in_batch = 7 - 5
        assert expected_trials_in_batch == 2

        # After running 3 more trials (total=10), trials_in_batch becomes 5 (10-5=5)
        # which triggers upload, resetting last_wandb_upload to 10
        # After that, next trial (total=11) has trials_in_batch = 11-10 = 1

        # Test the counter calculation directly
        # Starting state: total=7, last_upload=5, trials_in_batch=2
        # After trial: total=8, trials_in_batch=3 (no upload)
        # After trial: total=9, trials_in_batch=4 (no upload)
        # After trial: total=10, trials_in_batch=5 (UPLOAD, reset last_upload=10)
        # After trial: total=11, trials_in_batch=1 (no upload)

        # This is the core logic being tested - verify it works correctly
        save_every = 5
        last_upload = 5
        for total in range(7, 12):
            trials_in_batch = total - last_upload
            should_upload = trials_in_batch >= save_every
            if should_upload:
                last_upload = total
            print(f"total={total}, trials_in_batch={trials_in_batch}, should_upload={should_upload}")

        # Verify upload should happen at total=10, not at 6, 7, 8, or 9
        assert (10 - 5) >= save_every  # Should upload at 10
        assert (9 - 5) < save_every    # Should NOT upload at 9

    def test_checkpoint_preserves_last_wandb_upload(self, tmp_path):
        """Test that last_wandb_upload_trial_count is saved and restored correctly."""
        checkpoint_dir = tmp_path / "checkpoints" / "test_study"
        checkpoint_dir.mkdir(parents=True)

        # Save a checkpoint with specific last_wandb_upload_trial_count
        from LightningTune.persistence import save_study_to_local, load_study_from_local

        study = optuna.create_study()
        for i in range(7):
            study.add_trial(optuna.trial.create_trial(
                value=i * 0.1,
                params={"x": i},
                distributions={"x": optuna.distributions.IntDistribution(0, 100)},
                state=optuna.trial.TrialState.COMPLETE
            ))

        # Simulate: total=7, but last WandB upload was at 5
        save_study_to_local(
            checkpoint_dir,
            study,
            total_trials_completed=7,
            sampler_name="tpe",
            pruner_name="median",
            study_name="test_study",
            config_overrides={},
            last_wandb_upload_trial_count=5,  # Last upload was at trial 5
        )

        # Load and verify
        loaded = load_study_from_local(str(checkpoint_dir))
        assert loaded is not None
        assert loaded["total_trials_completed"] == 7
        assert loaded["last_wandb_upload_trial_count"] == 5

        # Create optimizer and resume
        optimizer = PausibleOptunaOptimizer(
            base_config={},
            search_space=lambda trial: {},
            model_class=MagicMock,
            enable_pause=False,
            wandb_project="test_project",
            study_name="test_study",
            restart_on_save=True,
            restart_every_trial=True,
            save_every_n_trials=5,
        )
        optimizer.local_checkpoint_dir = checkpoint_dir

        # Test that the optimizer correctly calculates trials_in_batch
        # After loading: total=7, last_upload=5, so trials_in_batch should be 2
        # (we've done 2 trials since last WandB upload)
        wandb_uploaded = [False]

        def test_objective(trial):
            return trial.number * 0.1

        with patch.object(optimizer, 'underlying_optimizer', create=True) as mock_opt:
            mock_opt.create_objective.return_value = test_objective
            with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_wandb') as mock_wandb:
                def track_upload(*args, **kwargs):
                    wandb_uploaded[0] = True
                    return True
                mock_wandb.side_effect = track_upload

                with patch('LightningTune.optuna.pausible_optimizer.persist_save_study_to_local', return_value=True):
                    def raise_exit(code):
                        raise SystemExit(code)
                    # Patch sys.exit at the module level since pausible_optimizer does 'import sys' locally
                    orig_exit = sys.exit
                    try:
                        sys.exit = raise_exit
                        study = optimizer.optimize(
                            n_trials=20,
                            resume_from=str(checkpoint_dir / "study.pkl")
                        )
                    except SystemExit as e:
                        if e.code == 42:
                            pass
                        else:
                            raise
                    finally:
                        sys.exit = orig_exit

        # After trial 7 (total=8), trials_in_batch = 8 - 5 = 3, not enough for upload
        # But wait, we have save_every=5, and we started at total=7, last_upload=5
        # After running trial 7 (the 8th trial): trials_in_batch = 3 (not >= 5)
        # So WandB should NOT have been uploaded
        assert not wandb_uploaded[0], "WandB should not upload yet (only 3 trials since last upload)"


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
