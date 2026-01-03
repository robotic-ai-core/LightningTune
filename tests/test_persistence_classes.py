"""
Tests for persistence protocol and implementations.
"""

import pickle
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import optuna


class TestStudyMetadata:
    """Tests for StudyMetadata dataclass."""

    def test_default_values(self):
        """Test metadata creation with required fields."""
        from LightningTune.persistence import StudyMetadata

        metadata = StudyMetadata(
            study_name="test_study",
            total_trials_completed=10,
            sampler_name="tpe",
            pruner_name="median",
            config_overrides={},
        )

        assert metadata.study_name == "test_study"
        assert metadata.total_trials_completed == 10
        assert metadata.sampler_name == "tpe"
        assert metadata.pruner_name == "median"
        assert metadata.config_overrides == {}
        assert metadata.last_wandb_upload_trial_count is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from LightningTune.persistence import StudyMetadata

        metadata = StudyMetadata(
            study_name="test",
            total_trials_completed=5,
            sampler_name="random",
            pruner_name="hyperband",
            config_overrides={"key": "value"},
            last_wandb_upload_trial_count=3,
        )

        d = metadata.to_dict()

        assert d["study_name"] == "test"
        assert d["total_trials_completed"] == 5
        assert d["sampler_name"] == "random"
        assert d["pruner_name"] == "hyperband"
        assert d["config_overrides"] == {"key": "value"}
        assert d["last_wandb_upload_trial_count"] == 3

    def test_from_dict(self):
        """Test creation from dictionary."""
        from LightningTune.persistence import StudyMetadata

        d = {
            "study_name": "from_dict_test",
            "total_trials_completed": 15,
            "sampler_name": "cmaes",
            "pruner_name": "none",
            "config_overrides": {"a": 1},
            "last_wandb_upload_trial_count": 10,
        }

        metadata = StudyMetadata.from_dict(d)

        assert metadata.study_name == "from_dict_test"
        assert metadata.total_trials_completed == 15
        assert metadata.last_wandb_upload_trial_count == 10

    def test_from_dict_with_defaults(self):
        """Test creation from partial dictionary uses defaults."""
        from LightningTune.persistence import StudyMetadata

        metadata = StudyMetadata.from_dict({})

        assert metadata.study_name == "unknown"
        assert metadata.total_trials_completed == 0
        assert metadata.sampler_name == "tpe"
        assert metadata.pruner_name == "median"


class TestLocalPersistence:
    """Tests for LocalPersistence class."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading a study."""
        from LightningTune.persistence import LocalPersistence, StudyMetadata

        checkpoint_dir = tmp_path / "checkpoints"
        persistence = LocalPersistence(checkpoint_dir)

        # Create a simple study
        study = optuna.create_study()
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=3)

        metadata = StudyMetadata(
            study_name="test_local",
            total_trials_completed=3,
            sampler_name="tpe",
            pruner_name="median",
            config_overrides={"key": "value"},
        )

        # Save
        result = persistence.save_study(study, metadata)
        assert result is True
        assert (checkpoint_dir / "study.pkl").exists()
        assert (checkpoint_dir / "session_args.yaml").exists()

        # Load
        session = persistence.load_study("latest")
        assert session is not None
        assert session["study_name"] == "test_local"
        assert session["total_trials_completed"] == 3
        assert len(session["study"].trials) == 3

    def test_load_by_path(self, tmp_path):
        """Test loading by explicit path."""
        from LightningTune.persistence import LocalPersistence, StudyMetadata

        checkpoint_dir = tmp_path / "checkpoints"
        persistence = LocalPersistence(checkpoint_dir)

        study = optuna.create_study()
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=2)

        metadata = StudyMetadata(
            study_name="path_test",
            total_trials_completed=2,
            sampler_name="tpe",
            pruner_name="median",
            config_overrides={},
        )

        persistence.save_study(study, metadata)

        # Load by explicit path
        session = persistence.load_study(str(checkpoint_dir / "study.pkl"))
        assert session is not None
        assert session["study_name"] == "path_test"

    def test_load_nonexistent(self, tmp_path):
        """Test loading from nonexistent location."""
        from LightningTune.persistence import LocalPersistence

        persistence = LocalPersistence(tmp_path / "nonexistent")
        session = persistence.load_study("latest")

        assert session is None

    def test_list_checkpoints(self, tmp_path):
        """Test listing checkpoints."""
        from LightningTune.persistence import LocalPersistence, StudyMetadata

        checkpoint_dir = tmp_path / "checkpoints"
        persistence = LocalPersistence(checkpoint_dir)

        # Initially empty
        assert persistence.list_checkpoints() == []

        # After save
        study = optuna.create_study()
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)
        metadata = StudyMetadata(
            study_name="list_test",
            total_trials_completed=1,
            sampler_name="tpe",
            pruner_name="median",
            config_overrides={},
        )
        persistence.save_study(study, metadata)

        checkpoints = persistence.list_checkpoints()
        assert len(checkpoints) >= 1
        assert any("study.pkl" in cp for cp in checkpoints)


class TestCompositePersistence:
    """Tests for CompositePersistence class."""

    def test_save_to_multiple_backends(self, tmp_path):
        """Test saving to multiple backends."""
        from LightningTune.persistence import (
            CompositePersistence,
            LocalPersistence,
            StudyMetadata,
        )

        dir1 = tmp_path / "backend1"
        dir2 = tmp_path / "backend2"
        backend1 = LocalPersistence(dir1)
        backend2 = LocalPersistence(dir2)

        composite = CompositePersistence([backend1, backend2])

        study = optuna.create_study()
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=2)

        metadata = StudyMetadata(
            study_name="composite_test",
            total_trials_completed=2,
            sampler_name="tpe",
            pruner_name="median",
            config_overrides={},
        )

        result = composite.save_study(study, metadata)
        assert result is True

        # Both backends should have the checkpoint
        assert (dir1 / "study.pkl").exists()
        assert (dir2 / "study.pkl").exists()

    def test_load_from_primary(self, tmp_path):
        """Test loading from primary backend."""
        from LightningTune.persistence import (
            CompositePersistence,
            LocalPersistence,
            StudyMetadata,
        )

        dir1 = tmp_path / "primary"
        dir2 = tmp_path / "secondary"
        backend1 = LocalPersistence(dir1)
        backend2 = LocalPersistence(dir2)

        composite = CompositePersistence([backend1, backend2], primary_index=0)

        study = optuna.create_study()
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=2)

        metadata = StudyMetadata(
            study_name="primary_test",
            total_trials_completed=2,
            sampler_name="tpe",
            pruner_name="median",
            config_overrides={},
        )

        composite.save_study(study, metadata)

        # Load should come from primary
        session = composite.load_study("latest")
        assert session is not None
        assert session["study_name"] == "primary_test"

    def test_fallback_on_primary_failure(self, tmp_path):
        """Test fallback to secondary when primary fails."""
        from LightningTune.persistence import (
            CompositePersistence,
            LocalPersistence,
            StudyMetadata,
        )

        dir1 = tmp_path / "will_be_deleted"
        dir2 = tmp_path / "secondary"
        backend1 = LocalPersistence(dir1)
        backend2 = LocalPersistence(dir2)

        composite = CompositePersistence([backend1, backend2], primary_index=0)

        study = optuna.create_study()
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=2)

        metadata = StudyMetadata(
            study_name="fallback_test",
            total_trials_completed=2,
            sampler_name="tpe",
            pruner_name="median",
            config_overrides={},
        )

        composite.save_study(study, metadata)

        # Delete primary
        import shutil
        shutil.rmtree(dir1)

        # Load should fall back to secondary
        session = composite.load_study("latest")
        assert session is not None
        assert session["study_name"] == "fallback_test"


class TestStudyPersistenceProtocol:
    """Tests for StudyPersistence protocol compliance."""

    def test_local_persistence_implements_protocol(self):
        """Test LocalPersistence implements StudyPersistence protocol."""
        from LightningTune.persistence import LocalPersistence, StudyPersistence

        persistence = LocalPersistence(Path("/tmp/test"))

        # Protocol check
        assert isinstance(persistence, StudyPersistence)

    def test_custom_backend_protocol_compliance(self, tmp_path):
        """Test custom backend can implement protocol."""
        from LightningTune.persistence import StudyPersistence, StudyMetadata

        class CustomPersistence:
            def __init__(self):
                self.storage = {}

            def save_study(self, study, metadata: StudyMetadata) -> bool:
                self.storage["study"] = study
                self.storage["metadata"] = metadata
                return True

            def load_study(self, identifier: str = "latest"):
                if "study" in self.storage:
                    return {
                        "study": self.storage["study"],
                        "study_name": self.storage["metadata"].study_name,
                        "total_trials_completed": self.storage["metadata"].total_trials_completed,
                    }
                return None

            def list_checkpoints(self):
                return ["latest"] if self.storage else []

        custom = CustomPersistence()
        assert isinstance(custom, StudyPersistence)

        # Use it
        study = optuna.create_study()
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)

        metadata = StudyMetadata(
            study_name="custom_test",
            total_trials_completed=1,
            sampler_name="tpe",
            pruner_name="median",
            config_overrides={},
        )

        assert custom.save_study(study, metadata) is True
        session = custom.load_study("latest")
        assert session["study_name"] == "custom_test"


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
