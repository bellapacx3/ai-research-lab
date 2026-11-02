import json

import pytest

from src.experiments.reproducibility import (
    build_experiment_metadata,
    fingerprint_file,
)


def test_fingerprint_is_stable_for_unchanged_file(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_text("x,y\n1,2\n", encoding="utf-8")

    assert fingerprint_file(path) == fingerprint_file(path)


def test_fingerprint_changes_when_file_changes(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_text("x,y\n1,2\n", encoding="utf-8")
    original = fingerprint_file(path)

    path.write_text("x,y\n1,3\n", encoding="utf-8")

    assert fingerprint_file(path) != original


def test_experiment_metadata_contains_reproducibility_details(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_text("x,y\n1,2\n", encoding="utf-8")

    metadata = build_experiment_metadata(
        dataset_path=path,
        experiment_name="Baseline regression",
        random_seed=42,
        parameters={"test_size": 0.2},
    )

    assert metadata["experiment_name"] == "Baseline regression"
    assert metadata["random_seed"] == 42
    assert metadata["parameters"]["test_size"] == 0.2
    assert len(metadata["dataset_sha256"]) == 64
    assert "python_version" in metadata["environment"]

    # Verify the metadata can be serialized for later storage.
    json.dumps(metadata)


def test_empty_experiment_name_is_rejected(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_text("x,y\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Experiment name"):
        build_experiment_metadata(path, "   ", 42)


def test_missing_dataset_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        fingerprint_file(tmp_path / "missing.csv")
