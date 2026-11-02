import json
from types import SimpleNamespace

import pytest

from src.storage.database import initialize_database
from src.storage.repositories import (
    get_experiment,
    list_experiments,
    save_experiment,
)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_research_lab.db"


@pytest.fixture
def metadata():
    return {
        "experiment_name": "Baseline regression",
        "dataset_path": "data/sample.csv",
        "dataset_sha256": "a" * 64,
        "random_seed": 42,
        "parameters": {"test_size": 0.2},
        "environment": {"python_version": "3.12"},
    }


@pytest.fixture
def model_results():
    return {
        "Linear Regression": SimpleNamespace(
            metrics={"mae": 0.25, "rmse": 0.31, "r2": 0.61}
        ),
        "Random Forest": SimpleNamespace(
            metrics={"mae": 0.30, "rmse": 0.33, "r2": 0.59}
        ),
    }


def test_database_initializes(db_path):
    initialize_database(db_path)
    assert db_path.exists()


def test_save_and_retrieve_experiment(db_path, metadata, model_results):
    experiment_id = save_experiment(metadata, model_results, db_path)

    record = get_experiment(experiment_id, db_path)

    assert record is not None
    assert record["experiment_name"] == "Baseline regression"
    assert record["random_seed"] == 42
    assert record["parameters"] == {"test_size": 0.2}
    assert len(record["model_results"]) == 2


def test_experiments_are_ordered_newest_first(
    db_path, metadata, model_results
):
    first_id = save_experiment(metadata, model_results, db_path)

    second_metadata = {
        **metadata,
        "experiment_name": "Second run",
    }
    second_id = save_experiment(second_metadata, model_results, db_path)

    records = list_experiments(db_path)

    assert records[0]["id"] == second_id
    assert records[1]["id"] == first_id


def test_missing_metadata_is_rejected(db_path, model_results):
    with pytest.raises(ValueError, match="Missing metadata fields"):
        save_experiment({}, model_results, db_path)


def test_empty_model_results_are_rejected(db_path, metadata):
    with pytest.raises(ValueError, match="At least one model"):
        save_experiment(metadata, {}, db_path)


def test_invalid_limit_is_rejected(db_path):
    with pytest.raises(ValueError, match="limit"):
        list_experiments(db_path, limit=0)


def test_missing_experiment_returns_none(db_path):
    assert get_experiment(999, db_path) is None


def test_metadata_is_valid_json(db_path, metadata, model_results):
    experiment_id = save_experiment(metadata, model_results, db_path)
    record = get_experiment(experiment_id, db_path)

    json.dumps(record["parameters"])
    json.dumps(record["environment"])
