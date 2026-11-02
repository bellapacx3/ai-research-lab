
from types import SimpleNamespace

import numpy as np
import pytest

from src.storage.repositories import (
    get_model_predictions,
    save_experiment,
    save_model_predictions,
)


@pytest.fixture
def saved_experiment(tmp_path):
    db_path = tmp_path / "research_lab.db"

    metadata = {
        "experiment_name": "Prediction persistence test",
        "dataset_path": "data/sample.csv",
        "dataset_sha256": "a" * 64,
        "random_seed": 42,
        "parameters": {"test_size": 0.25},
        "environment": {"python_version": "3.12"},
    }

    results = {
        "Linear Regression": SimpleNamespace(
            metrics={"mae": 0.2, "rmse": 0.3, "r2": 0.8}
        ),
        "Random Forest": SimpleNamespace(
            metrics={"mae": 0.1, "rmse": 0.2, "r2": 0.9}
        ),
    }

    experiment_id = save_experiment(metadata, results, db_path)

    return experiment_id, db_path


def test_save_and_retrieve_predictions(saved_experiment):
    experiment_id, db_path = saved_experiment

    count = save_model_predictions(
        experiment_id=experiment_id,
        model_name="Random Forest",
        sample_indices=[10, 20, 30],
        actual_values=[3.0, 4.0, 5.0],
        predicted_values=[2.5, 4.5, 5.0],
        db_path=db_path,
    )

    records = get_model_predictions(
        experiment_id,
        "Random Forest",
        db_path,
    )

    assert count == 3
    assert len(records) == 3

    first = next(
        row for row in records if row["sample_index"] == "10"
    )

    assert first["actual_value"] == pytest.approx(3.0)
    assert first["predicted_value"] == pytest.approx(2.5)
    assert first["residual"] == pytest.approx(0.5)
    assert first["absolute_error"] == pytest.approx(0.5)
    assert first["squared_error"] == pytest.approx(0.25)


def test_predictions_can_be_filtered_by_model(saved_experiment):
    experiment_id, db_path = saved_experiment

    save_model_predictions(
        experiment_id,
        "Linear Regression",
        [1, 2],
        [2.0, 3.0],
        [2.1, 2.9],
        db_path,
    )

    save_model_predictions(
        experiment_id,
        "Random Forest",
        [1, 2],
        [2.0, 3.0],
        [2.2, 2.8],
        db_path,
    )

    records = get_model_predictions(
        experiment_id,
        "Linear Regression",
        db_path,
    )

    assert len(records) == 2
    assert all(
        row["model_name"] == "Linear Regression"
        for row in records
    )


def test_predictions_can_be_retrieved_for_all_models(saved_experiment):
    experiment_id, db_path = saved_experiment

    save_model_predictions(
        experiment_id,
        "Linear Regression",
        [1],
        [2.0],
        [2.1],
        db_path,
    )

    save_model_predictions(
        experiment_id,
        "Random Forest",
        [1],
        [2.0],
        [1.9],
        db_path,
    )

    records = get_model_predictions(experiment_id, db_path=db_path)

    assert len(records) == 2
    assert {row["model_name"] for row in records} == {
        "Linear Regression",
        "Random Forest",
    }


def test_mismatched_lengths_are_rejected(saved_experiment):
    experiment_id, db_path = saved_experiment

    with pytest.raises(ValueError, match="equal lengths"):
        save_model_predictions(
            experiment_id,
            "Random Forest",
            [1, 2],
            [2.0],
            [2.1, 2.2],
            db_path,
        )


def test_non_finite_predictions_are_rejected(saved_experiment):
    experiment_id, db_path = saved_experiment

    with pytest.raises(ValueError, match="finite"):
        save_model_predictions(
            experiment_id,
            "Random Forest",
            [1, 2],
            [2.0, 3.0],
            [2.1, np.nan],
            db_path,
        )

    assert get_model_predictions(
        experiment_id,
        db_path=db_path,
    ) == []


def test_duplicate_sample_indices_are_rejected(saved_experiment):
    experiment_id, db_path = saved_experiment

    with pytest.raises(ValueError, match="Duplicate sample"):
        save_model_predictions(
            experiment_id,
            "Random Forest",
            [1, 1],
            [2.0, 3.0],
            [2.1, 3.1],
            db_path,
        )


def test_duplicate_prediction_insert_is_rejected(saved_experiment):
    experiment_id, db_path = saved_experiment

    arguments = {
        "experiment_id": experiment_id,
        "model_name": "Random Forest",
        "sample_indices": [1, 2],
        "actual_values": [2.0, 3.0],
        "predicted_values": [2.1, 3.1],
        "db_path": db_path,
    }

    save_model_predictions(**arguments)

    with pytest.raises(Exception):
        save_model_predictions(**arguments)


def test_predictions_require_an_existing_model(saved_experiment):
    experiment_id, db_path = saved_experiment

    with pytest.raises(Exception):
        save_model_predictions(
            experiment_id,
            "Nonexistent Model",
            [1],
            [2.0],
            [2.1],
            db_path,
        )


def test_invalid_experiment_id_is_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        get_model_predictions(0)
