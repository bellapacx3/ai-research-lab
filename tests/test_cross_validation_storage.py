
from types import SimpleNamespace

import pandas as pd
import pytest

from src.storage.repositories import (
    get_cross_validation_results,
    save_cross_validation_experiment,
)


@pytest.fixture
def cv_result():
    return SimpleNamespace(
        summary=pd.DataFrame([
            {
                "model": "Random Forest",
                "mae_mean": 0.32,
                "mae_std": 0.01,
                "rmse_mean": 0.50,
                "rmse_std": 0.02,
                "r2_mean": 0.81,
                "r2_std": 0.01,
            },
        ]),
        fold_metrics=pd.DataFrame([
            {
                "fold": 1,
                "model": "Random Forest",
                "train_rows": 80,
                "validation_rows": 20,
                "mae": 0.31,
                "rmse": 0.49,
                "r2": 0.82,
            },
            {
                "fold": 2,
                "model": "Random Forest",
                "train_rows": 80,
                "validation_rows": 20,
                "mae": 0.33,
                "rmse": 0.51,
                "r2": 0.80,
            },
        ]),
    )


@pytest.fixture
def cv_metadata():
    return {
        "experiment_name": "California Housing CV",
        "dataset_path": "data/datasets/california_housing.csv",
        "dataset_sha256": "b" * 64,
        "random_seed": 42,
        "parameters": {"n_splits": 2, "test_size": None},
        "environment": {"python_version": "3.12"},
    }


def test_save_and_retrieve_cross_validation(
    tmp_path, cv_metadata, cv_result
):
    db_path = tmp_path / "research_lab.db"

    experiment_id = save_cross_validation_experiment(
        cv_metadata, cv_result, db_path
    )
    result = get_cross_validation_results(experiment_id, db_path)

    assert result is not None
    assert result["experiment"]["experiment_name"] == (
        "California Housing CV"
    )
    assert len(result["summary"]) == 1
    assert len(result["folds"]) == 2
    assert result["summary"][0]["rmse_mean"] == pytest.approx(0.50)
    assert result["folds"][1]["fold"] == 2


def test_missing_cross_validation_experiment_returns_none(tmp_path):
    db_path = tmp_path / "research_lab.db"

    assert get_cross_validation_results(999, db_path) is None


def test_invalid_cross_validation_metrics_are_rejected(
    tmp_path, cv_metadata, cv_result
):
    cv_result.fold_metrics.loc[0, "rmse"] = float("nan")

    with pytest.raises(ValueError, match="invalid rmse"):
        save_cross_validation_experiment(
            cv_metadata, cv_result, tmp_path / "research_lab.db"
        )
