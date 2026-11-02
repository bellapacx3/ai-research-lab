from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.ml.checkpoints import (
    load_checkpoint,
    predict_with_checkpoint,
    save_checkpoint,
)
from src.storage.repositories import get_experiment, save_experiment


@pytest.fixture
def fitted_components():
    features = pd.DataFrame({
        "temperature": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        "humidity": [30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
    })
    target = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])

    preprocessor = StandardScaler()
    transformed = preprocessor.fit_transform(features)

    model = LinearRegression()
    model.fit(transformed, target)

    return features, preprocessor, model


@pytest.fixture
def metadata():
    return {
        "experiment_name": "Checkpoint integration test",
        "dataset_path": "data/sample.csv",
        "dataset_sha256": "b" * 64,
        "random_seed": 42,
        "parameters": {"test_size": 0.2},
        "environment": {"python_version": "3.12"},
    }


@pytest.fixture
def results():
    return {
        "Linear Regression": SimpleNamespace(
            metrics={"mae": 0.1, "rmse": 0.2, "r2": 0.98}
        )
    }


def test_save_and_reload_checkpoint(tmp_path, fitted_components):
    features, preprocessor, model = fitted_components
    path = tmp_path / "models" / "linear.joblib"

    expected = model.predict(preprocessor.transform(features))

    save_checkpoint(
        model=model,
        preprocessor=preprocessor,
        feature_columns=list(features.columns),
        target_column="measurement",
        task="regression",
        checkpoint_path=path,
    )

    checkpoint = load_checkpoint(path)
    actual = checkpoint["model"].predict(
        checkpoint["preprocessor"].transform(
            features[checkpoint["feature_columns"]]
        )
    )

    np.testing.assert_allclose(actual, expected)
    assert checkpoint["target_column"] == "measurement"
    assert checkpoint["task"] == "regression"


def test_predict_from_raw_features(tmp_path, fitted_components):
    features, preprocessor, model = fitted_components
    path = tmp_path / "linear.joblib"

    save_checkpoint(
        model,
        preprocessor,
        list(features.columns),
        "measurement",
        "regression",
        path,
    )

    expected = model.predict(preprocessor.transform(features))
    actual = predict_with_checkpoint(path, features)

    np.testing.assert_allclose(actual, expected)


def test_missing_feature_is_rejected(tmp_path, fitted_components):
    features, preprocessor, model = fitted_components
    path = tmp_path / "linear.joblib"

    save_checkpoint(
        model,
        preprocessor,
        list(features.columns),
        "measurement",
        "regression",
        path,
    )

    with pytest.raises(ValueError, match="Missing required feature"):
        predict_with_checkpoint(path, features.drop(columns=["humidity"]))


def test_missing_checkpoint_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.joblib")


def test_checkpoint_is_linked_to_experiment(
    tmp_path, fitted_components, metadata, results
):
    features, preprocessor, model = fitted_components
    db_path = tmp_path / "experiments.db"

    experiment_id = save_experiment(metadata, results, db_path)
    path = tmp_path / "linear.joblib"

    save_checkpoint(
        model,
        preprocessor,
        list(features.columns),
        "measurement",
        "regression",
        path,
        metrics={"mae": 0.1, "rmse": 0.2, "r2": 0.98},
        experiment_id=experiment_id,
        model_name="Linear Regression",
        db_path=db_path,
    )

    record = get_experiment(experiment_id, db_path)

    assert record is not None
    assert record["model_results"][0]["checkpoint_path"] == str(path.resolve())


def test_wrong_extension_is_rejected(tmp_path, fitted_components):
    features, preprocessor, model = fitted_components

    with pytest.raises(ValueError, match=".joblib"):
        save_checkpoint(
            model,
            preprocessor,
            list(features.columns),
            "measurement",
            "regression",
            tmp_path / "model.pkl",
        )
