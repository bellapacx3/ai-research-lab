import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    PreparedDataset,
    PreprocessingError,
    prepare_dataset,
)
from src.ml.classical_models import evaluate_regression_models


@pytest.fixture
def prepared_data():
    frame = pd.DataFrame({
        "temperature": range(18, 30),
        "humidity": [40, 42, 43, 45, 48, 50, 51, 53, 55, 57, 58, 60],
        "material": [
            "steel", "wood", "steel", "plastic", "wood", "steel",
            "plastic", "wood", "steel", "plastic", "wood", "steel",
        ],
        "measurement": [
            10.0, 8.0, 10.5, 9.0, 8.5, 11.0,
            9.5, 8.2, 10.8, 9.7, 8.8, 11.4,
        ],
    })

    return prepare_dataset(
        frame,
        target_column="measurement",
        task="regression",
        test_size=0.25,
        random_seed=42,
    )


def test_both_models_return_evaluation_metrics(prepared_data):
    results = evaluate_regression_models(prepared_data)

    assert set(results) == {"Linear Regression", "Random Forest"}

    for result in results.values():
        assert len(result.predictions) == len(prepared_data.y_test)
        assert set(result.metrics) == {"mae", "rmse", "r2"}
        assert result.metrics["mae"] >= 0
        assert result.metrics["rmse"] >= 0
        assert np.isfinite(result.predictions).all()


def test_random_forest_is_reproducible(prepared_data):
    first = evaluate_regression_models(prepared_data, random_seed=7)
    second = evaluate_regression_models(prepared_data, random_seed=7)

    np.testing.assert_allclose(
        first["Random Forest"].predictions,
        second["Random Forest"].predictions,
    )


def test_classification_data_is_rejected(prepared_data):
    frame = pd.DataFrame({
        "feature": range(20),
        "label": ["a", "b"] * 10,
    })
    classification_data = prepare_dataset(
        frame, "label", task="classification"
    )

    with pytest.raises(PreprocessingError, match="prepared for regression"):
        evaluate_regression_models(classification_data)


def test_invalid_number_of_trees_is_rejected(prepared_data):
    with pytest.raises(ValueError, match="at least 1"):
        evaluate_regression_models(
            prepared_data,
            random_forest_estimators=0,
        )


def test_r2_is_nan_for_one_test_sample(prepared_data):
    single_sample = PreparedDataset(
        X_train=prepared_data.X_train,
        X_test=prepared_data.X_test[:1],
        y_train=prepared_data.y_train,
        y_test=np.asarray(prepared_data.y_test)[:1],
        preprocessor=prepared_data.preprocessor,
        feature_columns=prepared_data.feature_columns,
        target_column=prepared_data.target_column,
        task="regression",
    )

    results = evaluate_regression_models(single_sample)

    assert np.isnan(results["Linear Regression"].metrics["r2"])
