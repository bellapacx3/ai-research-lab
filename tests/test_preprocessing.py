import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    PreprocessingError,
    prepare_dataset,
)


@pytest.fixture
def research_data():
    return pd.DataFrame(
        {
            "temperature": [
                18.0, 19.0, 20.0, 21.0, 22.0,
                23.0, 24.0, 25.0, 26.0, 27.0,
            ],
            "humidity": [
                40.0, 42.0, np.nan, 45.0, 48.0,
                50.0, 51.0, 53.0, 55.0, 57.0,
            ],
            "material": [
                "steel", "wood", "steel", "plastic", "wood",
                "steel", "plastic", "wood", "steel", "plastic",
            ],
            "measurement": [
                10.0, 8.0, 10.5, 9.0, 8.5,
                11.0, 9.5, 8.2, 10.8, 9.7,
            ],
        }
    )


def test_regression_prepares_numeric_and_categorical_features(
    research_data,
):
    result = prepare_dataset(
        research_data,
        target_column="measurement",
        task="regression",
        random_seed=42,
    )

    assert result.X_train.shape[0] == 8
    assert result.X_test.shape[0] == 2
    assert result.X_train.shape[1] == result.X_test.shape[1]
    assert np.isfinite(result.X_train).all()
    assert np.isfinite(result.X_test).all()
    assert result.target_column == "measurement"


def test_same_seed_produces_same_split(research_data):
    first = prepare_dataset(
        research_data, "measurement", random_seed=42
    )
    second = prepare_dataset(
        research_data, "measurement", random_seed=42
    )

    np.testing.assert_array_equal(first.y_train, second.y_train)
    np.testing.assert_array_equal(first.y_test, second.y_test)


def test_missing_target_rows_are_excluded(research_data):
    research_data.loc[0, "measurement"] = np.nan

    result = prepare_dataset(
        research_data, "measurement"
    )

    assert len(result.y_train) + len(result.y_test) == 9


def test_missing_target_column_is_rejected(research_data):
    with pytest.raises(PreprocessingError, match="was not found"):
        prepare_dataset(research_data, "unknown_target")


def test_duplicate_column_names_are_rejected(research_data):
    research_data.columns = [
        "temperature",
        "humidity",
        "material",
        "material",
    ]

    with pytest.raises(PreprocessingError, match="Duplicate column names"):
        prepare_dataset(research_data, "material")


def test_invalid_task_is_rejected(research_data):
    with pytest.raises(PreprocessingError, match="Task must be"):
        prepare_dataset(
            research_data,
            "measurement",
            task="clustering",
        )


def test_non_numeric_regression_target_is_rejected(research_data):
    with pytest.raises(PreprocessingError, match="numeric target"):
        prepare_dataset(
            research_data,
            "material",
            task="regression",
        )
