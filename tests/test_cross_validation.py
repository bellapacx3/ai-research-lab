
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import PreprocessingError
from src.experiments.cross_validation import (
    cross_validate_regression_models,
)


def make_frame():
    rng = np.random.default_rng(123)
    rows = 60

    temperature = rng.uniform(10, 35, rows)
    humidity = rng.uniform(20, 90, rows)
    material = rng.choice(["steel", "wood", "plastic"], rows)

    material_effect = pd.Series(material).map(
        {"steel": 2.0, "wood": -1.0, "plastic": 0.5}
    ).to_numpy()

    measurement = (
        0.4 * temperature
        - 0.03 * humidity
        + material_effect
        + rng.normal(0, 0.5, rows)
    )

    return pd.DataFrame(
        {
            "temperature": temperature,
            "humidity": humidity,
            "material": material,
            "measurement": measurement,
        }
    )


def test_cross_validation_returns_three_models_and_all_folds():
    result = cross_validate_regression_models(
        make_frame(),
        "measurement",
        n_splits=5,
        random_seed=42,
        random_forest_estimators=20,
    )

    assert len(result.fold_metrics) == 15
    assert set(result.fold_metrics["model"]) == {
        "Linear Regression",
        "Random Forest",
        "Mean Baseline",
    }

    assert len(result.summary) == 3
    assert set(result.summary["model"]) == {
        "Linear Regression",
        "Random Forest",
        "Mean Baseline",
    }

    assert set(result.fold_metrics["fold"]) == {1, 2, 3, 4, 5}
    assert np.isfinite(
        result.fold_metrics[["mae", "rmse", "r2"]].to_numpy()
    ).all()


def test_cross_validation_is_reproducible():
    frame = make_frame()

    first = cross_validate_regression_models(
        frame,
        "measurement",
        random_seed=7,
        random_forest_estimators=20,
    )
    second = cross_validate_regression_models(
        frame,
        "measurement",
        random_seed=7,
        random_forest_estimators=20,
    )

    pd.testing.assert_frame_equal(
        first.fold_metrics,
        second.fold_metrics,
    )
    pd.testing.assert_frame_equal(
        first.summary,
        second.summary,
    )


def test_missing_target_rows_are_excluded():
    frame = make_frame()
    frame.loc[0, "measurement"] = np.nan

    result = cross_validate_regression_models(
        frame,
        "measurement",
        n_splits=5,
        random_forest_estimators=10,
    )

    assert result.rows_evaluated == len(frame) - 1


def test_excluded_columns_are_not_used_as_features():
    frame = make_frame()
    frame["record_id"] = np.arange(len(frame))

    result = cross_validate_regression_models(
        frame,
        "measurement",
        exclude_columns=["record_id"],
        random_forest_estimators=10,
    )

    assert "record_id" not in result.feature_columns


def test_missing_target_column_is_rejected():
    with pytest.raises(PreprocessingError, match="was not found"):
        cross_validate_regression_models(
            make_frame(),
            "unknown_target",
        )


def test_invalid_fold_count_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        cross_validate_regression_models(
            make_frame(),
            "measurement",
            n_splits=1,
        )


def test_too_few_rows_for_requested_folds_is_rejected():
    with pytest.raises(ValueError, match="two rows per validation fold"):
        cross_validate_regression_models(
            make_frame().head(8),
            "measurement",
            n_splits=5,
        )


def test_invalid_target_type_is_rejected():
    frame = make_frame()
    frame["measurement"] = frame["material"]

    with pytest.raises(PreprocessingError, match="numeric target"):
        cross_validate_regression_models(
            frame,
            "measurement",
        )
