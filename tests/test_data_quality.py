
"""Tests for data-quality checks and feature exclusions."""

import pandas as pd
import pytest

from run_neural_network_experiment import find_exact_target_matches
from src.data.preprocessing import PreprocessingError, prepare_dataset


def make_dataset() -> pd.DataFrame:
    """Create a small dataset with an identifier and numeric features."""
    return pd.DataFrame(
        {
            "sample_id": [f"S{i:03d}" for i in range(10)],
            "temperature": [float(i + 20) for i in range(10)],
            "measurement": [float(i * 2 + 5) for i in range(10)],
        }
    )


def test_identifier_can_be_excluded_from_features():
    frame = make_dataset()

    prepared = prepare_dataset(
        frame,
        target_column="measurement",
        test_size=0.2,
        random_seed=42,
        exclude_columns=["sample_id"],
    )

    assert prepared.feature_columns == ["temperature"]


def test_missing_excluded_column_raises_error():
    frame = make_dataset()

    with pytest.raises(
        PreprocessingError,
        match="Excluded columns were not found",
    ):
        prepare_dataset(
            frame,
            target_column="measurement",
            exclude_columns=["unknown_id"],
        )


def test_target_cannot_be_excluded():
    frame = make_dataset()

    with pytest.raises(
        PreprocessingError,
        match="target column cannot be excluded",
    ):
        prepare_dataset(
            frame,
            target_column="measurement",
            exclude_columns=["measurement"],
        )


def test_exact_target_copy_is_detected():
    frame = make_dataset()
    frame["leaked_measurement"] = frame["measurement"]

    matches = find_exact_target_matches(frame, "measurement")

    assert matches == ["leaked_measurement"]


def test_unrelated_feature_is_not_flagged_as_exact_target_match():
    frame = make_dataset()

    matches = find_exact_target_matches(frame, "measurement")

    assert matches == []
