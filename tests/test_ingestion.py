import pandas as pd
import pytest

from src.data.ingestion import (
    DatasetError,
    load_dataset,
    profile_dataframe,
    validate_dataframe,
)


def test_load_sample_csv():
    frame = load_dataset("data/samples/sample_research.csv")

    assert frame.shape == (8, 5)
    assert "temperature" in frame.columns


def test_validation_detects_missing_and_duplicate_rows():
    frame = load_dataset("data/samples/sample_research.csv")
    result = validate_dataframe(frame)

    assert result["missing_cells"] == 1
    assert result["duplicate_rows"] == 1
    assert result["row_count"] == 8
    assert result["column_count"] == 5


def test_duplicate_column_names_are_reported():
    frame = pd.DataFrame([[1, 2]], columns=["measurement", "measurement"])

    result = validate_dataframe(frame)

    assert result["duplicate_columns"] == ["measurement"]


def test_empty_columns_are_reported():
    frame = pd.DataFrame({"empty": [None, None], "value": [1, 2]})

    result = validate_dataframe(frame)

    assert result["empty_columns"] == ["empty"]
    assert result["constant_columns"] == ["empty"]


def test_profile_contains_numeric_and_categorical_columns():
    frame = load_dataset("data/samples/sample_research.csv")
    result = profile_dataframe(frame)

    assert "temperature" in result["numeric_columns"]
    assert "material" in result["categorical_columns"]
    assert result["unique_values_by_column"]["material"] == 3


def test_missing_file_raises_clear_error():
    with pytest.raises(DatasetError, match="File not found"):
        load_dataset("data/samples/does_not_exist.csv")


def test_unsupported_format_raises_clear_error(tmp_path):
    path = tmp_path / "dataset.txt"
    path.write_text("example", encoding="utf-8")

    with pytest.raises(DatasetError, match="Unsupported format"):
        load_dataset(path)


def test_profile_preserves_duplicate_column_names():
    frame = pd.DataFrame(
        [[1, 2], [3, 4]],
        columns=["measurement", "measurement"],
    )

    result = profile_dataframe(frame)

    assert result["duplicate_columns"] == ["measurement"]
    assert result["numeric_columns"] == [
        "measurement",
        "measurement [2]",
    ]
    assert "measurement" in result["numeric_summary"]
    assert "measurement [2]" in result["numeric_summary"]
