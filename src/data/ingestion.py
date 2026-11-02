"""Dataset loading and validation for AI Research Lab."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls", ".parquet"}


class DatasetError(ValueError):
    """Raised when a dataset cannot be loaded or interpreted."""


def _read_csv(path: Path) -> pd.DataFrame:
    """Read CSV while retaining original column names, including duplicates."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            header = next(csv.reader(file), None)

        if not header:
            raise DatasetError("The CSV file is empty or has no header.")

        frame = pd.read_csv(path, encoding="utf-8-sig")

        if len(header) != len(frame.columns):
            raise DatasetError(
                "The CSV header does not match the parsed column count."
            )

        frame.columns = header
        return frame
    except UnicodeDecodeError as exc:
        raise DatasetError(
            "Could not decode the CSV as UTF-8. Convert it to UTF-8 and retry."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise DatasetError("The CSV file contains no readable data.") from exc
    except pd.errors.ParserError as exc:
        raise DatasetError(f"Could not parse CSV: {exc}") from exc


def _read_excel(path: Path) -> pd.DataFrame:
    """Read the first Excel worksheet while retaining duplicate headers."""
    try:
        raw = pd.read_excel(path, header=None)
    except ImportError as exc:
        raise DatasetError(
            "Excel support is missing. Install openpyxl and retry."
        ) from exc

    if raw.empty:
        raise DatasetError("The Excel worksheet is empty.")

    headers = raw.iloc[0].tolist()
    frame = raw.iloc[1:].reset_index(drop=True)
    frame.columns = headers
    return frame


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV, Excel, or Parquet dataset from a local path."""
    path = Path(file_path).expanduser()

    if not path.exists():
        raise DatasetError(f"File not found: {path}")

    if not path.is_file():
        raise DatasetError(f"Path is not a file: {path}")

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_FORMATS:
        raise DatasetError(
            f"Unsupported format '{suffix}'. Supported: "
            f"{', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    try:
        if suffix == ".csv":
            frame = _read_csv(path)
        elif suffix in {".xlsx", ".xls"}:
            frame = _read_excel(path)
        else:
            frame = pd.read_parquet(path)
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"Failed to load '{path.name}': {exc}") from exc

    if frame.empty:
        raise DatasetError("The dataset has no data rows.")

    if frame.shape[1] == 0:
        raise DatasetError("The dataset has no columns.")

    return frame




def _column_labels(frame: pd.DataFrame) -> list[str]:
    """Return unique display labels while preserving original column names."""
    labels = []
    counts = {}

    for name in frame.columns:
        base = str(name)
        counts[base] = counts.get(base, 0) + 1
        occurrence = counts[base]
        labels.append(base if occurrence == 1 else f"{base} [{occurrence}]")

    return labels


def validate_dataframe(frame: pd.DataFrame) -> dict:
    """Return data-quality findings without silently modifying the data."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    labels = _column_labels(frame)
    missing_counts = frame.isna().sum().tolist()

    missing_by_column = {
        labels[index]: int(count)
        for index, count in enumerate(missing_counts)
        if count > 0
    }

    duplicate_column_names = frame.columns.duplicated(keep=False)
    duplicate_columns = list(
        dict.fromkeys(
            str(name)
            for name, is_duplicate in zip(
                frame.columns, duplicate_column_names
            )
            if is_duplicate
        )
    )

    constant_columns = []
    empty_columns = []

    for index, label in enumerate(labels):
        column = frame.iloc[:, index]

        if column.isna().all():
            empty_columns.append(label)

        if column.nunique(dropna=False) <= 1:
            constant_columns.append(label)

    return {
        "row_count": int(frame.shape[0]),
        "column_count": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_columns": duplicate_columns,
        "missing_cells": int(sum(missing_counts)),
        "missing_by_column": missing_by_column,
        "constant_columns": constant_columns,
        "empty_columns": empty_columns,
    }


def profile_dataframe(frame: pd.DataFrame) -> dict:
    """Create a compact, machine-readable dataset profile."""
    validation = validate_dataframe(frame)
    labels = _column_labels(frame)

    numeric_columns = []
    categorical_columns = []
    datetime_columns = []
    numeric_summary = {}
    unique_values = {}

    for index, label in enumerate(labels):
        column = frame.iloc[:, index]
        dtype = column.dtype

        if pd.api.types.is_numeric_dtype(dtype):
            numeric_columns.append(label)
            if not pd.api.types.is_bool_dtype(dtype):
                numeric_summary[label] = column.describe().to_dict()
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_columns.append(label)
        else:
            categorical_columns.append(label)

        unique_values[label] = int(column.nunique(dropna=True))

    return {
        **validation,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "numeric_summary": numeric_summary,
        "unique_values_by_column": unique_values,
    }
