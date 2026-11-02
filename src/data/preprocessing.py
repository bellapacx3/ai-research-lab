
"""Leakage-aware preprocessing for tabular machine learning."""

from dataclasses import dataclass
from math import ceil
from typing import Any, Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class PreprocessingError(ValueError):
    """Raised when a dataset cannot be prepared safely."""


@dataclass
class PreparedDataset:
    """Model-ready data and fitted preprocessing objects."""

    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    preprocessor: ColumnTransformer
    feature_columns: list[str]
    target_column: str
    task: str


def prepare_dataset(
    frame: pd.DataFrame,
    target_column: str,
    task: str = "regression",
    test_size: float = 0.2,
    random_seed: int = 42,
    exclude_columns: Sequence[str] | None = None,
) -> PreparedDataset:
    """Split data and fit preprocessing on training data only.

    Columns listed in exclude_columns are removed before the split.
    This is useful for identifiers that should not be model features.
    """
    if not isinstance(frame, pd.DataFrame):
        raise PreprocessingError("Input must be a pandas DataFrame.")

    if frame.empty:
        raise PreprocessingError("Dataset must not be empty.")

    if not frame.columns.is_unique:
        raise PreprocessingError(
            "Duplicate column names must be resolved before preprocessing."
        )

    if target_column not in frame.columns:
        raise PreprocessingError(
            f"Target column '{target_column}' was not found."
        )

    if task not in {"regression", "classification"}:
        raise PreprocessingError(
            "Task must be 'regression' or 'classification'."
        )

    if not isinstance(test_size, (int, float)) or isinstance(test_size, bool):
        raise PreprocessingError("test_size must be a number.")

    if not 0 < test_size < 1:
        raise PreprocessingError("test_size must be between 0 and 1.")

    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise PreprocessingError("random_seed must be an integer.")

    excluded = list(exclude_columns or [])

    if len(excluded) != len(set(excluded)):
        raise PreprocessingError(
            "exclude_columns must not contain duplicate column names."
        )

    if target_column in excluded:
        raise PreprocessingError(
            "The target column cannot be excluded from the dataset."
        )

    missing_exclusions = [
        column for column in excluded if column not in frame.columns
    ]
    if missing_exclusions:
        raise PreprocessingError(
            f"Excluded columns were not found: {missing_exclusions}"
        )

    data = frame.dropna(subset=[target_column]).copy()

    if len(data) < 5:
        raise PreprocessingError(
            "At least 5 rows with a non-missing target are required."
        )

    y = data[target_column]
    X = data.drop(columns=[target_column, *excluded])

    if X.shape[1] == 0:
        raise PreprocessingError(
            "At least one feature column is required besides the target."
        )

    if X.columns.duplicated().any():
        raise PreprocessingError("Feature column names must be unique.")

    if y.nunique() < 2:
        raise PreprocessingError(
            "The target must contain at least two distinct values."
        )

    if task == "regression" and not pd.api.types.is_numeric_dtype(y):
        raise PreprocessingError(
            "Regression requires a numeric target column."
        )

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = [
        column for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    stratify = None

    if task == "classification":
        class_counts = y.value_counts()

        if class_counts.min() >= 2:
            n_test = max(1, int(ceil(len(y) * test_size)))
            n_train = len(y) - n_test

            if (
                n_test >= y.nunique()
                and n_train >= y.nunique()
            ):
                stratify = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify,
        )

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

    except ValueError as exc:
        raise PreprocessingError(
            f"Could not prepare dataset: {exc}"
        ) from exc

    return PreparedDataset(
        X_train=X_train_transformed,
        X_test=X_test_transformed,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_columns=X.columns.tolist(),
        target_column=target_column,
        task=task,
    )
