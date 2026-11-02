
"""Leakage-aware cross-validation for tabular regression experiments."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.preprocessing import PreprocessingError


@dataclass
class CrossValidationResult:
    """Fold-level metrics and aggregate cross-validation statistics."""

    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    target_column: str
    feature_columns: list[str]
    n_splits: int
    random_seed: int
    rows_evaluated: int


def _build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build a fresh preprocessing pipeline for the supplied fold."""

    numeric_columns = features.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in features.columns
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

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def _regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics for one fold."""

    predicted = np.asarray(predicted, dtype=float).reshape(-1)

    if len(predicted) != len(actual):
        raise ValueError(
            "Prediction count does not match validation target count."
        )

    if not np.isfinite(predicted).all():
        raise ValueError("Model produced non-finite predictions.")

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": (
            float(r2_score(actual, predicted))
            if len(actual) >= 2
            else float("nan")
        ),
    }


def cross_validate_regression_models(
    frame: pd.DataFrame,
    target_column: str,
    *,
    exclude_columns: Sequence[str] | None = None,
    n_splits: int = 5,
    random_seed: int = 42,
    random_forest_estimators: int = 200,
) -> CrossValidationResult:
    """Evaluate three regression models using shuffled K-fold CV.

    Preprocessing is fitted independently on each fold's training data.
    Each validation fold remains unseen during preprocessing and fitting.

    The function returns fold-level metrics and their mean and standard
    deviation. It does not fit or evaluate a separate final test set.
    """

    if not isinstance(frame, pd.DataFrame):
        raise PreprocessingError("Input must be a pandas DataFrame.")

    if frame.empty:
        raise PreprocessingError("Dataset must not be empty.")

    if not frame.columns.is_unique:
        raise PreprocessingError(
            "Duplicate column names must be resolved before cross-validation."
        )

    if target_column not in frame.columns:
        raise PreprocessingError(
            f"Target column '{target_column}' was not found."
        )

    if (
        not isinstance(n_splits, int)
        or isinstance(n_splits, bool)
        or n_splits < 2
    ):
        raise ValueError("n_splits must be an integer of at least 2.")

    if (
        not isinstance(random_seed, int)
        or isinstance(random_seed, bool)
    ):
        raise TypeError("random_seed must be an integer.")

    if (
        not isinstance(random_forest_estimators, int)
        or isinstance(random_forest_estimators, bool)
        or random_forest_estimators < 1
    ):
        raise ValueError(
            "random_forest_estimators must be an integer of at least 1."
        )

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

    # Remove rows without targets, matching prepare_dataset behavior.
    data = frame.dropna(subset=[target_column]).copy()

    if not pd.api.types.is_numeric_dtype(data[target_column]):
        raise PreprocessingError(
            "Regression requires a numeric target column."
        )

    y = data[target_column].to_numpy(dtype=float)

    if not np.isfinite(y).all():
        raise PreprocessingError(
            "Regression targets must contain only finite values."
        )

    if len(np.unique(y)) < 2:
        raise PreprocessingError(
            "The target must contain at least two distinct values."
        )

    X = data.drop(columns=[target_column, *excluded])

    if X.shape[1] == 0:
        raise PreprocessingError(
            "At least one feature column is required besides the target."
        )

    if len(data) < n_splits * 2:
        raise ValueError(
            "At least two rows per validation fold are required; "
            f"received {len(data)} rows for {n_splits} folds."
        )

    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )

    fold_records = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_train = X.iloc[train_indices]
        X_validation = X.iloc[validation_indices]

        y_train = y[train_indices]
        y_validation = y[validation_indices]

        # Critical: fit a NEW preprocessor on this fold's training rows only.
        preprocessor = _build_preprocessor(X_train)

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_validation_transformed = preprocessor.transform(X_validation)

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=random_forest_estimators,
                random_state=random_seed,
                n_jobs=-1,
            ),
            "Mean Baseline": DummyRegressor(strategy="mean"),
        }

        for model_name, model in models.items():
            model.fit(X_train_transformed, y_train)
            predictions = model.predict(X_validation_transformed)

            metrics = _regression_metrics(
                y_validation,
                predictions,
            )

            fold_records.append(
                {
                    "fold": fold_number,
                    "model": model_name,
                    "train_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    **metrics,
                }
            )

    fold_metrics = pd.DataFrame(fold_records)

    summary = (
        fold_metrics.groupby("model", sort=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .reset_index()
        .sort_values("rmse_mean")
        .reset_index(drop=True)
    )

    return CrossValidationResult(
        fold_metrics=fold_metrics,
        summary=summary,
        target_column=target_column,
        feature_columns=X.columns.tolist(),
        n_splits=n_splits,
        random_seed=random_seed,
        rows_evaluated=len(data),
    )
