
"""Train and evaluate baseline classical machine-learning models."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.data.preprocessing import PreparedDataset, PreprocessingError


@dataclass
class RegressionResult:
    """Evaluation results for one fitted regression model."""

    model_name: str
    model: Any
    predictions: np.ndarray
    metrics: dict[str, float]


def _validate_prepared_regression_data(
    prepared: PreparedDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate regression data and return finite target arrays."""

    if prepared.task != "regression":
        raise PreprocessingError(
            "Regression models require a dataset prepared for regression."
        )

    y_train = np.asarray(prepared.y_train, dtype=float).reshape(-1)
    y_test = np.asarray(prepared.y_test, dtype=float).reshape(-1)

    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("Training and test targets must be non-empty.")

    if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
        raise ValueError("Regression targets must contain only finite values.")

    train_rows = prepared.X_train.shape[0]
    test_rows = prepared.X_test.shape[0]

    if train_rows != len(y_train):
        raise ValueError("Training features and targets must have equal lengths.")

    if test_rows != len(y_test):
        raise ValueError("Test features and targets must have equal lengths.")

    return y_train, y_test


def _build_regression_result(
    *,
    name: str,
    model: Any,
    X_test: Any,
    y_test: np.ndarray,
) -> RegressionResult:
    """Fit a model, validate predictions, and calculate evaluation metrics."""

  
    # Fitting is performed by the caller. This helper evaluates its predictions.
    predictions = np.asarray(model.predict(X_test), dtype=float).reshape(-1)

    if len(predictions) != len(y_test):
        raise ValueError(
            f"Model '{name}' returned {len(predictions)} predictions "
            f"for {len(y_test)} test samples."
        )

    if not np.isfinite(predictions).all():
        raise ValueError(
            f"Model '{name}' produced non-finite predictions."
        )

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": (
            float(r2_score(y_test, predictions))
            if len(y_test) >= 2
            else float("nan")
        ),
    }

    return RegressionResult(
        model_name=name,
        model=model,
        predictions=predictions,
        metrics=metrics,
    )


def evaluate_regression_models(
    prepared: PreparedDataset,
    random_seed: int = 42,
    random_forest_estimators: int = 200,
) -> dict[str, RegressionResult]:
    """Train and evaluate Linear Regression and Random Forest."""

    if random_forest_estimators < 1:
        raise ValueError("random_forest_estimators must be at least 1.")

    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise TypeError("random_seed must be an integer.")

    y_train, y_test = _validate_prepared_regression_data(prepared)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=random_forest_estimators,
            random_state=random_seed,
            n_jobs=-1,
        ),
    }

    results = {}

    for name, model in models.items():
        model.fit(prepared.X_train, y_train)
        results[name] = _build_regression_result(
            name=name,
            model=model,
            X_test=prepared.X_test,
            y_test=y_test,
        )

    return results


def evaluate_regression_baseline(
    prepared: PreparedDataset,
) -> RegressionResult:
    """Evaluate a mean-prediction baseline on the held-out test set."""

    y_train, y_test = _validate_prepared_regression_data(prepared)

    model = DummyRegressor(strategy="mean")
    model.fit(prepared.X_train, y_train)

    return _build_regression_result(
        name="Mean Baseline",
        model=model,
        X_test=prepared.X_test,
        y_test=y_test,
    )
