
"""Statistical residual analysis for persisted regression predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.storage.repositories import get_model_predictions


def analyze_residuals(predictions: list[dict]) -> dict:
    """Calculate statistical metrics from persisted predictions."""

    if not predictions:
        raise ValueError("At least one prediction is required.")

    frame = pd.DataFrame(predictions)

    required_columns = {
        "actual_value",
        "predicted_value",
        "residual",
        "absolute_error",
        "squared_error",
    }

    missing = required_columns.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Prediction records are missing columns: {sorted(missing)}"
        )

    numeric = frame[list(required_columns)].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError("Prediction records contain non-finite values.")

    actual = frame["actual_value"].to_numpy(dtype=float)
    predicted = frame["predicted_value"].to_numpy(dtype=float)
    residuals = actual - predicted
    absolute_errors = np.abs(residuals)
    squared_errors = residuals ** 2

    if not np.allclose(
        residuals,
        frame["residual"].to_numpy(dtype=float),
    ):
        raise ValueError("Stored residuals do not match actual predictions.")

    if not np.allclose(
        absolute_errors,
        frame["absolute_error"].to_numpy(dtype=float),
    ):
        raise ValueError("Stored absolute errors are inconsistent.")

    if not np.allclose(
        squared_errors,
        frame["squared_error"].to_numpy(dtype=float),
    ):
        raise ValueError("Stored squared errors are inconsistent.")

    percentiles = np.percentile(
        absolute_errors,
        [50, 75, 90, 95, 99],
    )

    overpredictions = int(np.sum(residuals < 0))
    underpredictions = int(np.sum(residuals > 0))
    exact_predictions = int(np.sum(residuals == 0))

    return {
        "sample_count": len(residuals),
        "mean_residual": float(np.mean(residuals)),
        "median_residual": float(np.median(residuals)),
        "residual_std": float(np.std(residuals, ddof=1))
        if len(residuals) > 1 else 0.0,
        "mae": float(np.mean(absolute_errors)),
        "rmse": float(np.sqrt(np.mean(squared_errors))),
        "mean_absolute_error_percentiles": {
            "50th": float(percentiles[0]),
            "75th": float(percentiles[1]),
            "90th": float(percentiles[2]),
            "95th": float(percentiles[3]),
            "99th": float(percentiles[4]),
        },
        "overpredictions": overpredictions,
        "underpredictions": underpredictions,
        "exact_predictions": exact_predictions,
        "overprediction_percentage": float(
            overpredictions / len(residuals) * 100
        ),
        "underprediction_percentage": float(
            underpredictions / len(residuals) * 100
        ),
    }


def analyze_experiment_residuals(
    experiment_id: int,
    db_path=None,
) -> dict:
    """Analyze every model's saved predictions for an experiment."""

    if (
        not isinstance(experiment_id, int)
        or isinstance(experiment_id, bool)
        or experiment_id < 1
    ):
        raise ValueError("experiment_id must be a positive integer.")

    from src.storage.repositories import DEFAULT_DB_PATH

    database_path = db_path if db_path is not None else DEFAULT_DB_PATH

    predictions = get_model_predictions(
        experiment_id=experiment_id,
        db_path=database_path,
    )

    if not predictions:
        raise ValueError(
            f"No saved predictions found for experiment {experiment_id}."
        )

    frame = pd.DataFrame(predictions)
    results = {}

    for model_name, group in frame.groupby("model_name", sort=True):
        results[model_name] = analyze_residuals(
            group.to_dict(orient="records")
        )

    return {
        "experiment_id": experiment_id,
        "models": results,
    }
