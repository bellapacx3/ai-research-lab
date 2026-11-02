"""Persistence operations for research experiments."""

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from src.storage.database import (
    DEFAULT_DB_PATH,
    get_connection,
    initialize_database,
)


def _json_safe(value: Any) -> Any:
    """Convert common scientific Python values into JSON-safe values."""
    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    raise TypeError(f"Unsupported metadata value: {type(value).__name__}")


def save_experiment(
    metadata: dict,
    model_results: dict,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Save one experiment and its model metrics; return its database ID."""
    required = {
        "experiment_name",
        "dataset_path",
        "dataset_sha256",
        "random_seed",
        "parameters",
        "environment",
    }

    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"Missing metadata fields: {sorted(missing)}")

    if not model_results:
        raise ValueError("At least one model result is required.")

    for name, result in model_results.items():
        metrics = result.metrics
        for metric in ("mae", "rmse"):
            value = metrics.get(metric)
            if value is None or not math.isfinite(float(value)):
                raise ValueError(
                    f"Model '{name}' has an invalid {metric} metric."
                )

    initialize_database(db_path)

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO experiments (
                experiment_name,
                dataset_path,
                dataset_sha256,
                random_seed,
                parameters_json,
                environment_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                metadata["experiment_name"],
                metadata["dataset_path"],
                metadata["dataset_sha256"],
                metadata["random_seed"],
                json.dumps(_json_safe(metadata["parameters"])),
                json.dumps(_json_safe(metadata["environment"])),
            ),
        )

        experiment_id = cursor.lastrowid

        connection.executemany(
            """
            INSERT INTO model_results (
                experiment_id, model_name, mae, rmse, r2
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    experiment_id,
                    name,
                    float(result.metrics["mae"]),
                    float(result.metrics["rmse"]),
                    (
                        float(result.metrics["r2"])
                        if result.metrics.get("r2") is not None
                        and math.isfinite(float(result.metrics["r2"]))
                        else None
                    ),
                )
                for name, result in model_results.items()
            ],
        )

    return int(experiment_id)


def list_experiments(
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 50,
) -> list[dict]:
    """Return the latest experiment records with their model metrics."""
    if limit < 1:
        raise ValueError("limit must be at least 1.")

    initialize_database(db_path)

    with get_connection(db_path) as connection:
        experiments = connection.execute(
            """
            SELECT *
            FROM experiments
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        output = []

        for experiment in experiments:
            item = dict(experiment)

            item["parameters"] = json.loads(item.pop("parameters_json"))
            item["environment"] = json.loads(item.pop("environment_json"))

            results = connection.execute(
                """
                SELECT model_name, mae, rmse, r2, checkpoint_path
                FROM model_results
                WHERE experiment_id = ?
                ORDER BY rmse ASC
                """,
                (item["id"],),
            ).fetchall()

            item["model_results"] = [dict(row) for row in results]
            output.append(item)

    return output


def get_experiment(
    experiment_id: int,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict | None:
    """Retrieve one experiment by ID, or return None if absent."""
    if experiment_id < 1:
        raise ValueError("experiment_id must be positive.")

    experiments = list_experiments(db_path, limit=2**31 - 1)

    return next(
        (item for item in experiments if item["id"] == experiment_id),
        None,
    )


def save_cross_validation_experiment(
    metadata: dict,
    cv_result: Any,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Persist cross-validation summaries and fold metrics atomically."""
    required = {
        "experiment_name",
        "dataset_path",
        "dataset_sha256",
        "random_seed",
        "parameters",
        "environment",
    }
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"Missing metadata fields: {sorted(missing)}")

    fold_metrics = cv_result.fold_metrics
    summary = cv_result.summary

    if fold_metrics.empty or summary.empty:
        raise ValueError("Cross-validation results must not be empty.")

    required_fold_columns = {
        "fold", "model", "train_rows", "validation_rows",
        "mae", "rmse", "r2",
    }
    required_summary_columns = {
        "model", "mae_mean", "mae_std",
        "rmse_mean", "rmse_std", "r2_mean", "r2_std",
    }

    if not required_fold_columns.issubset(fold_metrics.columns):
        raise ValueError("Fold metrics are missing required columns.")

    if not required_summary_columns.issubset(summary.columns):
        raise ValueError("Summary metrics are missing required columns.")

    def finite_or_none(value):
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    # Validate mandatory metrics before opening the transaction.
    for _, row in fold_metrics.iterrows():
        for metric in ("mae", "rmse"):
            value = finite_or_none(row[metric])
            if value is None:
                raise ValueError(
                    f"Fold {row['fold']} has invalid {metric}."
                )
            if value < 0:
                raise ValueError(f"{metric} cannot be negative.")

        for count in ("fold", "train_rows", "validation_rows"):
            value = row[count]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or int(value) != value
                or value < 1
            ):
                raise ValueError(f"Invalid fold field: {count}.")

    for _, row in summary.iterrows():
        for metric in ("mae_mean", "rmse_mean"):
            value = finite_or_none(row[metric])
            if value is None or value < 0:
                raise ValueError(
                    f"Model '{row['model']}' has invalid {metric}."
                )

    initialize_database(db_path)

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO experiments (
                experiment_name,
                dataset_path,
                dataset_sha256,
                random_seed,
                parameters_json,
                environment_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                metadata["experiment_name"],
                metadata["dataset_path"],
                metadata["dataset_sha256"],
                int(metadata["random_seed"]),
                json.dumps(_json_safe(metadata["parameters"])),
                json.dumps(_json_safe(metadata["environment"])),
            ),
        )
        experiment_id = int(cursor.lastrowid)

        connection.executemany(
            """
            INSERT INTO cross_validation_results (
                experiment_id, model_name,
                mae_mean, mae_std,
                rmse_mean, rmse_std,
                r2_mean, r2_std
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    experiment_id,
                    str(row["model"]),
                    float(row["mae_mean"]),
                    finite_or_none(row["mae_std"]),
                    float(row["rmse_mean"]),
                    finite_or_none(row["rmse_std"]),
                    finite_or_none(row["r2_mean"]),
                    finite_or_none(row["r2_std"]),
                )
                for _, row in summary.iterrows()
            ],
        )

        connection.executemany(
            """
            INSERT INTO cross_validation_folds (
                experiment_id, model_name, fold,
                train_rows, validation_rows,
                mae, rmse, r2
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    experiment_id,
                    str(row["model"]),
                    int(row["fold"]),
                    int(row["train_rows"]),
                    int(row["validation_rows"]),
                    float(row["mae"]),
                    float(row["rmse"]),
                    finite_or_none(row["r2"]),
                )
                for _, row in fold_metrics.iterrows()
            ],
        )

    return experiment_id


def get_cross_validation_results(
    experiment_id: int,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict | None:
    """Retrieve aggregate and fold-level cross-validation results."""
    if (
        not isinstance(experiment_id, int)
        or isinstance(experiment_id, bool)
        or experiment_id < 1
    ):
        raise ValueError("experiment_id must be a positive integer.")

    initialize_database(db_path)

    with get_connection(db_path) as connection:
        experiment = connection.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()

        if experiment is None:
            return None

        summaries = connection.execute(
            """
            SELECT model_name, mae_mean, mae_std,
                   rmse_mean, rmse_std, r2_mean, r2_std
            FROM cross_validation_results
            WHERE experiment_id = ?
            ORDER BY rmse_mean ASC
            """,
            (experiment_id,),
        ).fetchall()

        folds = connection.execute(
            """
            SELECT model_name, fold, train_rows, validation_rows,
                   mae, rmse, r2
            FROM cross_validation_folds
            WHERE experiment_id = ?
            ORDER BY fold ASC, model_name ASC
            """,
            (experiment_id,),
        ).fetchall()

    return {
        "experiment": dict(experiment),
        "summary": [dict(row) for row in summaries],
        "folds": [dict(row) for row in folds],
    }


def save_model_predictions(
    experiment_id: int,
    model_name: str,
    sample_indices,
    actual_values,
    predicted_values,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Save per-sample predictions for an evaluated model.

    Returns the number of saved prediction records.
    """
    if (
        not isinstance(experiment_id, int)
        or isinstance(experiment_id, bool)
        or experiment_id < 1
    ):
        raise ValueError("experiment_id must be a positive integer.")

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")

    indices = list(sample_indices)
    actuals = list(actual_values)
    predictions = list(predicted_values)

    if not indices:
        raise ValueError("At least one prediction is required.")

    if len(indices) != len(actuals) or len(indices) != len(predictions):
        raise ValueError(
            "Sample indices, actual values, and predictions "
            "must have equal lengths."
        )

    normalized_indices = []

    for index in indices:
        if index is None:
            raise ValueError("Sample indices cannot contain None.")

        normalized_indices.append(str(index))

    if len(set(normalized_indices)) != len(normalized_indices):
        raise ValueError("Duplicate sample indices are not allowed.")

    records = []

    for index, actual, predicted in zip(
        normalized_indices,
        actuals,
        predictions,
    ):
        if isinstance(actual, bool) or isinstance(predicted, bool):
            raise ValueError("Prediction values must be numeric.")

        try:
            actual = float(actual)
            predicted = float(predicted)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Actual and predicted values must be numeric."
            ) from exc

        if not math.isfinite(actual) or not math.isfinite(predicted):
            raise ValueError(
                "Actual and predicted values must be finite."
            )

        residual = actual - predicted
        absolute_error = abs(residual)
        squared_error = residual ** 2

        if not all(
            math.isfinite(value)
            for value in (residual, absolute_error, squared_error)
        ):
            raise ValueError("Calculated prediction errors are not finite.")

        records.append(
            (
                experiment_id,
                model_name.strip(),
                index,
                actual,
                predicted,
                residual,
                absolute_error,
                squared_error,
            )
        )

    initialize_database(db_path)

    with get_connection(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO model_predictions (
                experiment_id,
                model_name,
                sample_index,
                actual_value,
                predicted_value,
                residual,
                absolute_error,
                squared_error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )

    return len(records)


def get_model_predictions(
    experiment_id: int,
    model_name: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict]:
    """Retrieve saved predictions for an experiment.

    If model_name is supplied, only that model's predictions are returned.
    """
    if (
        not isinstance(experiment_id, int)
        or isinstance(experiment_id, bool)
        or experiment_id < 1
    ):
        raise ValueError("experiment_id must be a positive integer.")

    if model_name is not None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

    initialize_database(db_path)

    query = """
        SELECT
            experiment_id,
            model_name,
            sample_index,
            actual_value,
            predicted_value,
            residual,
            absolute_error,
            squared_error
        FROM model_predictions
        WHERE experiment_id = ?
    """

    parameters = [experiment_id]

    if model_name is not None:
        query += " AND model_name = ?"
        parameters.append(model_name.strip())

    query += " ORDER BY model_name ASC, sample_index ASC"

    with get_connection(db_path) as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [dict(row) for row in rows]
