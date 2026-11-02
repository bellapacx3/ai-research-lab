"""Save, load, and use trained scikit-learn model checkpoints."""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.storage.database import DEFAULT_DB_PATH, get_connection


def save_checkpoint(
    model,
    preprocessor,
    feature_columns,
    target_column,
    task,
    checkpoint_path,
    metrics=None,
    experiment_id=None,
    model_name=None,
    db_path=DEFAULT_DB_PATH,
):
    """Save a fitted model and its fitted preprocessing pipeline."""

    if task not in {"regression", "classification"}:
        raise ValueError("task must be regression or classification.")

    if not feature_columns:
        raise ValueError("feature_columns cannot be empty.")

    if not target_column:
        raise ValueError("target_column cannot be empty.")

    if (experiment_id is None) != (model_name is None):
        raise ValueError(
            "Provide both experiment_id and model_name to link a checkpoint."
        )

    destination = Path(checkpoint_path)
    if destination.suffix != ".joblib":
        raise ValueError("Checkpoint files must use the .joblib extension.")

    destination.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": list(feature_columns),
        "target_column": target_column,
        "task": task,
        "metrics": metrics or {},
        "sklearn_version": sklearn.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write to a temporary file first, so an interrupted save does not
    # leave a partially written checkpoint at the final path.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        joblib.dump(checkpoint, temp_path)
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    # Link the saved artifact to its model result in SQLite.
    if experiment_id is not None:
        with get_connection(db_path) as connection:
            cursor = connection.execute(
                """
                UPDATE model_results
                SET checkpoint_path = ?
                WHERE experiment_id = ? AND model_name = ?
                """,
                (str(destination.resolve()), experiment_id, model_name),
            )

            if cursor.rowcount != 1:
                raise ValueError(
                    "No matching model result found for this experiment "
                    "and model name. The checkpoint file was saved, "
                    "but the database link was not updated."
                )

    return destination.resolve()


def load_checkpoint(checkpoint_path):
    """Load a checkpoint from a trusted local file.

    Only load joblib files you created or otherwise trust. Loading a
    malicious serialized file can execute arbitrary code.
    """
    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.suffix != ".joblib":
        raise ValueError("Checkpoint files must use the .joblib extension.")

    checkpoint = joblib.load(path)

    required = {
        "model",
        "preprocessor",
        "feature_columns",
        "target_column",
        "task",
    }
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Checkpoint is missing required fields.")

    if checkpoint["task"] not in {"regression", "classification"}:
        raise ValueError("Checkpoint contains an unsupported task.")

    return checkpoint


def predict_with_checkpoint(checkpoint_path, frame):
    """Predict from raw feature columns using the saved preprocessing."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    checkpoint = load_checkpoint(checkpoint_path)
    feature_columns = checkpoint["feature_columns"]

    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Preserve the training-time feature order.
    features = frame.loc[:, feature_columns]
    transformed = checkpoint["preprocessor"].transform(features)
    predictions = checkpoint["model"].predict(transformed)

    return np.asarray(predictions)
