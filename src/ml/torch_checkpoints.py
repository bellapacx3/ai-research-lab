"""Save, load, and use trusted local PyTorch research checkpoints."""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.ml.neural_network import ResearchNeuralNetwork
from src.storage.database import DEFAULT_DB_PATH, get_connection


def _architecture_from_model(model: nn.Module) -> dict:
    """Extract the dimensions needed to reconstruct the network."""
    linear_layers = [
        layer for layer in model.modules()
        if isinstance(layer, nn.Linear)
    ]

    if not linear_layers:
        raise ValueError("The model does not contain any Linear layers.")

    hidden_sizes = tuple(
        layer.out_features for layer in linear_layers[:-1]
    )

    return {
        "input_features": linear_layers[0].in_features,
        "hidden_sizes": hidden_sizes,
        "output_features": linear_layers[-1].out_features,
    }


def save_torch_checkpoint(
    model: nn.Module,
    preprocessor,
    feature_columns: list[str],
    target_column: str,
    target_mean: float,
    target_scale: float,
    checkpoint_path: str | Path,
    metrics: dict | None = None,
    experiment_id: int | None = None,
    model_name: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> Path:
    """Save a PyTorch state dict and the metadata required for prediction."""
    if not isinstance(model, ResearchNeuralNetwork):
        raise TypeError("model must be a ResearchNeuralNetwork.")

    if not feature_columns:
        raise ValueError("feature_columns cannot be empty.")

    if not target_column:
        raise ValueError("target_column cannot be empty.")

    if not np.isfinite(target_mean):
        raise ValueError("target_mean must be finite.")

    if not np.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("target_scale must be positive and finite.")

    if (experiment_id is None) != (model_name is None):
        raise ValueError(
            "Provide both experiment_id and model_name to link a checkpoint."
        )

    destination = Path(checkpoint_path)

    if destination.suffix != ".pt":
        raise ValueError("PyTorch checkpoints must use the .pt extension.")

    destination.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "format_version": 1,
        "state_dict": model.state_dict(),
        "architecture": _architecture_from_model(model),
        "preprocessor": preprocessor,
        "feature_columns": list(feature_columns),
        "target_column": target_column,
        "task": "regression",
        "target_mean": float(target_mean),
        "target_scale": float(target_scale),
        "metrics": metrics or {},
        "torch_version": torch.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, destination)

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    if experiment_id is not None:
        with get_connection(db_path) as connection:
            cursor = connection.execute(
                """
                UPDATE model_results
                SET checkpoint_path = ?
                WHERE experiment_id = ? AND model_name = ?
                """,
                (
                    str(destination.resolve()),
                    experiment_id,
                    model_name,
                ),
            )

            if cursor.rowcount != 1:
                raise ValueError(
                    "Checkpoint saved, but no matching model result "
                    "was found to link it in the database."
                )

    return destination.resolve()


def load_torch_checkpoint(
    checkpoint_path: str | Path,
) -> dict:
    """Load a trusted local PyTorch checkpoint.

    The checkpoint includes a serialized preprocessing object. Loading it
    uses pickle-based deserialization and must only be done for trusted files.
    """
    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.suffix != ".pt":
        raise ValueError("PyTorch checkpoints must use the .pt extension.")

    # weights_only=False is required because the checkpoint also contains
    # a fitted scikit-learn preprocessing object. Only load trusted files.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "format_version",
        "state_dict",
        "architecture",
        "preprocessor",
        "feature_columns",
        "target_column",
        "task",
        "target_mean",
        "target_scale",
    }

    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Checkpoint is missing required fields.")

    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported PyTorch checkpoint format.")

    if checkpoint["task"] != "regression":
        raise ValueError("Unsupported checkpoint task.")

    architecture = checkpoint["architecture"]

    model = ResearchNeuralNetwork(
        input_features=architecture["input_features"],
        hidden_sizes=tuple(architecture["hidden_sizes"]),
        output_features=architecture["output_features"],
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    checkpoint["model"] = model

    return checkpoint


def predict_with_torch_checkpoint(
    checkpoint_path: str | Path,
    frame: pd.DataFrame,
) -> np.ndarray:
    """Predict from raw features using a saved PyTorch checkpoint."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    checkpoint = load_torch_checkpoint(checkpoint_path)

    feature_columns = checkpoint["feature_columns"]

    missing = [
        column for column in feature_columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    from src.ml.neural_network import predict_neural_network

    return predict_neural_network(
        model=checkpoint["model"],
        preprocessor=checkpoint["preprocessor"],
        raw_features=frame,
        feature_columns=feature_columns,
        target_mean=checkpoint["target_mean"],
        target_scale=checkpoint["target_scale"],
    )
