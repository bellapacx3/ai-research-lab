"""Run and persist train/test regression experiments."""

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import prepare_dataset
from src.experiments.reproducibility import build_experiment_metadata
from src.ml.classical_models import (
    evaluate_regression_baseline,
    evaluate_regression_models,
)
from src.ml.checkpoints import save_checkpoint
from src.ml.neural_network import train_neural_network
from src.ml.torch_checkpoints import save_torch_checkpoint
from src.storage.database import DEFAULT_DB_PATH
from src.storage.repositories import (
    save_experiment,
    save_model_predictions,
)


SUPPORTED_MODELS = {
    "Linear Regression",
    "Random Forest",
    "Mean Baseline",
    "PyTorch Neural Network",
}


def _safe_filename(value: str) -> str:
    """Convert a model name to a filesystem-safe filename."""
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return result or "model"


def run_regression_experiment(
    frame: pd.DataFrame,
    dataset_path: str | Path,
    target_column: str,
    feature_columns: Sequence[str],
    experiment_name: str,
    random_seed: int = 42,
    test_size: float = 0.2,
    selected_models: Sequence[str] = (
        "Linear Regression",
        "Random Forest",
        "Mean Baseline",
        "PyTorch Neural Network",
    ),
    random_forest_estimators: int = 200,
    nn_hidden_sizes: tuple[int, ...] = (32, 16),
    nn_epochs: int = 200,
    nn_learning_rate: float = 0.001,
    nn_batch_size: int = 16,
    nn_patience: int = 25,
    checkpoint_dir: str | Path = "artifacts/checkpoints",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict:
    """Train selected regression models and persist their results.

    Predictions are saved with the original test-set DataFrame index values.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    if frame.empty:
        raise ValueError("The dataset must not be empty.")

    if not isinstance(target_column, str) or target_column not in frame.columns:
        raise ValueError("A valid target column must be selected.")

    features = list(feature_columns)

    if not features:
        raise ValueError("Select at least one feature column.")

    if len(features) != len(set(features)):
        raise ValueError("Feature columns must not contain duplicates.")

    if target_column in features:
        raise ValueError("The target column cannot also be a feature.")

    missing_features = [
        column for column in features if column not in frame.columns
    ]

    if missing_features:
        raise ValueError(f"Feature columns not found: {missing_features}")

    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("Experiment name must not be empty.")

    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise TypeError("random_seed must be an integer.")

    if not isinstance(selected_models, (list, tuple)):
        raise TypeError("selected_models must be a list or tuple.")

    models_to_run = list(selected_models)

    if not models_to_run:
        raise ValueError("Select at least one model.")

    if len(models_to_run) != len(set(models_to_run)):
        raise ValueError("Selected models must not contain duplicates.")

    unsupported = set(models_to_run) - SUPPORTED_MODELS

    if unsupported:
        raise ValueError(f"Unsupported models selected: {sorted(unsupported)}")

    if (
        not isinstance(random_forest_estimators, int)
        or isinstance(random_forest_estimators, bool)
        or random_forest_estimators < 1
    ):
        raise ValueError("random_forest_estimators must be at least 1.")

    dataset_path = Path(dataset_path).expanduser()

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Keep only the requested features and target. The original index is
    # retained so held-out predictions can be traced to source rows.
    experiment_frame = frame.loc[:, [*features, target_column]].copy()

    prepared = prepare_dataset(
        frame=experiment_frame,
        target_column=target_column,
        task="regression",
        test_size=test_size,
        random_seed=random_seed,
    )

    if not prepared.y_test.index.is_unique:
        raise ValueError(
            "The dataset has duplicate row indices. Use a unique DataFrame "
            "index before running an experiment so prediction records can "
            "be mapped unambiguously to source rows."
        )

    parameters = {
        "task": "regression",
        "evaluation_mode": "train_test_split",
        "target_column": target_column,
        "feature_columns": features,
        "test_size": float(test_size),
        "selected_models": models_to_run,
        "random_forest_estimators": random_forest_estimators,
        "neural_network": {
            "hidden_sizes": list(nn_hidden_sizes),
            "epochs": nn_epochs,
            "learning_rate": nn_learning_rate,
            "batch_size": nn_batch_size,
            "patience": nn_patience,
        },
    }

    metadata = build_experiment_metadata(
        dataset_path=dataset_path,
        experiment_name=experiment_name,
        random_seed=random_seed,
        parameters=parameters,
    )

    metadata["environment"]["pytorch_version"] = torch.__version__

    trained_results = {}

    classical_selected = {
        name for name in models_to_run
        if name in {"Linear Regression", "Random Forest"}
    }

    if classical_selected:
        classical_results = evaluate_regression_models(
            prepared=prepared,
            random_seed=random_seed,
            random_forest_estimators=random_forest_estimators,
        )

        for name in classical_selected:
            trained_results[name] = classical_results[name]

    if "Mean Baseline" in models_to_run:
        baseline_result = evaluate_regression_baseline(prepared)
        trained_results[baseline_result.model_name] = baseline_result

    neural_result = None

    if "PyTorch Neural Network" in models_to_run:
        neural_result = train_neural_network(
            prepared=prepared,
            random_seed=random_seed,
            hidden_sizes=nn_hidden_sizes,
            epochs=nn_epochs,
            learning_rate=nn_learning_rate,
            batch_size=nn_batch_size,
            patience=nn_patience,
        )

        trained_results[neural_result.model_name] = neural_result

    # Persist all model metrics first. Checkpoint links and prediction rows
    # reference these database records.
    experiment_id = save_experiment(
        metadata=metadata,
        model_results=trained_results,
        db_path=db_path,
    )

    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = {}
    prediction_counts = {}

    for model_name, result in trained_results.items():
        filename = (
            f"experiment_{experiment_id}_"
            f"{_safe_filename(model_name)}"
        )

        if model_name == "PyTorch Neural Network":
            checkpoint_path = save_torch_checkpoint(
                model=result.model,
                preprocessor=result.preprocessor,
                feature_columns=result.feature_columns,
                target_column=result.target_column,
                target_mean=result.target_mean,
                target_scale=result.target_scale,
                checkpoint_path=checkpoint_root / f"{filename}.pt",
                metrics=result.metrics,
                experiment_id=experiment_id,
                model_name=model_name,
                db_path=db_path,
            )
        else:
            checkpoint_path = save_checkpoint(
                model=result.model,
                preprocessor=prepared.preprocessor,
                feature_columns=prepared.feature_columns,
                target_column=prepared.target_column,
                task="regression",
                checkpoint_path=checkpoint_root / f"{filename}.joblib",
                metrics=result.metrics,
                experiment_id=experiment_id,
                model_name=model_name,
                db_path=db_path,
            )

        checkpoint_paths[model_name] = str(checkpoint_path)

        prediction_counts[model_name] = save_model_predictions(
            experiment_id=experiment_id,
            model_name=model_name,
            sample_indices=prepared.y_test.index.tolist(),
            actual_values=prepared.y_test.to_numpy(),
            predicted_values=result.predictions,
            db_path=db_path,
        )

    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name.strip(),
        "dataset_path": str(dataset_path),
        "target_column": target_column,
        "feature_columns": features,
        "random_seed": random_seed,
        "test_rows": len(prepared.y_test),
        "train_rows": len(prepared.y_train),
        "model_results": {
            name: {
                "mae": float(result.metrics["mae"]),
                "rmse": float(result.metrics["rmse"]),
                "r2": (
                    float(result.metrics["r2"])
                    if np.isfinite(result.metrics["r2"])
                    else None
                ),
            }
            for name, result in trained_results.items()
        },
        "checkpoint_paths": checkpoint_paths,
        "prediction_counts": prediction_counts,
    }
