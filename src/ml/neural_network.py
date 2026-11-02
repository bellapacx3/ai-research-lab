
"""PyTorch feed-forward neural network for regression experiments."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class NeuralNetworkResult:
    """Results and diagnostics from a neural-network experiment."""

    model_name: str
    model: nn.Module
    predictions: np.ndarray
    metrics: dict[str, float]
    training_losses: list[float]
    validation_losses: list[float]
    preprocessor: Any
    feature_columns: list[str]
    target_column: str
    target_mean: float
    target_scale: float
    task: str = "regression"


class ResearchNeuralNetwork(nn.Module):
    """Configurable fully connected neural network."""

    def __init__(
        self,
        input_features: int,
        hidden_sizes: tuple[int, ...] = (32, 16),
        output_features: int = 1,
    ):
        super().__init__()

        if input_features < 1:
            raise ValueError("input_features must be positive.")
        if output_features < 1:
            raise ValueError("output_features must be positive.")
        if not hidden_sizes or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 1
            for size in hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain positive integers.")

        layers: list[nn.Module] = []
        previous_size = input_features

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, output_features))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map input features to predicted target values."""
        return self.network(features)


def _to_dense_float32(features) -> np.ndarray:
    """Convert dense or sparse preprocessed features to finite float32."""
    if sparse.issparse(features):
        values = features.toarray()
    else:
        values = np.asarray(features)

    values = np.asarray(values, dtype=np.float32)

    if values.ndim != 2:
        raise ValueError("Preprocessed features must be two-dimensional.")

    if not np.isfinite(values).all():
        raise ValueError("Preprocessed features must contain finite values.")

    return values


def _predict(model: nn.Module, features: np.ndarray) -> np.ndarray:
    """Generate predictions from preprocessed feature values."""
    model.eval()

    tensor = torch.as_tensor(
        _to_dense_float32(features),
        dtype=torch.float32,
    )

    with torch.no_grad():
        predictions = model(tensor).cpu().numpy().reshape(-1)

    return predictions


def train_neural_network(
    prepared,
    random_seed: int = 42,
    hidden_sizes: tuple[int, ...] = (32, 16),
    epochs: int = 200,
    learning_rate: float = 0.001,
    batch_size: int = 16,
    patience: int = 25,
) -> NeuralNetworkResult:
    """Train and evaluate a neural network using a PreparedDataset.

    The test split is reserved for final evaluation. When enough training
    rows exist, a validation subset is held out from the training data
    for early stopping.

    Target scaling is fitted on the fitting subset only. Predictions and
    evaluation metrics are returned in the original target units.
    """
    if prepared.task != "regression":
        raise ValueError(
            "This initial neural network supports regression only."
        )

    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if patience < 1:
        raise ValueError("patience must be at least 1.")

    X_train = _to_dense_float32(prepared.X_train)
    X_test = _to_dense_float32(prepared.X_test)
    y_train = np.asarray(prepared.y_train, dtype=np.float32).reshape(-1)
    y_test = np.asarray(prepared.y_test, dtype=np.float32).reshape(-1)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(
            "Training and test features must be two-dimensional."
        )
    if len(X_train) != len(y_train):
        raise ValueError(
            "Training features and targets have different lengths."
        )
    if len(X_test) != len(y_test):
        raise ValueError(
            "Test features and targets have different lengths."
        )
    if len(X_train) < 2 or len(X_test) < 1:
        raise ValueError(
            "At least two training rows and one test row are required."
        )
    if X_train.shape[1] < 1 or X_test.shape[1] != X_train.shape[1]:
        raise ValueError(
            "Training and test feature counts must match."
        )
    if not all(
        np.isfinite(array).all()
        for array in (X_train, X_test, y_train, y_test)
    ):
        raise ValueError(
            "Training and test arrays must contain finite values."
        )

    # Reserve part of the training data for validation when possible.
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(len(X_train))

    if len(X_train) >= 5:
        validation_count = max(
            1,
            int(round(len(X_train) * 0.2)),
        )
        validation_count = min(
            validation_count,
            len(X_train) - 2,
        )
        validation_indices = indices[:validation_count]
        fit_indices = indices[validation_count:]
    else:
        validation_indices = np.array([], dtype=int)
        fit_indices = indices

    X_fit = X_train[fit_indices]
    y_fit = y_train[fit_indices]

    X_validation = X_train[validation_indices]
    y_validation = y_train[validation_indices]

    # Fit target scaling on the fitting subset only.
    target_mean = float(np.mean(y_fit))
    target_scale = float(np.std(y_fit))

    # Avoid division by zero when fitting targets are constant.
    if target_scale < 1e-8:
        target_scale = 1.0

    y_fit_scaled = (y_fit - target_mean) / target_scale

    y_validation_scaled = (
        (y_validation - target_mean) / target_scale
        if len(y_validation)
        else y_validation
    )

    # Seed the CPU generator without explicitly initializing CUDA.
    torch.random.default_generator.manual_seed(random_seed)

    model = ResearchNeuralNetwork(
        input_features=X_train.shape[1],
        hidden_sizes=hidden_sizes,
    )

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    fit_dataset = TensorDataset(
        torch.as_tensor(X_fit, dtype=torch.float32),
        torch.as_tensor(
            y_fit_scaled.reshape(-1, 1),
            dtype=torch.float32,
        ),
    )

    loader_generator = torch.Generator().manual_seed(random_seed)
    loader = DataLoader(
        fit_dataset,
        batch_size=min(batch_size, len(fit_dataset)),
        shuffle=True,
        generator=loader_generator,
    )

    training_losses: list[float] = []
    validation_losses: list[float] = []

    best_state = deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(epochs):
        # Training: forward pass, loss, backpropagation, optimizer update.
        model.train()
        batch_losses = []

        for batch_features, batch_targets in loader:
            optimizer.zero_grad()

            predictions = model(batch_features)
            loss = loss_function(predictions, batch_targets)

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        training_losses.append(float(np.mean(batch_losses)))

        # Validation measures generalization; it does not update weights.
        if len(validation_indices):
            model.eval()

            with torch.no_grad():
                validation_predictions = model(
                    torch.as_tensor(
                        X_validation,
                        dtype=torch.float32,
                    )
                )
                validation_targets = torch.as_tensor(
                    y_validation_scaled.reshape(-1, 1),
                    dtype=torch.float32,
                )

                validation_loss = loss_function(
                    validation_predictions,
                    validation_targets,
                ).item()

            validation_losses.append(float(validation_loss))

            if validation_loss < best_validation_loss - 1e-8:
                best_validation_loss = validation_loss
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

    # Restore the best validation model when validation was available.
    if len(validation_indices):
        model.load_state_dict(best_state)

    # Convert predictions back to the original target units.
    predictions = (
        _predict(model, X_test) * target_scale + target_mean
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

    return NeuralNetworkResult(
        model_name="PyTorch Neural Network",
        model=model,
        predictions=predictions,
        metrics=metrics,
        training_losses=training_losses,
        validation_losses=validation_losses,
        preprocessor=prepared.preprocessor,
        feature_columns=list(prepared.feature_columns),
        target_column=prepared.target_column,
        target_mean=target_mean,
        target_scale=target_scale,
    )


def predict_neural_network(
    model: nn.Module,
    preprocessor,
    raw_features: pd.DataFrame,
    feature_columns: list[str],
    target_mean: float = 0.0,
    target_scale: float = 1.0,
) -> np.ndarray:
    """Predict from raw features and return values in original target units."""
    if not isinstance(raw_features, pd.DataFrame):
        raise TypeError("raw_features must be a pandas DataFrame.")

    if not np.isfinite(target_mean):
        raise ValueError("target_mean must be finite.")
    if not np.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("target_scale must be positive and finite.")

    missing = [
        column
        for column in feature_columns
        if column not in raw_features.columns
    ]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    transformed = preprocessor.transform(
        raw_features.loc[:, feature_columns]
    )

    # Apply the same inverse target scaling used after training.
    return _predict(model, transformed) * target_scale + target_mean
