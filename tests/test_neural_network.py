
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.ml.neural_network import (
    ResearchNeuralNetwork,
    train_neural_network,
)


def make_dataset():
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(40, 3)).astype(np.float32)
    y_train = 2 * X_train[:, 0] - X_train[:, 1] + 0.5 * X_train[:, 2]

    X_test = rng.normal(size=(10, 3)).astype(np.float32)
    y_test = 2 * X_test[:, 0] - X_test[:, 1] + 0.5 * X_test[:, 2]

    return SimpleNamespace(
        task="regression",
        X_train=X_train,
        X_test=X_test,
        y_train=pd.Series(y_train),
        y_test=pd.Series(y_test),
        preprocessor=None,
        feature_columns=["a", "b", "c"],
        target_column="target",
    )


def test_network_output_shape():
    import torch

    model = ResearchNeuralNetwork(input_features=3)
    output = model(torch.randn(5, 3))

    assert output.shape == (5, 1)


def test_training_produces_predictions_and_metrics():
    result = train_neural_network(make_dataset(), epochs=10)

    assert result.predictions.shape == (10,)
    assert set(result.metrics) == {"mae", "rmse", "r2"}
    assert np.isfinite(result.predictions).all()
    assert result.metrics["mae"] >= 0
    assert result.metrics["rmse"] >= 0
    assert len(result.training_losses) > 0


def test_training_is_reproducible():
    dataset = make_dataset()

    first = train_neural_network(dataset, random_seed=42, epochs=10)
    second = train_neural_network(dataset, random_seed=42, epochs=10)

    np.testing.assert_allclose(
        first.predictions,
        second.predictions,
        rtol=1e-6,
        atol=1e-6,
    )


def test_rejects_classification():
    dataset = make_dataset()
    dataset.task = "classification"

    with pytest.raises(ValueError, match="regression only"):
        train_neural_network(dataset)


def test_rejects_invalid_epoch_count():
    with pytest.raises(ValueError, match="epochs"):
        train_neural_network(make_dataset(), epochs=0)
