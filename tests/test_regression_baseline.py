
import numpy as np

from src.ml.classical_models import (
    evaluate_regression_baseline,
    evaluate_regression_models,
)


def test_mean_baseline_returns_expected_metrics(prepared_data):
    result = evaluate_regression_baseline(prepared_data)

    assert result.model_name == "Mean Baseline"
    assert set(result.metrics) == {"mae", "rmse", "r2"}
    assert len(result.predictions) == len(prepared_data.y_test)
    assert np.isfinite(result.predictions).all()


def test_mean_baseline_predicts_training_target_mean(prepared_data):
    result = evaluate_regression_baseline(prepared_data)

    expected = float(np.mean(prepared_data.y_train))

    np.testing.assert_allclose(
        result.predictions,
        np.full(len(prepared_data.y_test), expected),
    )


def test_existing_evaluator_still_returns_two_models(prepared_data):
    results = evaluate_regression_models(prepared_data)

    assert set(results) == {"Linear Regression", "Random Forest"}
