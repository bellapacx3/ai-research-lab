import pandas as pd
import pytest

from src.data.preprocessing import prepare_dataset
from src.ml.classical_models import evaluate_regression_models
from src.experiments.comparison import compare_regression_results


@pytest.fixture
def model_results():
    frame = pd.DataFrame({
        "temperature": range(18, 30),
        "humidity": [40, 42, 43, 45, 48, 50, 51, 53, 55, 57, 58, 60],
        "material": [
            "steel", "wood", "steel", "plastic", "wood", "steel",
            "plastic", "wood", "steel", "plastic", "wood", "steel",
        ],
        "measurement": [
            10.0, 8.0, 10.5, 9.0, 8.5, 11.0,
            9.5, 8.2, 10.8, 9.7, 8.8, 11.4,
        ],
    })

    prepared = prepare_dataset(
        frame,
        target_column="measurement",
        task="regression",
        test_size=0.25,
        random_seed=42,
    )

    return evaluate_regression_models(prepared)


def test_comparison_returns_expected_columns(model_results):
    result = compare_regression_results(model_results)

    assert list(result.columns) == ["rank", "model", "mae", "rmse", "r2"]
    assert len(result) == 2
    assert result["rank"].tolist() == [1, 2]


def test_rmse_ranking_places_lower_error_first(model_results):
    result = compare_regression_results(model_results, sort_by="rmse")

    assert result.iloc[0]["rmse"] <= result.iloc[1]["rmse"]


def test_r2_ranking_places_higher_score_first(model_results):
    result = compare_regression_results(model_results, sort_by="r2")

    assert result.iloc[0]["r2"] >= result.iloc[1]["r2"]


def test_empty_results_are_rejected():
    with pytest.raises(ValueError, match="At least one"):
        compare_regression_results({})


def test_unsupported_metric_is_rejected(model_results):
    with pytest.raises(ValueError, match="Unsupported metric"):
        compare_regression_results(model_results, sort_by="accuracy")
