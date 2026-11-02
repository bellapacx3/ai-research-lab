
import pytest

from src.statistics.residual_analysis import (
    analyze_residuals,
)


def make_predictions():
    actual = [3.0, 4.0, 5.0, 6.0]
    predicted = [2.5, 4.5, 5.0, 7.0]

    records = []

    for y, prediction in zip(actual, predicted):
        residual = y - prediction

        records.append({
            "actual_value": y,
            "predicted_value": prediction,
            "residual": residual,
            "absolute_error": abs(residual),
            "squared_error": residual ** 2,
        })

    return records


def test_analyze_residuals_returns_expected_metrics():
    result = analyze_residuals(make_predictions())

    assert result["sample_count"] == 4
    assert result["mean_residual"] == pytest.approx(-0.25)
    assert result["median_residual"] == pytest.approx(-0.25)
    assert result["mae"] == pytest.approx(0.5)
    assert result["rmse"] == pytest.approx(0.6123724357)


def test_prediction_direction_counts():
    result = analyze_residuals(make_predictions())

    assert result["overpredictions"] == 2
    assert result["underpredictions"] == 1
    assert result["exact_predictions"] == 1


def test_error_percentiles_are_calculated():
    result = analyze_residuals(make_predictions())

    percentiles = result["mean_absolute_error_percentiles"]

    assert percentiles["50th"] == pytest.approx(0.5)
    assert percentiles["75th"] == pytest.approx(0.625)
    assert percentiles["90th"] == pytest.approx(0.85)
    assert percentiles["95th"] == pytest.approx(0.925)
    assert percentiles["99th"] == pytest.approx(0.985)


def test_empty_predictions_are_rejected():
    with pytest.raises(ValueError):
        analyze_residuals([])


def test_inconsistent_residuals_are_rejected():
    records = make_predictions()
    records[0]["residual"] = 100.0

    with pytest.raises(ValueError, match="residuals"):
        analyze_residuals(records)


def test_non_finite_values_are_rejected():
    records = make_predictions()
    records[0]["actual_value"] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        analyze_residuals(records)
