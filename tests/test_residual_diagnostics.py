
import pytest

from src.statistics.residual_diagnostics import (
    _validate_predictions,
    generate_model_diagnostics,
)


def make_predictions():
    actual = [1.0, 2.0, 3.0, 4.0, 5.0]
    predicted = [1.1, 1.8, 3.2, 3.7, 5.3]

    records = []

    for y, prediction in zip(actual, predicted):
        records.append({
            "actual_value": y,
            "predicted_value": prediction,
            "residual": y - prediction,
        })

    return records


def test_prediction_validation():
    frame = _validate_predictions(make_predictions())

    assert len(frame) == 5
    assert "residual" in frame.columns


def test_empty_predictions_are_rejected():
    with pytest.raises(ValueError):
        _validate_predictions([])


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="Missing"):
        _validate_predictions([
            {"actual_value": 1.0}
        ])


def test_inconsistent_residuals_are_rejected():
    records = make_predictions()
    records[0]["residual"] = 100.0

    with pytest.raises(ValueError, match="inconsistent"):
        _validate_predictions(records)


def test_non_finite_predictions_are_rejected():
    records = make_predictions()
    records[0]["predicted_value"] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        _validate_predictions(records)


def test_all_diagnostic_plots_are_generated(tmp_path):
    results = generate_model_diagnostics(
        predictions=make_predictions(),
        model_name="Test Model",
        output_dir=tmp_path,
    )

    assert len(results) == 5

    for path in results.values():
        from pathlib import Path
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0
