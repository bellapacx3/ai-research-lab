"""Utilities for comparing evaluated regression models."""

import pandas as pd

from src.ml.classical_models import RegressionResult


SUPPORTED_METRICS = {"mae", "rmse", "r2"}


def compare_regression_results(
    results: dict[str, RegressionResult],
    sort_by: str = "rmse",
) -> pd.DataFrame:
    """Build a ranked comparison table from regression results."""

    if not results:
        raise ValueError("At least one model result is required.")

    if sort_by not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric '{sort_by}'. "
            f"Choose from {sorted(SUPPORTED_METRICS)}."
        )

    rows = []

    for name, result in results.items():
        rows.append({
            "model": name,
            "mae": result.metrics["mae"],
            "rmse": result.metrics["rmse"],
            "r2": result.metrics["r2"],
        })

    comparison = pd.DataFrame(rows)

    # Lower MAE/RMSE is better; higher R² is better.
    ascending = sort_by != "r2"

    comparison = comparison.sort_values(
        by=sort_by,
        ascending=ascending,
        na_position="last",
    ).reset_index(drop=True)

    comparison.insert(
        0,
        "rank",
        range(1, len(comparison) + 1),
    )

    return comparison
