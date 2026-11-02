
"""Residual diagnostics and visualization for regression experiments."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.storage.repositories import get_model_predictions


def _validate_predictions(predictions: list[dict]) -> pd.DataFrame:
    """Validate and prepare persisted prediction records."""

    if not predictions:
        raise ValueError("At least one prediction is required.")

    frame = pd.DataFrame(predictions)

    required = {
        "actual_value",
        "predicted_value",
        "residual",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing)}"
        )

    for column in required:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    if not np.isfinite(frame[list(required)].to_numpy()).all():
        raise ValueError("Prediction data contains non-finite values.")

    expected_residuals = (
        frame["actual_value"] - frame["predicted_value"]
    )

    if not np.allclose(
        expected_residuals,
        frame["residual"],
    ):
        raise ValueError("Stored residuals are inconsistent.")

    return frame.sort_values("actual_value").reset_index(drop=True)


def plot_actual_vs_predicted(
    frame: pd.DataFrame,
    output_path: Path,
    model_name: str,
) -> None:
    """Plot actual values against predicted values."""

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        frame["actual_value"],
        frame["predicted_value"],
        alpha=0.35,
        s=12,
    )

    minimum = min(
        frame["actual_value"].min(),
        frame["predicted_value"].min(),
    )
    maximum = max(
        frame["actual_value"].max(),
        frame["predicted_value"].max(),
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        label="Perfect prediction",
    )

    ax.set(
        title=f"Actual vs Predicted - {model_name}",
        xlabel="Actual value",
        ylabel="Predicted value",
    )
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residual_distribution(
    frame: pd.DataFrame,
    output_path: Path,
    model_name: str,
) -> None:
    """Plot the distribution of residuals."""

    residuals = frame["residual"]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(
        residuals,
        bins=50,
        edgecolor="black",
        alpha=0.75,
    )

    ax.axvline(
        residuals.mean(),
        linestyle="--",
        label=f"Mean: {residuals.mean():.3f}",
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1,
        label="Zero residual",
    )

    ax.set(
        title=f"Residual Distribution - {model_name}",
        xlabel="Residual (actual - predicted)",
        ylabel="Frequency",
    )
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residuals_vs_predicted(
    frame: pd.DataFrame,
    output_path: Path,
    model_name: str,
) -> None:
    """Plot residuals against predicted values."""

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        frame["predicted_value"],
        frame["residual"],
        alpha=0.35,
        s=12,
    )

    ax.axhline(0, linestyle="--", color="black")

    ax.set(
        title=f"Residuals vs Predicted - {model_name}",
        xlabel="Predicted value",
        ylabel="Residual",
    )
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residual_qq(
    frame: pd.DataFrame,
    output_path: Path,
    model_name: str,
) -> None:
    """Generate a normal Q-Q plot of residuals."""

    fig, ax = plt.subplots(figsize=(9, 6))

    stats.probplot(
        frame["residual"],
        dist="norm",
        plot=ax,
    )

    ax.set_title(f"Residual Q-Q Plot - {model_name}")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_error_by_target_range(
    frame: pd.DataFrame,
    output_path: Path,
    model_name: str,
) -> None:
    """Compare absolute prediction errors across target ranges."""

    analysis = frame.copy()

    analysis["target_range"] = pd.cut(
        analysis["actual_value"],
        bins=5,
        duplicates="drop",
    )

    grouped = (
        analysis.groupby("target_range", observed=True)
        ["residual"]
        .apply(lambda values: np.mean(np.abs(values)))
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    grouped.plot(kind="bar", ax=ax)

    ax.set(
        title=f"MAE by Actual Target Range - {model_name}",
        xlabel="Actual target range",
        ylabel="Mean absolute error",
    )

    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_model_diagnostics(
    predictions: list[dict],
    model_name: str,
    output_dir: Path,
) -> dict:
    """Generate all diagnostic plots for one model."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _validate_predictions(predictions)

    plots = {
        "actual_vs_predicted": plot_actual_vs_predicted,
        "residual_distribution": plot_residual_distribution,
        "residuals_vs_predicted": plot_residuals_vs_predicted,
        "qq_plot": plot_residual_qq,
        "error_by_target_range": plot_error_by_target_range,
    }

    generated = {}

    for filename, plot_function in plots.items():
        path = output_dir / f"{filename}.png"

        plot_function(
            frame,
            path,
            model_name,
        )

        generated[filename] = str(path)

    return generated


def generate_experiment_diagnostics(
    experiment_id: int,
    output_dir: Path,
    db_path=None,
) -> dict:
    """Generate diagnostic plots for all models in an experiment."""

    if (
        not isinstance(experiment_id, int)
        or isinstance(experiment_id, bool)
        or experiment_id < 1
    ):
        raise ValueError("experiment_id must be a positive integer.")

    output_dir = Path(output_dir)

    if db_path is None:
        predictions = get_model_predictions(
        experiment_id=experiment_id,
    )
    else:
        predictions = get_model_predictions(
        experiment_id=experiment_id,
        db_path=db_path,
    )

    if not predictions:
        raise ValueError(
            f"No predictions found for experiment {experiment_id}."
        )

    frame = pd.DataFrame(predictions)
    results = {}

    for model_name, group in frame.groupby("model_name", sort=True):
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("/", "_")
        )

        results[model_name] = generate_model_diagnostics(
            predictions=group.to_dict(orient="records"),
            model_name=model_name,
            output_dir=output_dir / safe_name,
        )

    return {
        "experiment_id": experiment_id,
        "models": results,
    }
