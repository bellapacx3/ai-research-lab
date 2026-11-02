"""Generate research reports from persisted cross-validation experiments."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.storage.repositories import get_cross_validation_results


DEFAULT_REPORT_DIR = Path("artifacts/reports")


def generate_cross_validation_report(
    experiment_id: int,
    output_dir: str | Path = DEFAULT_REPORT_DIR,
) -> dict[str, str]:
    """Generate Markdown, CSV, and JSON reports for a saved experiment."""
    if (
        not isinstance(experiment_id, int)
        or isinstance(experiment_id, bool)
        or experiment_id < 1
    ):
        raise ValueError("experiment_id must be a positive integer.")

    result = get_cross_validation_results(experiment_id)

    if result is None:
        raise ValueError(f"Experiment {experiment_id} was not found.")

    experiment = result["experiment"]
    summaries = result["summary"]
    folds = result["folds"]

    if not summaries:
        raise ValueError(
            f"Experiment {experiment_id} has no cross-validation summaries."
        )

    if not folds:
        raise ValueError(
            f"Experiment {experiment_id} has no cross-validation fold records."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stem = f"experiment_{experiment_id}_cross_validation"

    markdown_path = output_path / f"{stem}.md"
    summary_csv_path = output_path / f"{stem}_summary.csv"
    folds_csv_path = output_path / f"{stem}_folds.csv"
    json_path = output_path / f"{stem}.json"

    parameters = json.loads(experiment["parameters_json"])
    environment = json.loads(experiment["environment_json"])

    summary_frame = pd.DataFrame(summaries)
    folds_frame = pd.DataFrame(folds)

    best_model = min(summaries, key=lambda row: row["rmse_mean"])

    lines = [
        f"# {experiment['experiment_name']}",
        "",
        f"- **Experiment ID:** {experiment['id']}",
        f"- **Created at (UTC):** {experiment['created_at']}",
        f"- **Dataset:** `{experiment['dataset_path']}`",
        f"- **Dataset SHA-256:** `{experiment['dataset_sha256']}`",
        f"- **Random seed:** {experiment['random_seed']}",
        f"- **Best model by mean RMSE:** {best_model['model_name']}",
        "",
        "## Cross-validation summary",
        "",
        "| Model | Mean MAE | MAE std. | Mean RMSE | RMSE std. | Mean R² | R² std. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summaries:
        lines.append(
            "| {model} | {mae:.6f} | {mae_std} | {rmse:.6f} | "
            "{rmse_std} | {r2} | {r2_std} |".format(
                model=row["model_name"],
                mae=row["mae_mean"],
                mae_std=_format_metric(row["mae_std"]),
                rmse=row["rmse_mean"],
                rmse_std=_format_metric(row["rmse_std"]),
                r2=_format_metric(row["r2_mean"]),
                r2_std=_format_metric(row["r2_std"]),
            )
        )

    lines.extend(
        [
            "",
            "## Experiment parameters",
            "",
            "```json",
            json.dumps(parameters, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Software environment",
            "",
            "```json",
            json.dumps(environment, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Fold-level results",
            "",
            "| Fold | Model | Train rows | Validation rows | MAE | RMSE | R² |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )

    for row in folds:
        lines.append(
            "| {fold} | {model} | {train} | {validation} | "
            "{mae:.6f} | {rmse:.6f} | {r2} |".format(
                fold=row["fold"],
                model=row["model_name"],
                train=row["train_rows"],
                validation=row["validation_rows"],
                mae=row["mae"],
                rmse=row["rmse"],
                r2=_format_metric(row["r2"]),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"**{best_model['model_name']}** has the lowest mean RMSE "
            f"({best_model['rmse_mean']:.6f}) among the saved models.",
            "",
            "These metrics summarize the recorded cross-validation folds. "
            "They do not replace evaluation on a separately reserved final "
            "test set.",
            "",
        ]
    )

    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    summary_frame.to_csv(summary_csv_path, index=False)
    folds_frame.to_csv(folds_csv_path, index=False)

    json_payload: dict[str, Any] = {
        "experiment": {
            "id": experiment["id"],
            "experiment_name": experiment["experiment_name"],
            "created_at": experiment["created_at"],
            "dataset_path": experiment["dataset_path"],
            "dataset_sha256": experiment["dataset_sha256"],
            "random_seed": experiment["random_seed"],
        },
        "parameters": parameters,
        "environment": environment,
        "summary": summaries,
        "folds": folds,
        "best_model_by_mean_rmse": best_model["model_name"],
    }

    json_path.write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    return {
        "markdown": str(markdown_path),
        "summary_csv": str(summary_csv_path),
        "folds_csv": str(folds_csv_path),
        "json": str(json_path),
    }


def _format_metric(value: Any) -> str:
    """Format an optional metric for readable reports."""
    if value is None:
        return "N/A"

    return f"{value:.6f}"
