
"""Run statistical residual analysis for a persisted experiment."""

import argparse
import csv
import json
from pathlib import Path

from src.statistics.residual_analysis import (
    analyze_experiment_residuals,
)


REPORT_DIR = Path("artifacts/reports")


def export_json(results: dict, output_path: Path) -> None:
    """Export complete residual analysis results to JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )


def export_csv(results: dict, output_path: Path) -> None:
    """Export model-level residual statistics to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for model_name, metrics in results["models"].items():
        percentiles = metrics["mean_absolute_error_percentiles"]

        rows.append({
            "experiment_id": results["experiment_id"],
            "model_name": model_name,
            "sample_count": metrics["sample_count"],
            "mean_residual": metrics["mean_residual"],
            "median_residual": metrics["median_residual"],
            "residual_std": metrics["residual_std"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mae_p50": percentiles["50th"],
            "mae_p75": percentiles["75th"],
            "mae_p90": percentiles["90th"],
            "mae_p95": percentiles["95th"],
            "mae_p99": percentiles["99th"],
            "overpredictions": metrics["overpredictions"],
            "underpredictions": metrics["underpredictions"],
            "exact_predictions": metrics["exact_predictions"],
            "overprediction_percentage": metrics[
                "overprediction_percentage"
            ],
            "underprediction_percentage": metrics[
                "underprediction_percentage"
            ],
        })

    if not rows:
        raise ValueError("No model analysis results are available.")

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(results: dict) -> None:
    """Print a readable summary of residual analysis."""

    print("\nRESIDUAL ANALYSIS SUMMARY")
    print("=" * 75)
    print(f"Experiment ID: {results['experiment_id']}")

    for model_name, metrics in results["models"].items():
        print(f"\nModel: {model_name}")
        print("-" * 50)

        print(f"Samples analyzed: {metrics['sample_count']}")
        print(f"Mean residual: {metrics['mean_residual']:.6f}")
        print(f"Median residual: {metrics['median_residual']:.6f}")
        print(f"Residual standard deviation: {metrics['residual_std']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")

        print("\nPrediction direction:")
        print(f"  Overpredictions: {metrics['overpredictions']}")
        print(f"  Underpredictions: {metrics['underpredictions']}")
        print(f"  Exact predictions: {metrics['exact_predictions']}")

        print("\nAbsolute error percentiles:")

        for percentile, value in (
            metrics["mean_absolute_error_percentiles"].items()
        ):
            print(f"  {percentile}: {value:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze persisted regression predictions."
    )

    parser.add_argument(
        "experiment_id",
        type=int,
        help="ID of the experiment to analyze.",
    )

    args = parser.parse_args()

    if args.experiment_id < 1:
        parser.error("Experiment ID must be a positive integer.")

    print(
        f"\nLoading predictions for experiment "
        f"{args.experiment_id}..."
    )

    results = analyze_experiment_residuals(args.experiment_id)

    print_summary(results)

    json_path = REPORT_DIR / (
        f"experiment_{args.experiment_id}_residual_analysis.json"
    )

    csv_path = REPORT_DIR / (
        f"experiment_{args.experiment_id}_residual_analysis.csv"
    )

    export_json(results, json_path)
    export_csv(results, csv_path)

    print("\nREPORTS GENERATED")
    print("=" * 75)
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")

    print("\nSUCCESS: residual analysis completed.")


if __name__ == "__main__":
    main()
