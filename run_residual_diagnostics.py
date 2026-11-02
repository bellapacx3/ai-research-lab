
"""Generate residual diagnostic plots for a persisted experiment."""

import argparse
from pathlib import Path

from src.statistics.residual_diagnostics import (
    generate_experiment_diagnostics,
)


REPORT_DIR = Path("artifacts/reports")


def main():
    parser = argparse.ArgumentParser(
        description="Generate regression residual diagnostic plots."
    )

    parser.add_argument(
        "experiment_id",
        type=int,
        help="Experiment ID to analyze.",
    )

    args = parser.parse_args()

    if args.experiment_id < 1:
        parser.error("Experiment ID must be a positive integer.")

    output_dir = REPORT_DIR / (
        f"experiment_{args.experiment_id}_diagnostics"
    )

    print(
        f"\nGenerating diagnostic plots for "
        f"experiment {args.experiment_id}..."
    )

    results = generate_experiment_diagnostics(
        experiment_id=args.experiment_id,
        output_dir=output_dir,
    )

    total_plots = 0

    for model_name, plots in results["models"].items():
        print(f"\nModel: {model_name}")
        print("-" * 50)

        for plot_name, plot_path in plots.items():
            print(f"  {plot_name}: {plot_path}")
            total_plots += 1

    expected_plots = len(results["models"]) * 5

    if total_plots != expected_plots:
        raise RuntimeError(
            f"Expected {expected_plots} plots, "
            f"but generated {total_plots}."
        )

    print("\nDIAGNOSTICS COMPLETED")
    print("=" * 65)
    print(f"Experiment ID: {args.experiment_id}")
    print(f"Models analyzed: {len(results['models'])}")
    print(f"Plots generated: {total_plots}")
    print(f"Output directory: {output_dir}")

    print("\nSUCCESS: residual diagnostics completed.")


if __name__ == "__main__":
    main()
