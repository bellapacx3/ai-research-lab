"""Generate reports for a saved cross-validation experiment."""

import argparse

from src.reports.experiment_report import generate_cross_validation_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reports from a saved cross-validation experiment."
    )
    parser.add_argument(
        "experiment_id",
        type=int,
        help="ID of the saved cross-validation experiment.",
    )
    args = parser.parse_args()

    try:
        paths = generate_cross_validation_report(args.experiment_id)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    print(f"Reports generated for experiment {args.experiment_id}:")
    for report_type, path in paths.items():
        print(f"{report_type}: {path}")


if __name__ == "__main__":
    main()
