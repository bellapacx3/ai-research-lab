
"""Run, report, and persist a cross-validation experiment."""

import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src.data.ingestion import load_dataset
from src.experiments.cross_validation import (
    cross_validate_regression_models,
)
from src.experiments.reproducibility import (
    build_experiment_metadata,
)
from src.storage.repositories import (
    save_cross_validation_experiment,
)


DATASET = Path("data/datasets/california_housing.csv")
TARGET = "MedHouseVal"
SEED = 42
FOLDS = 5
RANDOM_FOREST_ESTIMATORS = 200


def main():
    print("Loading California Housing dataset...")
    frame = load_dataset(DATASET)

    print(f"Rows: {len(frame):,}")
    print(f"Target: {TARGET}")
    print(f"Cross-validation folds: {FOLDS}")

    result = cross_validate_regression_models(
        frame,
        target_column=TARGET,
        n_splits=FOLDS,
        random_seed=SEED,
        random_forest_estimators=RANDOM_FOREST_ESTIMATORS,
    )

    print("\nFold-level results")
    print(result.fold_metrics.to_string(index=False))

    print("\nCross-validation summary")
    print(result.summary.to_string(index=False))

    print("\nExperiment configuration")
    print(f"Rows evaluated: {result.rows_evaluated:,}")
    print(f"Features: {result.feature_columns}")
    print(f"Random seed: {result.random_seed}")
    print(f"Folds: {result.n_splits}")

    metadata = build_experiment_metadata(
    dataset_path=DATASET,
    experiment_name="California Housing - 5-Fold Cross-Validation",
    random_seed=SEED,
    parameters={
        "target_column": TARGET,
        "n_splits": FOLDS,
        "random_forest_estimators": RANDOM_FOREST_ESTIMATORS,
        "feature_columns": result.feature_columns,
        "rows_evaluated": result.rows_evaluated,
        "validation_method": "shuffled_k_fold",
    },
)

    experiment_id = save_cross_validation_experiment(
    metadata=metadata,
    cv_result=result,
)


    print("\nPersistence")
    print("Status: SUCCESS")
    print(f"Experiment ID: {experiment_id}")
    print("Database: artifacts/research_lab.db")


if __name__ == "__main__":
    main()
