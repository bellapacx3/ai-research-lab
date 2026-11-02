
"""Compare classical models with a PyTorch neural network."""

from pathlib import Path

import pandas as pd

from src.data.ingestion import load_dataset
from src.data.preprocessing import prepare_dataset
from src.experiments.comparison import compare_regression_results
from src.ml.neural_network import train_neural_network
from src.ml.classical_models import (
    evaluate_regression_baseline,
    evaluate_regression_models,
)

DATASET = Path("data/datasets/california_housing.csv")
TARGET = "MedHouseVal"
SEED = 42
TEST_SIZE = 0.25

EXCLUDED_FEATURES = []


def find_exact_target_matches(
    frame: pd.DataFrame,
    target_column: str,
) -> list[str]:
    """Find columns whose values exactly reproduce the target.

    This is a basic warning check, not a complete leakage detector.
    """
    if target_column not in frame.columns:
        return []

    valid = frame.dropna(subset=[target_column])
    target = valid[target_column]
    matches = []

    for column in valid.columns:
        if column == target_column:
            continue

        if valid[column].equals(target):
            matches.append(column)

    return matches


def main():
    print("\n[1/5] Loading and auditing dataset...")
    original_frame = load_dataset(DATASET)

    print(
        f"Original rows: {len(original_frame)} | "
        f"Columns: {len(original_frame.columns)}"
    )

    duplicate_count = int(original_frame.duplicated().sum())
    print(f"Exact duplicate rows found: {duplicate_count}")

    # Remove exact duplicate records for this experiment only.
    frame = original_frame.drop_duplicates().copy()

    print(f"Rows after duplicate removal: {len(frame)}")

    if duplicate_count:
        print(
            f"Removed {duplicate_count} exact duplicate row(s) "
            "from this experiment's working dataset."
        )

    print("\nChecking for exact target leakage...")
    target_matches = find_exact_target_matches(frame, TARGET)

    if target_matches:
        print(
            "WARNING: These feature columns exactly match the target: "
            f"{target_matches}"
        )
        raise ValueError(
            "Potential target leakage detected. Investigate the listed "
            "columns before training."
        )

    print("No exact target-matching feature columns found.")

    missing_exclusions = [
        column
        for column in EXCLUDED_FEATURES
        if column not in frame.columns
    ]
    excluded_features = [
        column
        for column in EXCLUDED_FEATURES
        if column in frame.columns
    ]

    if missing_exclusions:
        print(
            "Note: configured exclusion columns not present in dataset: "
            f"{missing_exclusions}"
        )

    print(f"Excluded metadata columns: {excluded_features or 'None'}")

    print("\n[2/5] Preparing features...")
    prepared = prepare_dataset(
        frame,
        target_column=TARGET,
        task="regression",
        test_size=TEST_SIZE,
        random_seed=SEED,
        exclude_columns=excluded_features,
    )

    print(f"Feature columns: {prepared.feature_columns}")
    print(f"Training rows: {len(prepared.X_train)}")
    print(f"Test rows: {len(prepared.X_test)}")

    print("\n[3/5] Training classical models...")
    classical_results = evaluate_regression_models(
        prepared,
        random_seed=SEED,
    )

    print("Training PyTorch neural network...")
    neural_result = train_neural_network(
        prepared,
        random_seed=SEED,
        hidden_sizes=(32, 16),
        epochs=200,
        learning_rate=0.001,
        batch_size=16,
        patience=25,
    ) 
    baseline_result = evaluate_regression_baseline(prepared)

    all_results = dict(classical_results)
    all_results[baseline_result.model_name] = baseline_result
    all_results[neural_result.model_name] = neural_result

    print("\n[4/5] Comparing model performance...")
    comparison = compare_regression_results(
        all_results,
        sort_by="rmse",
    )
    print(comparison.to_string(index=False))

    print("\n[5/5] Experiment diagnostics")
    print("\nNeural-network test predictions")
    print(f"Actual values:     {prepared.y_test.to_numpy()}")
    print(f"Predicted values:  {neural_result.predictions}")
    print(f"Training target mean: {prepared.y_train.mean():.4f}")

    print("\nData quality summary")
    print(f"Original exact duplicates: {duplicate_count}")
    print(f"Rows used after deduplication: {len(frame)}")
    print(f"Excluded columns: {excluded_features or 'None'}")
    print(f"Features used: {prepared.feature_columns}")

    print("\nNeural-network training summary")
    print(f"Epochs completed: {len(neural_result.training_losses)}")
    print(f"Final training loss: {neural_result.training_losses[-1]:.6f}")

    if neural_result.validation_losses:
        best_validation_loss = min(neural_result.validation_losses)
        print(f"Best validation loss: {best_validation_loss:.6f}")
    else:
        print("Validation loss unavailable for this small training split.")


if __name__ == "__main__":
    main()
