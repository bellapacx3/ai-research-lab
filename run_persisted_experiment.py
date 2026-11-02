
from pathlib import Path

import numpy as np

from src.data.ingestion import load_dataset
from src.data.preprocessing import prepare_dataset
from src.experiments.comparison import compare_regression_results
from src.experiments.reproducibility import build_experiment_metadata
from src.ml.checkpoints import (
    predict_with_checkpoint,
    save_checkpoint,
)
from src.ml.classical_models import (
    evaluate_regression_models,
    evaluate_regression_baseline,
)
from src.storage.repositories import (
    save_experiment,
    get_experiment,
    save_model_predictions,
    get_model_predictions,
)


DATASET = Path("data/datasets/california_housing.csv")
TARGET = "MedHouseVal"
SEED = 42
TEST_SIZE = 0.25
CHECKPOINT_DIR = Path("artifacts/models")


def main():
    # 1. Load and inspect the dataset.
    print("\n[1/7] Loading dataset...")
    frame = load_dataset(DATASET)
    print(f"Rows: {len(frame)} | Columns: {len(frame.columns)}")

    # Preserve the original data for checkpoint prediction.
    raw_frame = frame.copy()

    # 2. Prepare features and target.
    print("\n[2/7] Preparing features and target...")
    prepared = prepare_dataset(
        frame,
        target_column=TARGET,
        task="regression",
        test_size=TEST_SIZE,
        random_seed=SEED,
    )

    print(f"Training rows: {len(prepared.X_train)}")
    print(f"Test rows: {len(prepared.X_test)}")

    # Reconstruct raw test features using the original row indices.
    raw_test_features = raw_frame.loc[
        prepared.y_test.index,
        list(prepared.feature_columns),
    ]

    # 3. Train and evaluate baseline models.
    print("\n[3/7] Training models...")
    results = evaluate_regression_models(
        prepared,
        random_seed=SEED,
    )

    baseline_result = evaluate_regression_baseline(prepared)
    results[baseline_result.model_name] = baseline_result

    # 4. Compare model performance.
    print("\n[4/7] Comparing models...")
    comparison = compare_regression_results(results, sort_by="rmse")
    print(comparison.to_string(index=False))

    # 5. Save experiment metadata and metrics.
    print("\n[5/7] Saving experiment to SQLite...")
    metadata = build_experiment_metadata(
        dataset_path=str(DATASET),
        experiment_name="California Housing - Classical Baselines",
        random_seed=SEED,
        parameters={
            "target_column": TARGET,
            "task": "regression",
            "test_size": TEST_SIZE,
            "models": list(results.keys()),
        },
    )

    experiment_id = save_experiment(metadata, results)
    print(f"Saved experiment ID: {experiment_id}")

    # 6. Save checkpoints, predictions, and verify results.
    print("\n[6/7] Saving checkpoints and predictions...")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    expected_prediction_count = len(prepared.y_test)

    for model_name, result in results.items():
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("/", "_")
        )

        checkpoint_path = CHECKPOINT_DIR / (
            f"experiment_{experiment_id}_{safe_name}.joblib"
        )

        save_checkpoint(
            model=result.model,
            preprocessor=prepared.preprocessor,
            feature_columns=prepared.feature_columns,
            target_column=TARGET,
            task="regression",
            checkpoint_path=checkpoint_path,
            metrics=result.metrics,
            experiment_id=experiment_id,
            model_name=model_name,
        )

        # Predict from raw features. The checkpoint applies the
        # fitted preprocessing pipeline internally.
        restored_predictions = predict_with_checkpoint(
            checkpoint_path,
            raw_test_features,
        )

        # Ensure the reloaded model matches the original predictions.
        np.testing.assert_allclose(
            restored_predictions,
            np.asarray(result.predictions),
            rtol=1e-7,
            atol=1e-8,
        )

        # Save individual predictions and residuals to SQLite.
        prediction_count = save_model_predictions(
            experiment_id=experiment_id,
            model_name=model_name,
            sample_indices=prepared.y_test.index.tolist(),
            actual_values=prepared.y_test.to_numpy(),
            predicted_values=result.predictions,
        )

        if prediction_count != expected_prediction_count:
            raise RuntimeError(
                f"Incorrect prediction count for {model_name}: "
                f"expected {expected_prediction_count}, "
                f"got {prediction_count}."
            )

        print(f"Verified: {model_name}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Saved predictions: {prediction_count}")

    # 7. Retrieve the experiment and verify persisted records.
    print("\n[7/7] Verifying saved experiment...")
    record = get_experiment(experiment_id)

    if record is None:
        raise RuntimeError("Saved experiment could not be retrieved.")

    if len(record["model_results"]) != len(results):
        raise RuntimeError(
            "The retrieved model-result count is incorrect."
        )

    for item in record["model_results"]:
        model_name = item["model_name"]
        checkpoint = item["checkpoint_path"]

        if not checkpoint or not Path(checkpoint).is_file():
            raise RuntimeError(
                f"Missing checkpoint link or file for {model_name}."
            )

        # Retrieve and verify persisted predictions.
        predictions = get_model_predictions(
            experiment_id=experiment_id,
            model_name=model_name,
        )

        if len(predictions) != expected_prediction_count:
            raise RuntimeError(
                f"Incorrect saved prediction count for {model_name}: "
                f"expected {expected_prediction_count}, "
                f"got {len(predictions)}."
            )

        print(
            f"  {model_name}: "
            f"RMSE={item['rmse']:.4f} | "
            f"Checkpoint linked | "
            f"Predictions={len(predictions)}"
        )

    print("\nSUCCESS: full experiment workflow completed.")
    print(f"Experiment ID: {experiment_id}")
    print(f"Dataset SHA-256: {record['dataset_sha256']}")
    print(f"Random seed: {record['random_seed']}")
    print(
        "Total saved predictions: "
        f"{expected_prediction_count * len(results)}"
    )


if __name__ == "__main__":
    main()
