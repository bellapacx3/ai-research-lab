
"""New experiment — dataset upload, configuration, training, and results."""

import pandas as pd
import streamlit as st

from components.dataset_uploader import render_dataset_uploader
from components.data_quality import render_data_quality
from components.dataset_preview import render_dataset_preview


def _render_experiment_results(result):
    """Display persisted experiment results."""

    st.divider()
    st.subheader("Experiment results")

    experiment_id = result.get("experiment_id")
    if experiment_id is not None:
        st.success(f"Experiment completed successfully. ID: {experiment_id}")

    model_results = result.get(
        "model_results",
        result.get("metrics", {}),
    )

    if model_results:
        st.markdown("**Model performance**")

        metrics_frame = pd.DataFrame.from_dict(
            model_results,
            orient="index",
        )
        metrics_frame.index.name = "Model"
        metrics_frame = metrics_frame.reset_index()

        st.dataframe(
            metrics_frame,
            use_container_width=True,
            hide_index=True,
        )

    prediction_counts = result.get("prediction_counts", {})

    if prediction_counts:
        st.markdown("**Saved predictions**")

        prediction_frame = pd.DataFrame(
            [
                {
                    "Model": model_name,
                    "Saved predictions": count,
                }
                for model_name, count in prediction_counts.items()
            ]
        )

        st.dataframe(
            prediction_frame,
            use_container_width=True,
            hide_index=True,
        )

    checkpoint_paths = result.get("checkpoint_paths", {})

    if checkpoint_paths:
        with st.expander("Saved model checkpoints"):
            for model_name, checkpoint_path in checkpoint_paths.items():
                st.write(f"**{model_name}**")
                st.code(str(checkpoint_path), language=None)


def render_new_experiment(dataset_dir):
    st.markdown(
        '<p class="main-title">New Experiment</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">'
        'Upload, inspect, configure, and run a machine learning experiment.'
        '</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    dataset_info = render_dataset_uploader(dataset_dir)

    if dataset_info is None:
        st.info("Upload a CSV dataset to begin configuring an experiment.")
        return

    frame = dataset_info["frame"]
    validation = dataset_info["validation"]
    profile = dataset_info["profile"]

    st.success(f"Dataset loaded: {dataset_info['original_name']}")
    st.caption(f"Stored at: {dataset_info['path']}")

    render_data_quality(validation, profile)

    st.divider()
    render_dataset_preview(frame)
    st.divider()

    st.subheader("Experiment configuration")

    if validation.get("duplicate_columns"):
        st.error(
            "Experiment configuration is unavailable until duplicate "
            "column names are corrected."
        )
        return

    if frame.empty or len(frame.columns) < 2:
        st.error(
            "The dataset needs at least one feature column and one target column."
        )
        return

    target_column = st.selectbox(
        "Target column",
        options=list(frame.columns),
        key="new_experiment_target",
    )

    feature_candidates = [
        column for column in frame.columns if column != target_column
    ]

    selected_features = st.multiselect(
        "Feature columns",
        options=feature_candidates,
        default=feature_candidates,
        key="new_experiment_features",
        help="Select the columns used to predict the target.",
    )

    col1, col2 = st.columns(2)

    with col1:
        experiment_name = st.text_input(
            "Experiment name",
            value="New regression experiment",
            key="new_experiment_name",
        )

        random_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
            key="new_experiment_seed",
        )

    with col2:
        test_size = st.slider(
            "Test dataset proportion",
            min_value=0.10,
            max_value=0.40,
            value=0.20,
            step=0.05,
            key="new_experiment_test_size",
        )

        training_mode = st.radio(
            "Evaluation mode",
            ["Train / Test Split", "Cross-validation"],
            key="new_experiment_mode",
        )

    if training_mode == "Cross-validation":
        n_splits = st.number_input(
            "Number of folds",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
            key="new_experiment_folds",
        )
    else:
        n_splits = None

    st.markdown("**Models**")

    selected_models = st.multiselect(
        "Select regression models",
        options=[
            "Linear Regression",
            "Random Forest",
            "Mean Baseline",
            "PyTorch Neural Network",
        ],
        default=[
            "Linear Regression",
            "Random Forest",
            "Mean Baseline",
        ],
        key="new_experiment_models",
    )

    random_forest_estimators = st.number_input(
        "Random Forest estimators",
        min_value=10,
        max_value=1000,
        value=200,
        step=10,
        key="new_experiment_estimators",
        disabled="Random Forest" not in selected_models,
    )

    nn_epochs = 200
    nn_learning_rate = 0.001
    nn_batch_size = 16
    nn_patience = 25

    if "PyTorch Neural Network" in selected_models:
        with st.expander("Neural network settings", expanded=True):
            nn_col1, nn_col2 = st.columns(2)

            with nn_col1:
                nn_epochs = st.number_input(
                    "Training epochs",
                    min_value=5,
                    max_value=2000,
                    value=200,
                    step=5,
                    key="new_experiment_nn_epochs",
                )

                nn_batch_size = st.number_input(
                    "Batch size",
                    min_value=1,
                    max_value=512,
                    value=16,
                    step=1,
                    key="new_experiment_nn_batch_size",
                )

            with nn_col2:
                nn_learning_rate = st.number_input(
                    "Learning rate",
                    min_value=0.00001,
                    max_value=0.1,
                    value=0.001,
                    step=0.0001,
                    format="%.5f",
                    key="new_experiment_nn_learning_rate",
                )

                nn_patience = st.number_input(
                    "Early stopping patience",
                    min_value=1,
                    max_value=200,
                    value=25,
                    step=1,
                    key="new_experiment_nn_patience",
                )

    st.divider()

    config = {
        "dataset_path": str(dataset_info["path"]),
        "dataset_sha256": dataset_info["sha256"],
        "target_column": target_column,
        "feature_columns": selected_features,
        "experiment_name": experiment_name.strip(),
        "random_seed": int(random_seed),
        "test_size": float(test_size),
        "training_mode": training_mode,
        "n_splits": int(n_splits) if n_splits is not None else None,
        "models": selected_models,
        "random_forest_estimators": int(random_forest_estimators),
        "nn_epochs": int(nn_epochs),
        "nn_learning_rate": float(nn_learning_rate),
        "nn_batch_size": int(nn_batch_size),
        "nn_patience": int(nn_patience),
    }

    st.session_state["new_experiment_config"] = config

    st.write("**Configuration summary**")
    st.write({
        "Dataset": dataset_info["original_name"],
        "Target": target_column,
        "Features": selected_features,
        "Evaluation mode": training_mode,
        "Models": selected_models,
        "Random seed": random_seed,
    })

    valid_features = bool(selected_features)
    valid_models = bool(selected_models)
    valid_name = bool(experiment_name.strip())
    numeric_target = pd.api.types.is_numeric_dtype(frame[target_column])

    if not valid_features:
        st.warning("Select at least one feature column.")

    if not valid_models:
        st.warning("Select at least one model.")

    if not valid_name:
        st.warning("Enter an experiment name.")

    if not numeric_target:
        st.warning(
            "The selected target column is not numeric. "
            "The current experiment runner supports regression with numeric targets."
        )

    if training_mode == "Cross-validation":
        st.info(
            "Cross-validation configuration is available, but its execution "
            "workflow has not yet been connected. Select Train / Test Split "
            "to run an experiment."
        )

    st.divider()

    can_run = (
        valid_features
        and valid_models
        and valid_name
        and numeric_target
        and training_mode == "Train / Test Split"
    )

    if st.button(
        "Run Experiment",
        type="primary",
        use_container_width=True,
        disabled=not can_run,
    ):
        try:
            from services.experiment_runner import run_regression_experiment

            with st.spinner(
                "Training models, evaluating predictions, and saving results..."
            ):
                result = run_regression_experiment(
                    frame=frame,
                    dataset_path=dataset_info["path"],
                    target_column=target_column,
                    feature_columns=selected_features,
                    experiment_name=experiment_name.strip(),
                    random_seed=int(random_seed),
                    test_size=float(test_size),
                    selected_models=selected_models,
                    random_forest_estimators=int(random_forest_estimators),
                    nn_epochs=int(nn_epochs),
                    nn_learning_rate=float(nn_learning_rate),
                    nn_batch_size=int(nn_batch_size),
                    nn_patience=int(nn_patience),
                )

            st.session_state["last_experiment_result"] = {
                "config": config,
                "result": result,
            }

        except Exception as exc:
            st.error(f"Experiment failed: {exc}")
            st.exception(exc)

    last_run = st.session_state.get("last_experiment_result")

    if last_run and last_run.get("config") == config:
        _render_experiment_results(last_run["result"])
