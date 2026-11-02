"""Experiment explorer page."""

import pandas as pd
import streamlit as st

from components.layout import show_page_header
from services.dashboard_data import load_cv_results


def render_experiments(experiments, experiment_options, experiment_labels):
    """Render persisted experiment details."""

    show_page_header(
        "Experiment Explorer",
        "Inspect persisted experiments, parameters, environments, and results.",
    )

    selected_label = st.selectbox(
        "Select experiment",
        experiment_labels,
    )

    experiment_id = experiment_options[selected_label]

    experiment = next(
        item for item in experiments
        if item["id"] == experiment_id
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Experiment ID", experiment["id"])
    col2.metric("Random seed", experiment["random_seed"])
    col3.metric(
        "Models evaluated",
        len(experiment.get("model_results", [])),
    )

    st.subheader("Experiment Metadata")

    metadata_col1, metadata_col2 = st.columns(2)

    with metadata_col1:
        st.write("**Dataset**")
        st.code(experiment["dataset_path"])

        st.write("**Dataset SHA-256**")
        st.code(experiment["dataset_sha256"])

    with metadata_col2:
        st.write("**Created at**")
        st.write(experiment["created_at"])

        st.write("**Parameters**")
        st.json(experiment["parameters"])

    with st.expander("Environment information"):
        st.json(experiment["environment"])

    st.subheader("Model Results")

    results = experiment.get("model_results", [])

    if results:
        st.dataframe(
            pd.DataFrame(results),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No model results available.")

    cv = load_cv_results(experiment_id)

    if cv and cv.get("summary"):
        st.subheader("Cross-validation Results")

        st.dataframe(
            pd.DataFrame(cv["summary"]),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("View individual folds"):
            st.dataframe(
                pd.DataFrame(cv["folds"]),
                use_container_width=True,
                hide_index=True,
            )
