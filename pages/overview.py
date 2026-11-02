"""Research overview page."""

import pandas as pd
import streamlit as st

from components.layout import show_page_header, format_metric


def render_overview(experiments):
    """Render the research overview dashboard."""

    show_page_header(
        "Research Overview",
        "Monitor experiments, model performance, and research activity.",
    )

    total_experiments = len(experiments)

    all_models = [
        result
        for experiment in experiments
        for result in experiment.get("model_results", [])
    ]

    total_models = len(all_models)

    latest = experiments[0]
    latest_results = latest.get("model_results", [])

    best_model = (
        min(latest_results, key=lambda item: item["rmse"])
        if latest_results else None
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Experiments", total_experiments)
    col2.metric("Model evaluations", total_models)
    col3.metric("Latest experiment", f"#{latest['id']}")
    col4.metric(
        "Best latest RMSE",
        format_metric(best_model["rmse"]) if best_model else "N/A",
    )

    st.subheader("Latest Experiment")

    st.write(f"**{latest['experiment_name']}**")
    st.caption(
        f"Created: {latest['created_at']} | "
        f"Dataset: {latest['dataset_path']} | "
        f"Random seed: {latest['random_seed']}"
    )

    if latest_results:
        results_df = pd.DataFrame(latest_results)

        st.dataframe(
            results_df[["model_name", "mae", "rmse", "r2"]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Recent Research Activity")

    history = []

    for experiment in experiments:
        for result in experiment.get("model_results", []):
            history.append({
                "Experiment": experiment["id"],
                "Name": experiment["experiment_name"],
                "Model": result["model_name"],
                "MAE": result["mae"],
                "RMSE": result["rmse"],
                "R²": result["r2"],
                "Date": experiment["created_at"],
            })

    if history:
        st.dataframe(
            pd.DataFrame(history),
            use_container_width=True,
            hide_index=True,
        )
