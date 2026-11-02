"""Model comparison page."""

import pandas as pd
import streamlit as st

from components.layout import show_page_header
from services.dashboard_data import load_cv_results


def render_model_comparison(experiments, experiment_options, experiment_labels):
    """Compare persisted model evaluation results."""

    show_page_header(
        "Model Comparison",
        "Compare model performance using persisted evaluation metrics.",
    )

    selected_label = st.selectbox(
        "Select experiment",
        experiment_labels,
        key="comparison_experiment",
    )

    experiment_id = experiment_options[selected_label]

    experiment = next(
        item for item in experiments
        if item["id"] == experiment_id
    )

    results = experiment.get("model_results", [])

    if not results:
        st.info("No model results available.")
        return

    comparison = pd.DataFrame(results).sort_values("rmse")

    st.subheader("Performance Ranking")

    st.dataframe(
        comparison[["model_name", "mae", "rmse", "r2"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("RMSE Comparison")
    st.bar_chart(comparison.set_index("model_name")[["rmse"]])

    st.subheader("MAE Comparison")
    st.bar_chart(comparison.set_index("model_name")[["mae"]])

    best = comparison.iloc[0]

    st.success(
        f"Best model by RMSE: {best['model_name']} "
        f"(RMSE: {best['rmse']:.4f})"
    )

    cv = load_cv_results(experiment_id)

    if cv and cv.get("summary"):
        st.subheader("Cross-validation comparison")

        st.dataframe(
            pd.DataFrame(cv["summary"]),
            use_container_width=True,
            hide_index=True,
        )
