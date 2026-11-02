"""AI Research Lab — Streamlit application entry point."""

import streamlit as st

from config.settings import (
    configure_page,
    REPORTS_DIR,
    DATASET_DIR,
)

from components.styling import apply_dashboard_styles
from components.layout import show_page_header, render_sidebar

from services.dashboard_data import load_experiments

from pages.new_experiment import render_new_experiment
from pages.overview import render_overview
from pages.experiments import render_experiments
from pages.model_comparison import render_model_comparison
from pages.diagnostics import render_diagnostics
from pages.predictions import render_predictions
from pages.datasets_reports import render_datasets_reports


def main():
    """Initialize the application and render the selected page."""

    configure_page()
    apply_dashboard_styles()

    page = render_sidebar()

    try:
        experiments = load_experiments()
    except Exception as exc:
        st.error(f"Unable to load experiment database: {exc}")
        experiments = []

    experiment_options = {
        f"Experiment {item['id']} — {item['experiment_name']}": item["id"]
        for item in experiments
    }

    experiment_labels = list(experiment_options.keys())

    if page == "New Experiment":
        render_new_experiment(DATASET_DIR)

    elif not experiments and page != "Datasets & Reports":
        show_page_header(
            "Research Workspace",
            "Your research environment is ready for its first experiment.",
        )

        st.info(
            "No saved experiments are available yet. "
            "Upload a CSV dataset and configure your first experiment "
            "using the New Experiment workspace."
        )

    elif page == "Overview":
        render_overview(experiments)

    elif page == "Experiments":
        render_experiments(
            experiments,
            experiment_options,
            experiment_labels,
        )

    elif page == "Model Comparison":
        render_model_comparison(
            experiments,
            experiment_options,
            experiment_labels,
        )

    elif page == "Diagnostics":
        render_diagnostics(experiments, REPORTS_DIR)

    elif page == "Predictions":
        render_predictions(
            experiments,
            experiment_options,
            experiment_labels,
        )

    elif page == "Datasets & Reports":
        render_datasets_reports(DATASET_DIR, REPORTS_DIR)


if __name__ == "__main__":
    main()
