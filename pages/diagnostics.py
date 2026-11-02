"""Residual diagnostics page."""

import streamlit as st

from components.layout import show_page_header, format_metric
from services.dashboard_data import load_residual_analysis


def render_diagnostics(experiments, reports_dir):
    """Render residual analysis and diagnostic plots."""

    show_page_header(
        "Residual Diagnostics",
        "Inspect model errors, residual distributions, and diagnostic plots.",
    )

    diagnostic_experiments = [
        item for item in experiments
        if (
            reports_dir / f"experiment_{item['id']}_diagnostics"
        ).is_dir()
    ]

    if not diagnostic_experiments:
        st.info("No diagnostic plot directories found.")
        return

    diagnostic_options = {
        f"Experiment {item['id']}": item["id"]
        for item in diagnostic_experiments
    }

    selected = st.selectbox(
        "Diagnostic experiment",
        list(diagnostic_options.keys()),
    )

    experiment_id = diagnostic_options[selected]

    try:
        analysis = load_residual_analysis(experiment_id)
    except ValueError as exc:
        st.warning(str(exc))
        return

    for model_name, metrics in analysis["models"].items():

        st.subheader(model_name)

        cols = st.columns(4)

        cols[0].metric("Samples", metrics["sample_count"])
        cols[1].metric("MAE", format_metric(metrics["mae"]))
        cols[2].metric("RMSE", format_metric(metrics["rmse"]))
        cols[3].metric(
            "Mean residual",
            format_metric(metrics["mean_residual"]),
        )

        with st.expander("Detailed residual statistics"):
            st.json(metrics)

        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("/", "_")
        )

        model_dir = (
            reports_dir
            / f"experiment_{experiment_id}_diagnostics"
            / safe_name
        )

        plot_names = {
            "Actual vs Predicted": "actual_vs_predicted.png",
            "Residual Distribution": "residual_distribution.png",
            "Residuals vs Predicted": "residuals_vs_predicted.png",
            "Normal Q-Q Plot": "qq_plot.png",
            "Error by Target Range": "error_by_target_range.png",
        }

        for title, filename in plot_names.items():
            image_path = model_dir / filename

            if image_path.is_file():
                st.markdown(f"**{title}**")
                st.image(str(image_path), use_container_width=True)
