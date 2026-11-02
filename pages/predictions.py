"""Prediction explorer page."""

import pandas as pd
import streamlit as st

from components.layout import show_page_header, format_metric
from services.dashboard_data import load_predictions


def render_predictions(experiments, experiment_options, experiment_labels):
    """Render saved model predictions and prediction errors."""

    show_page_header(
        "Prediction Explorer",
        "Explore actual values, model predictions, and individual errors.",
    )

    selected_label = st.selectbox(
        "Select experiment",
        experiment_labels,
        key="prediction_experiment",
    )

    experiment_id = experiment_options[selected_label]

    predictions = load_predictions(experiment_id)

    if not predictions:
        st.info("No saved predictions available for this experiment.")
        return

    prediction_frame = pd.DataFrame(predictions)

    model_names = sorted(prediction_frame["model_name"].unique())

    selected_model = st.selectbox(
        "Select model",
        model_names,
    )

    model_frame = prediction_frame[
        prediction_frame["model_name"] == selected_model
    ].copy()

    col1, col2, col3 = st.columns(3)

    col1.metric("Prediction samples", len(model_frame))

    col2.metric(
        "MAE",
        format_metric(model_frame["absolute_error"].mean()),
    )

    col3.metric(
        "RMSE",
        format_metric(
            model_frame["squared_error"].mean() ** 0.5
        ),
    )

    st.subheader("Actual vs Predicted")

    st.scatter_chart(
        model_frame,
        x="actual_value",
        y="predicted_value",
    )

    st.subheader("Residuals vs Predicted")

    st.scatter_chart(
        model_frame,
        x="predicted_value",
        y="residual",
    )

    st.subheader("Prediction Records")

    st.dataframe(
        model_frame[
            [
                "sample_index",
                "actual_value",
                "predicted_value",
                "residual",
                "absolute_error",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    csv_data = model_frame.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download prediction records (CSV)",
        data=csv_data,
        file_name=f"experiment_{experiment_id}_{selected_model}.csv",
        mime="text/csv",
    )
