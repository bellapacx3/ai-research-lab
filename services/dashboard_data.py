"""Cached data access for the Streamlit dashboard."""

import streamlit as st

from config.settings import DB_PATH

from src.storage.repositories import (
    list_experiments,
    get_cross_validation_results,
    get_model_predictions,
)

from src.statistics.residual_analysis import (
    analyze_experiment_residuals,
)


@st.cache_data(ttl=30)
def load_experiments():
    return list_experiments(
        db_path=DB_PATH,
        limit=100,
    )


@st.cache_data(ttl=30)
def load_cv_results(experiment_id):
    return get_cross_validation_results(
        experiment_id=experiment_id,
        db_path=DB_PATH,
    )


@st.cache_data(ttl=30)
def load_predictions(experiment_id, model_name=None):
    return get_model_predictions(
        experiment_id=experiment_id,
        model_name=model_name,
        db_path=DB_PATH,
    )


@st.cache_data(ttl=30)
def load_residual_analysis(experiment_id):
    return analyze_experiment_residuals(
        experiment_id=experiment_id,
        db_path=DB_PATH,
    )
