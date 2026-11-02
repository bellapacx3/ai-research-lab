"""Shared dashboard layout components and formatting helpers."""

import json
from pathlib import Path

import streamlit as st


WORKSPACE_PAGES = [
    "Overview",
    "New Experiment",
    "Experiments",
    "Model Comparison",
    "Diagnostics",
    "Predictions",
    "Datasets & Reports",
]


def show_page_header(title, description):
    """Render a consistent page heading."""

    st.markdown(
        f'<p class="main-title">{title}</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<p class="subtitle">{description}</p>',
        unsafe_allow_html=True,
    )

    st.divider()


def format_metric(value, digits=4):
    """Format a metric safely."""

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def load_json(path):
    """Load a JSON file, returning None if it does not exist."""

    path = Path(path)

    if not path.is_file():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def render_sidebar():
    """Render the main navigation and return the selected page."""

    with st.sidebar:
        st.markdown("## 🧪 AI RESEARCH LAB")
        st.caption("Experimental Research Environment")

        st.divider()

        page = st.radio(
            "WORKSPACE",
            WORKSPACE_PAGES,
        )

        st.divider()

        st.caption("SYSTEM")
        st.caption("Storage: SQLite")
        st.caption("Interface: Streamlit")

        if st.button("Refresh dashboard", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    return page
