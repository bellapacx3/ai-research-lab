
"""Central application configuration and project paths."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
MODELS_DIR = ARTIFACTS_DIR / "models"

DATA_DIR = ROOT_DIR / "data"
DATASET_DIR = DATA_DIR / "datasets"

DB_PATH = ARTIFACTS_DIR / "research_lab.db"


def configure_page():
    """Configure the Streamlit page. Call before rendering UI."""
    import streamlit as st

    st.set_page_config(
        page_title="AI Research Lab",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
