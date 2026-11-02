"""CSV dataset upload and ingestion component."""

import hashlib
import re
from pathlib import Path

import streamlit as st

from src.data.ingestion import (
    DatasetError,
    load_dataset,
    profile_dataframe,
    validate_dataframe,
)


def render_dataset_uploader(dataset_dir: Path):
    """Render CSV uploader and return dataset information."""

    st.subheader("Upload research dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV dataset",
        type=["csv"],
        help="Only CSV files are supported for dashboard uploads.",
        key="research_dataset_upload",
    )

    if uploaded_file is None:
        return None

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        st.error("The uploaded file is empty.")
        return None

    dataset_dir.mkdir(parents=True, exist_ok=True)

    sha256 = hashlib.sha256(file_bytes).hexdigest()

    original_name = Path(uploaded_file.name).stem
    safe_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        original_name,
    ).strip("_") or "dataset"

    dataset_path = dataset_dir / f"{safe_name}_{sha256[:10]}.csv"

    try:
        if not dataset_path.exists():
            dataset_path.write_bytes(file_bytes)

        frame = load_dataset(dataset_path)
        validation = validate_dataframe(frame)
        profile = profile_dataframe(frame)

        return {
            "frame": frame,
            "path": dataset_path,
            "sha256": sha256,
            "validation": validation,
            "profile": profile,
            "original_name": uploaded_file.name,
        }

    except (DatasetError, ValueError, OSError) as exc:
        st.error(f"Unable to process this dataset: {exc}")
        return None
    except Exception as exc:
        st.error(f"Unexpected dataset processing error: {exc}")
        return None
