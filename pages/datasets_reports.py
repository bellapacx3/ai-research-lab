"""Dataset browser and research reports page."""

import pandas as pd
import streamlit as st

from components.layout import show_page_header, load_json


def render_datasets_reports(dataset_dir, reports_dir):
    """Render available datasets and generated research reports."""

    show_page_header(
        "Datasets & Reports",
        "Browse research datasets, inspect their quality, and review generated reports.",
    )

    # ---------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------
    st.header("Datasets")

    dataset_files = sorted(dataset_dir.glob("*.csv"))

    if not dataset_files:
        st.info("No CSV datasets found.")
    else:
        selected_dataset = st.selectbox(
            "Select dataset",
            dataset_files,
            format_func=lambda path: path.name,
        )

        try:
            dataset = pd.read_csv(selected_dataset)

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", f"{len(dataset):,}")
            col2.metric("Columns", len(dataset.columns))
            col3.metric("Missing values", f"{int(dataset.isna().sum().sum()):,}")

            st.subheader("Dataset Preview")
            st.dataframe(
                dataset.head(100),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("Descriptive Statistics"):
                st.dataframe(
                    dataset.describe(include="all").T,
                    use_container_width=True,
                )

            metadata_path = (
                selected_dataset.parent
                / f"{selected_dataset.stem}_metadata.json"
            )

            if metadata_path.is_file():
                with st.expander("Dataset Metadata"):
                    st.json(load_json(metadata_path))

        except Exception as exc:
            st.error(f"Unable to read dataset: {exc}")

    # ---------------------------------------------------------
    # Reports
    # ---------------------------------------------------------
    st.divider()
    st.header("Research Reports")

    report_files = sorted(
        (
            path for path in reports_dir.glob("*")
            if path.is_file()
        ),
        key=lambda path: path.name.lower(),
    )

    if not report_files:
        st.info("No research reports found.")
        return

    selected_report = st.selectbox(
        "Select report",
        report_files,
        format_func=lambda path: path.name,
    )

    try:
        suffix = selected_report.suffix.lower()

        if suffix in (".md", ".markdown", ".txt"):
            st.markdown(
                selected_report.read_text(encoding="utf-8")
            )

        elif suffix == ".json":
            st.json(load_json(selected_report))

        elif suffix == ".csv":
            report_data = pd.read_csv(selected_report)
            st.dataframe(
                report_data,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info(
                f"Preview is not available for {suffix or 'this file'}."
            )

        st.download_button(
            "Download report",
            data=selected_report.read_bytes(),
            file_name=selected_report.name,
            key=f"download_{selected_report.name}",
        )

    except Exception as exc:
        st.error(f"Unable to display report: {exc}")
