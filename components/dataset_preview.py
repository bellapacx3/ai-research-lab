"""Dataset preview component."""

import streamlit as st


def render_dataset_preview(frame):
    """Display a sample of the uploaded dataset."""

    st.subheader("Dataset preview")

    preview_rows = st.select_slider(
        "Preview rows",
        options=[5, 10, 20, 50, 100],
        value=10,
        key="dataset_preview_rows",
    )

    st.dataframe(
        frame.head(preview_rows),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Column information"):
        column_data = []

        for column in frame.columns:
            column_data.append({
                "Column": str(column),
                "Data type": str(frame[column].dtype),
                "Non-null values": int(frame[column].notna().sum()),
                "Unique values": int(frame[column].nunique(dropna=True)),
            })

        st.dataframe(
            column_data,
            use_container_width=True,
            hide_index=True,
        )
