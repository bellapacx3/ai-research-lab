"""Dataset quality summary component."""

import streamlit as st


def render_data_quality(validation: dict, profile: dict):
    """Display dataset validation and profiling results."""

    st.subheader("Dataset quality")

    columns = st.columns(4)

    columns[0].metric(
        "Rows",
        f"{validation.get('row_count', 0):,}",
    )

    columns[1].metric(
        "Columns",
        f"{validation.get('column_count', 0):,}",
    )

    columns[2].metric(
        "Missing cells",
        f"{validation.get('missing_cells', 0):,}",
    )

    columns[3].metric(
        "Duplicate rows",
        f"{validation.get('duplicate_rows', 0):,}",
    )

    duplicate_columns = validation.get("duplicate_columns", [])
    empty_columns = validation.get("empty_columns", [])
    constant_columns = validation.get("constant_columns", [])

    if duplicate_columns:
        st.error(
            "Duplicate column names detected. Rename them before training: "
            + ", ".join(map(str, duplicate_columns))
        )

    if empty_columns:
        st.warning(
            "Completely empty columns: "
            + ", ".join(map(str, empty_columns))
        )

    if constant_columns:
        st.info(
            "Constant-value columns: "
            + ", ".join(map(str, constant_columns))
        )

    missing_by_column = validation.get("missing_by_column", {})

    if missing_by_column:
        missing_items = {
            name: count
            for name, count in missing_by_column.items()
            if count
        }

        if missing_items:
            st.markdown("**Missing values by column**")
            st.dataframe(
                [
                    {"Column": name, "Missing values": count}
                    for name, count in missing_items.items()
                ],
                use_container_width=True,
                hide_index=True,
            )

    numeric_columns = profile.get("numeric_columns", [])
    categorical_columns = profile.get("categorical_columns", [])
    datetime_columns = profile.get("datetime_columns", [])

    with st.expander("Column profiling"):
        col1, col2, col3 = st.columns(3)

        col1.write("**Numeric columns**")
        col1.write(numeric_columns or "None")

        col2.write("**Categorical columns**")
        col2.write(categorical_columns or "None")

        col3.write("**Datetime columns**")
        col3.write(datetime_columns or "None")

        numeric_summary = profile.get("numeric_summary")

        if numeric_summary:
            st.markdown("**Numeric statistics**")
            st.dataframe(
                numeric_summary,
                use_container_width=True,
            )
