
"""Dashboard visual styling."""

import streamlit as st


def apply_dashboard_styles():
    """Apply the global dashboard CSS."""

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #080D1B;
            color: #E8EDFF;
        }

        [data-testid="stSidebar"] {
            background-color: #0D1426;
            border-right: 1px solid #202C47;
        }

        [data-testid="stMetric"] {
            background-color: #111C32;
            border: 1px solid #263653;
            padding: 18px;
            border-radius: 10px;
        }

        [data-testid="stMetricLabel"] {
            color: #A8B7D8;
        }

        [data-testid="stMetricValue"] {
            color: #FFFFFF;
        }

        .main-title {
            font-size: 30px;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 0;
        }

        .subtitle {
            color: #93A5C7;
            font-size: 14px;
            margin-top: 4px;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #263653;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
