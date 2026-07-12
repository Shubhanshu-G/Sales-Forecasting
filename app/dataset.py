"""
Dataset Explorer page for the Superstore Sales Streamlit application.

This module details dataset characteristics including shapes, previews, dtypes,
statistical summaries, schema profiles, and sample download capabilities.
"""

import streamlit as st
import pandas as pd
from app.utils import load_data

def render_dataset_page() -> None:
    """Render the Dataset Explorer page to profile and preview the raw sales dataset."""
    st.title("Dataset Explorer")
    st.markdown(
        "This dataset contains 9,800 historical sales transactions from a United States retail superstore "
        "covering a four-year period (2015-2018). It includes key attributes regarding product categories, "
        "customer segments, regional divisions, order timelines, and sales values. It serves as the "
        "foundational source of truth for all downstream time series forecasting, weekly anomaly isolation, "
        "and product clustering models."
    )
    st.write(
        "Profile and preview the underlying transactional sales dataset. "
        "Review data types, missing records, duplicates, and summaries below."
    )

    # 1. Load Data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    # 2. Shape and Summary metrics (no emojis)
    st.subheader("Dataset Shape and Quality Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", f"{df.shape[1]:,}")
    with col3:
        null_count = int(df["Postal Code"].isnull().sum())
        st.metric("Missing Values", f"{null_count}")
    with col4:
        duplicate_count = int(df.duplicated().sum())
        st.metric("Duplicate Rows", f"{duplicate_count}")

    st.write("---")

    # 3. Interactive Dataframe Preview
    st.subheader("Data Preview")
    rows_to_show = st.slider("Select number of rows to preview", min_value=5, max_value=50, value=10)
    st.dataframe(df.head(rows_to_show), use_container_width=True)

    # Download Button
    csv_bytes = df.head(100).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Sample CSV (First 100 Rows)",
        data=csv_bytes,
        file_name="superstore_sample_100.csv",
        mime="text/csv"
    )

    st.write("---")

    # 4. Schema and Dtypes
    st.subheader("Dataset Schema")
    schema_records = []
    for col in df.columns:
        schema_records.append({
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Non-Null Count": df[col].notnull().sum(),
            "Null Count": df[col].isnull().sum(),
            "Unique Values": df[col].nunique()
        })
    df_schema = pd.DataFrame(schema_records)
    st.dataframe(df_schema, use_container_width=True, hide_index=True)

    st.write("---")

    # 5. Numeric and Categorical Summaries
    st.subheader("Statistical Summaries")
    tab_num, tab_cat = st.tabs(["Numerical Features", "Categorical Features"])
    
    with tab_num:
        st.write("Descriptive statistics for numerical variables:")
        st.dataframe(df.describe(), use_container_width=True)
        
    with tab_cat:
        st.write("Descriptive statistics for categorical variables:")
        cat_cols = df.select_dtypes(include=["object"])
        if not cat_cols.empty:
            st.dataframe(cat_cols.describe(), use_container_width=True)
        else:
            st.info("No categorical columns detected.")
