"""
About Project page for the Superstore Sales Streamlit application.

This module details project objectives, workflows, architectures, libraries,
final comparison tables, key findings, recommendations, and author metadata.
"""

import streamlit as st
import pandas as pd
from app.utils import MODEL_METRICS

def render_about_page() -> None:
    """Render the About Project page containing pipeline workflows, comparative model tables, and findings."""
    st.title("About Project")
    st.write(
        "A detailed overview of the Superstore Sales Forecasting and Product Segmentation project, "
        "covering methodology, model evaluations, findings, and author metadata."
    )

    # 1. Project Objective
    st.subheader("Project Objective")
    st.write(
        "The objective of this project is to build an end-to-end sales intelligence pipeline "
        "for a commercial retail superstore. By applying time series forecasting, anomaly detection, "
        "and product segmentation on historical transaction records, this project provides data-driven "
        "insights to optimize inventory planning, manage demand volatility, and isolate revenue drivers."
    )

    st.write("---")

    # 2. Workflow Diagram (Text-based flowchart)
    st.subheader("Workflow Methodology")
    st.markdown(
        """
        ```
        [Data Ingestion] ────> [Preprocessing & Date Engineering]
                                     │
                                     v
        [Exploratory Data Analysis & Decomposition]
                                     │
                 ┌───────────────────┼───────────────────┐
                 v                   v                   v
          [Forecasting]     [Anomaly Detection]    [Product Clustering]
          • SARIMA          • Isolation Forest     • KMeans
          • Prophet         • Rolling Z-Score      • PCA Projection
          • XGBoost
                 │                   │                   │
                 └───────────────────┼───────────────────┘
                                     v
                           [Dashboard Deployment]
        ```
        """
    )
    st.markdown(
        "1. **Data Ingestion**: Import raw historical transactional records (train.csv).\n"
        "2. **Date Engineering**: Parse formats, compute shipping durations, and extract temporal features (Year, Month, Season).\n"
        "3. **Decomposition**: Apply classical multiplicative decomposition to extract trend, seasonality, and residuals.\n"
        "4. **Modeling**: Execute parallel pipelines for univariate forecasting, outlier isolation, and category clustering.\n"
        "5. **Serialization**: Save the trained models in .pkl format (pre-compiled, no runtime retraining).\n"
        "6. **Dashboard Deployment**: Load serializations in a modular, clean Streamlit application."
    )

    st.write("---")

    # 3. Dataset Summary
    st.subheader("Dataset Summary")
    st.markdown(
        "- **Dataset Name**: US Superstore Sales Dataset\n"
        "- **Timeline Covered**: Jan 2015 to Dec 2018\n"
        "- **Dimensions**: Product Category, Sub-Category, Region, Customer Segment, Order Dates, and Ship Dates.\n"
        "- **Target Variable**: Sales volume in USD ($)"
    )

    st.write("---")

    # 4. Models and Libraries Used
    st.subheader("Machine Learning Models & Libraries")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**Models Implemented:**\n"
            "- **SARIMA**: Order (1,0,0) x (0,0,0)12 statistical baseline.\n"
            "- **Meta Prophet**: Additive time series decomposition.\n"
            "- **Tuned XGBoost**: Supervised learning with recursive lag and rolling mean features.\n"
            "- **Isolation Forest**: Outlier detection on weekly sales distributions.\n"
            "- **K-Means & PCA**: Product demand clustering and 2D representation."
        )
    with col2:
        st.markdown(
            "**Libraries Utilized:**\n"
            "- **streamlit**: Dashboard framework.\n"
            "- **pandas** & **numpy**: Data engineering and linear algebra.\n"
            "- **scikit-learn**: Scaling, KMeans, PCA, and Isolation Forest.\n"
            "- **xgboost**: Tree-based regression.\n"
            "- **statsmodels**: SARIMAX statistical modeling.\n"
            "- **prophet**: Meta time series library."
        )

    st.write("---")

    # 5. Final Model Comparison Table
    st.subheader("Final Model Comparison")
    st.write("Comparative model evaluation on the holdout test set (last 3 months of history):")
    
    metrics_list = []
    for model_name, m in MODEL_METRICS.items():
        metrics_list.append({
            "Model": model_name,
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "MAPE (%)": m["MAPE (%)"],
            "R² Score": m["R² Score"]
        })
    df_compare = pd.DataFrame(metrics_list)
    st.dataframe(df_compare, use_container_width=True, hide_index=True)
    st.write(
        "**Conclusion**: Tuned XGBoost achieves the best performance with the lowest MAE ($13,907.70), "
        "lowest MAPE (14.10%), and the only positive R² score (0.176), making it the recommended forecasting model."
    )

    st.write("---")

    # 6. Key Findings
    st.subheader("Key Findings")
    st.markdown(
        "- **High Seasonality**: Sales show a consistent recurring spike in Q4 (September, November, December).\n"
        "- **Dominant Region**: The West region leads sales growth. The East region exhibits a projected contraction.\n"
        "- **Technology Revenue**: Technology is the largest revenue-producing category ($827K+).\n"
        "- **Anomaly Consistency**: Outlier models matched only on 22 March 2015, showing that Isolation Forest isolating "
        "global extremes and Rolling Z-Score isolating local shocks capture different outlier mechanisms."
    )

    st.write("---")

    # 7. Business Recommendations
    st.subheader("Business Recommendations")
    st.markdown(
        "- **Replenishment Rules**: Maintain lean stock (JIT) for 'Low Volume, Stable' sub-categories (e.g. Envelopes, Labels). "
        "Establish safety stocks for 'Growing High-Value' items (e.g. Copiers) to avoid stockouts during demand volatility.\n"
        "- **Seasonality Prep**: Secure warehouse and distribution capacity by late July to accommodate the massive Q4 seasonal sales spike.\n"
        "- **Regional Shifts**: Reallocate marketing budgets to the West region to capitalize on growth, and review operations "
        "in the East region to reverse predicted sales declines."
    )

    st.write("---")

    # 8. Future Improvements
    st.subheader("Future Improvements")
    st.markdown(
        "1. **Real-time Pipeline**: Link dashboard directly to cloud warehouses (e.g., Snowflake, BigQuery) instead of CSV files.\n"
        "2. **Exogenous Features**: Feed promotional calendars and CPI/GDP data into the XGBoost model to improve accuracy.\n"
        "3. **Customer LTV**: Integrate RFM clustering to segment client accounts."
    )


