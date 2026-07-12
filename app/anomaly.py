"""
Anomaly Report page for the Superstore Sales Streamlit application.

This module displays weekly sales anomalies using pre-saved figures
and renders a table of detected outlier weeks.
"""

import streamlit as st
import pandas as pd
from app.utils import load_data
from app.charts import display_static_chart

# Flagged anomaly dates from the notebook results
ISO_FOREST_DATES = [
    "2015-01-04", "2015-02-08", "2015-02-22", "2015-03-22", 
    "2015-07-19", "2015-09-13", "2016-01-24", "2017-12-17", 
    "2018-11-04", "2018-11-18", "2018-12-02"
]
Z_SCORE_DATES = [
    "2015-03-22", "2015-07-26", "2016-08-28", "2016-09-18", 
    "2017-05-28", "2018-03-25"
]

def render_anomaly_page() -> None:
    """Render the Anomaly Report page showing global (Isolation Forest) and local (Z-Score) weekly sales outliers."""
    st.title("Anomaly Report")
    st.write(
        "Review weekly sales anomalies detected using two distinct methods: "
        "Isolation Forest (global distribution) and Rolling Z-Score (local 8-week trend)."
    )

    # 1. Load data and compute weekly sales to get exact sales figures for anomaly dates
    try:
        df = load_data()
        # Group by week matching the notebook's weekly sales logic
        # (Order Date grouped by week using pd.Grouper)
        weekly = (
            df.groupby(pd.Grouper(key="Order Date", freq="W-SUN"))["Sales"]
            .sum()
            .reset_index()
        )
        weekly["Date_Str"] = weekly["Order Date"].dt.strftime("%Y-%m-%d")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    # 2. Build the anomaly table
    anomaly_records = []
    all_anomaly_dates = sorted(list(set(ISO_FOREST_DATES + Z_SCORE_DATES)))
    
    for date_str in all_anomaly_dates:
        # Lookup weekly sales value
        match = weekly[weekly["Date_Str"] == date_str]
        sales_val = float(match["Sales"].values[0]) if not match.empty else 0.0
        
        # Determine detection method
        in_iso = date_str in ISO_FOREST_DATES
        in_z = date_str in Z_SCORE_DATES
        
        if in_iso and in_z:
            method = "Both"
        elif in_iso:
            method = "Isolation Forest"
        else:
            method = "Rolling Z-Score"
            
        anomaly_records.append({
            "Date": date_str,
            "Sales ($)": round(sales_val, 2),
            "Detection Method": method
        })
        
    df_anomalies = pd.DataFrame(anomaly_records)

    # 3. Anomaly Table
    st.subheader("Detected Outliers")
    st.write("Complete list of anomalous weeks identified by the detection systems:")
    st.dataframe(df_anomalies, use_container_width=True, hide_index=True)

    # Short Explanation
    st.write(
        "**Observation**: Only one week (**22 March 2015**) was flagged by both models, "
        "highlighting a major sales spike that was both globally and locally unique. "
        "Isolation Forest flagged more anomalies because it checks the absolute magnitude of sales. "
        "The Rolling Z-Score flagged weeks showing sudden spikes relative to the immediate past 8 weeks."
    )

    st.write("---")

    # 4. Charts Section
    st.subheader("Detection Plots")
    col1, col2 = st.columns(2)
    with col1:
        display_static_chart("iso anomalies.png", "Isolation Forest Global Anomaly Detection")
    with col2:
        display_static_chart("z anomalies.png", "Rolling Z-Score Local Anomaly Detection (2 Std Dev)")

    st.write("")
    display_static_chart("camparison of anomalies.png", "Method Comparison (Overlap on 22-Mar-2015)")

    st.write("---")
    st.subheader("Method Comparison & Key Finding")
    st.markdown(
        "**Critical Intersection (Overlap on 22-Mar-2015)**:\n\n"
        "- **The Event**: The week ending **22 March 2015** is the only date in the entire 4-year timeline flagged "
        "by both detection mechanisms.\n"
        "- **Interpretation**: This overlap confirms that this week contained a massive sales surge ($11,543.60) "
        "that represents both a **global outlier** (a rare event across the whole multi-year dataset distribution, "
        "captured by Isolation Forest) and a **local outlier** (a sudden, sharp surge compared to the adjacent 8 weeks, "
        "captured by Rolling Z-Score).\n"
        "- **Action Item**: Analysts should smooth this event when training baseline forecasts to avoid bias, "
        "and investigate historical promotions or corporate accounts active during this specific week."
    )
