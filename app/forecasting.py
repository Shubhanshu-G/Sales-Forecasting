"""
Forecast Explorer page for the Superstore Sales Streamlit application.

This module renders predictions, forecast tables, and performance evaluations
for Tuned XGBoost, SARIMA, and Prophet. It supports segment filtering and
horizon controls.
"""

import streamlit as st
import pandas as pd
from app.utils import MODEL_METRICS, load_data
from app.charts import display_static_chart, plot_dynamic_forecast

# Static dictionary of pre-computed forecasts from the notebook
# Oct, Nov, Dec 2018 predictions
FORECAST_VALUES = {
    "Global": {
        "SARIMA": [75881.02, 66833.84, 58865.35],
        "Prophet": [72019.23, 90838.29, 104188.75],
        "Tuned XGBoost": [82906.91, 108126.83, 89849.20]
    },
    "Furniture": {
        "Tuned XGBoost": [27243.60, 32842.11, 28155.03]
    },
    "Technology": {
        "Tuned XGBoost": [28495.20, 41563.81, 31140.51]
    },
    "Office Supplies": {
        "Tuned XGBoost": [25690.11, 32713.80, 27858.44]
    },
    "West": {
        "Tuned XGBoost": [29842.50, 43156.11, 33248.60]
    },
    "East": {
        "Tuned XGBoost": [25123.40, 24842.11, 18123.40]
    }
}

def render_forecasting_page() -> None:
    """Render the Forecast Explorer page to evaluate global monthly forecasts for XGBoost, SARIMA, and Prophet."""
    st.title("Forecast Explorer")
    st.write(
        "Evaluate time series forecasts. Adjust segment filters and the horizon slider "
        "to check projected sales numbers."
    )

    # 1. Load data for dynamic plotting
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    # Define default forecast settings (Global, 3-Month horizon)
    segment = "Global"
    horizon = 3

    st.write("---")

    # 3. Tabs for models (Tuned XGBoost is first to ensure segment changes are immediately visible)
    tab_names = ["Tuned XGBoost", "SARIMA", "Prophet"]
    tabs = st.tabs(tab_names)

    # Date range for predictions
    dates = ["2018-10-31", "2018-11-30", "2018-12-31"]

    for tab, model in zip(tabs, tab_names):
        with tab:
            if model == "Tuned XGBoost":
                st.success("Tuned XGBoost is recommended for production deployment due to its superior error reduction.")
            
            # Model Explanation (1-paragraph)
            if model == "SARIMA":
                explanation = (
                    "SARIMA (Seasonal Autoregressive Integrated Moving Average) is a statistical model that "
                    "forecasts future values using linear combinations of past observations, errors, and seasonal patterns. "
                    "Here it was trained with order=(1,0,0) and seasonal_order=(0,0,0,12) on monthly sales. "
                    "It acts as a linear baseline but fails to adjust to sharp, sudden spikes."
                )
            elif model == "Prophet":
                explanation = (
                    "Meta Prophet is an additive regression model designed to capture non-linear trends, "
                    "yearly/weekly seasonalities, and holiday shifts. It is highly robust to missing data and outliers. "
                    "While it successfully models long-term trend components, it tends to smooth out local fluctuations."
                )
            else:
                explanation = (
                    "Tuned XGBoost models time series forecasting as a supervised learning task. Features include "
                    "lags (1, 2, 3 months), a 3-month rolling mean, and calendar components (Month, Quarter, Season). "
                    "Its gradient-boosted decision trees capture complex non-linear relationships, yielding the highest accuracy."
                )
                
            st.markdown(f"**Model Explanation**: {explanation}")
            
            # Metrics
            st.markdown("#### Performance Metrics")
            m = MODEL_METRICS[model]
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("MAE", f"{m['MAE']:,}")
            with col_m2:
                st.metric("RMSE", f"{m['RMSE']:,}")
            with col_m3:
                st.metric("MAPE (%)", f"{m['MAPE (%)']:.2f}%")
            with col_m4:
                st.metric("R² Score", f"{m['R² Score']:.3f}")

            # Advantages & Limitations
            st.markdown("#### Pros and Cons")
            col_adv, col_lim = st.columns(2)
            with col_adv:
                st.markdown("**Advantages**")
                if model == "SARIMA":
                    st.write("- Well-established mathematical baseline")
                    st.write("- Models series on its own history without covariates")
                elif model == "Prophet":
                    st.write("- Resilient to outliers and missing records")
                    st.write("- Interpretable components (trend, yearly patterns)")
                else:
                    st.write("- Outperforms statistical models on complex data")
                    st.write("- Adapts dynamically using rolling average features")
            with col_lim:
                st.markdown("**Limitations**")
                if model == "SARIMA":
                    st.write("- Assumes linear relationships")
                    st.write("- Struggles to adapt to sudden demand jumps")
                elif model == "Prophet":
                    st.write("- Prone to over-smoothing high-volatility peaks")
                    st.write("- Sensitive to parameter initialization")
                else:
                    st.write("- Cannot extrapolate trends without target de-trending")
                    st.write("- Demands extensive feature engineering")

            st.write("---")

            # Determine query segment for predictions
            # If the selected model is SARIMA or Prophet, and a segment is chosen,
            # we fall back to displaying the Global model forecast and show a disclaimer.
            if model in ["SARIMA", "Prophet"] and segment != "Global":
                st.info(
                    f"{model} is modeled at the global level. "
                    f"Displaying global forecast below."
                )
                query_segment = "Global"
            else:
                query_segment = segment

            # Retrieve predictions
            if model in FORECAST_VALUES[query_segment]:
                st.markdown("#### Projected Sales Values")
                pred_sales = FORECAST_VALUES[query_segment][model][:horizon]
                df_pred = pd.DataFrame({
                    "Date": dates[:horizon],
                    "Projected Sales ($)": pred_sales
                })
                st.dataframe(df_pred, use_container_width=True, hide_index=True)
                
                st.markdown("#### Forecast Chart")
                # Plot interactive dynamic line chart
                plot_dynamic_forecast(df, query_segment, FORECAST_VALUES[query_segment][model], horizon)
                
                # Provide pre-rendered static plot as an expandable backup
                # Provide pre-rendered static plot directly
                if query_segment == "Global":
                    if model == "SARIMA":
                        display_static_chart("SARIMA forecast vs monthly sales.png", "SARIMA Monthly Forecast")
                    elif model == "Prophet":
                        display_static_chart("prophet forecast.png", "Prophet Monthly Forecast")
                    else:
                        display_static_chart("tuned xgb forecast.png", "Tuned XGBoost Monthly Forecast")
                else:
                    display_static_chart("sale forecast by category and region.png", f"Segment Forecast: {query_segment}")
            else:
                st.warning("No forecast data available for the selected settings.")
