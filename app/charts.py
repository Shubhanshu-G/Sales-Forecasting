"""
Charts helper module for the Superstore Sales Streamlit application.

This module provides functions for loading and displaying pre-rendered static charts
from the charts/ directory, and generating clean, interactive Plotly visualizations
when dynamic charts are required.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Base charts directory path is determined dynamically
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHARTS_DIR = os.path.join(BASE_DIR, "charts")

def display_static_chart(chart_filename: str, caption: str) -> None:
    """Load a pre-rendered static PNG chart and display it using st.image."""
    chart_path = os.path.join(CHARTS_DIR, chart_filename)
    if os.path.exists(chart_path):
        st.image(chart_path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Static chart not found: {chart_filename}")

def plot_sales_by_year(df: pd.DataFrame) -> None:
    """Generate and render an interactive Plotly bar chart showing Sales by Year."""
    # Group data by Year
    yearly_sales = df.groupby("Year")["Sales"].sum().reset_index()
    yearly_sales["Year"] = yearly_sales["Year"].astype(str)
    
    # Create a clean, minimalist Plotly bar chart
    fig = px.bar(
        yearly_sales,
        x="Year",
        y="Sales",
        text="Sales",
        title="Total Sales by Year",
        labels={"Sales": "Sales ($)", "Year": "Year"},
        color_discrete_sequence=["#1f77b4"]
    )
    
    # Clean up layout to make it feel data-first and premium (Power BI style)
    fig.update_traces(
        texttemplate='$%{text:,.0f}', 
        textposition='outside',
        marker_line_width=0
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=10, r=10),
        xaxis=dict(showgrid=False, linecolor="#ccc"),
        yaxis=dict(showgrid=True, gridcolor="#e9ecef", linecolor="#ccc"),
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_dynamic_forecast(df: pd.DataFrame, segment: str, predictions: list, horizon: int) -> None:
    """Generate and render an interactive Plotly line chart showing historical monthly sales and forecasts."""
    # 1. Filter dataset according to the selected segment
    if segment == "Global":
        filtered_df = df
    elif segment in ["Furniture", "Technology", "Office Supplies"]:
        filtered_df = df[df["Category"] == segment]
    elif segment in ["West", "East"]:
        filtered_df = df[df["Region"] == segment]
    else:
        filtered_df = df

    # 2. Aggregate monthly sales (using pd.Grouper ME for Month End)
    monthly = (
        filtered_df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
        .sum()
        .reset_index()
    )
    
    # Sort dates to ensure sequence is chronologically aligned
    monthly = monthly.sort_values("Order Date").reset_index(drop=True)
    
    # Splitting into historical actuals (all but last 3 months) and holdout actuals (last 3 months)
    # The last 3 rows represent Oct, Nov, Dec 2018
    train_seq = monthly.iloc[:-3].copy()
    test_seq = monthly.iloc[-3:].copy()
    
    # 3. Create Plotly figure
    fig = go.Figure()
    
    # Plot training actuals (Historical Sales)
    fig.add_trace(go.Scatter(
        x=train_seq["Order Date"],
        y=train_seq["Sales"],
        name="Historical Actuals",
        line=dict(color="#1f77b4", width=2.5)
    ))
    
    # Plot test actuals (Holdout Test) up to selected horizon
    fig.add_trace(go.Scatter(
        x=test_seq["Order Date"].iloc[:horizon],
        y=test_seq["Sales"].iloc[:horizon],
        name="Holdout Actuals",
        line=dict(color="#2ca02c", width=2.5)
    ))
    
    # Plot predicted values (Forecast) up to selected horizon
    fig.add_trace(go.Scatter(
        x=test_seq["Order Date"].iloc[:horizon],
        y=predictions[:horizon],
        name="Model Forecast",
        line=dict(color="#d62728", width=2.5, dash="dash")
    ))
    
    # 4. Clean layout styling
    fig.update_layout(
        title=f"Sales Forecast Analysis for {segment}",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=10, l=10, r=10),
        xaxis=dict(showgrid=True, gridcolor="#e9ecef", linecolor="#ccc"),
        yaxis=dict(showgrid=True, gridcolor="#e9ecef", linecolor="#ccc"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
