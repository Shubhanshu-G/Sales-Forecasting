"""
Sales Overview page (home) for the Superstore Sales Streamlit application.

This module renders the dashboard landing page containing filters, dynamic KPIs,
and data visualizations of sales history (reusing static figures).
"""

import streamlit as st
import pandas as pd
from app.utils import load_data
from app.charts import display_static_chart, plot_sales_by_year

def render_home_page() -> None:
    """Render the Sales Overview dashboard landing page with dynamic KPIs and charts."""
    st.title("Sales Overview")
    st.write(
        "Interactive dashboard displaying historical Superstore transaction trends, "
        "regional sales profiles, and categorical distributions."
    )

    # 1. Load Data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    # 2. Render Filters (horizontal bar at the top)
    st.markdown("##### Filter Controls")
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        years = sorted(df["Year"].unique())
        selected_years = st.multiselect("Year Selection", options=years, default=years)
        
    with f_col2:
        regions = sorted(df["Region"].unique())
        selected_regions = st.multiselect("Region Selection", options=regions, default=regions)
        
    with f_col3:
        categories = sorted(df["Category"].unique())
        selected_categories = st.multiselect("Category Selection", options=categories, default=categories)

    # Apply filters dynamically (default to all if selection is empty)
    filt_years = selected_years if selected_years else years
    filt_regions = selected_regions if selected_regions else regions
    filt_categories = selected_categories if selected_categories else categories

    filtered_df = df[
        (df["Year"].isin(filt_years)) &
        (df["Region"].isin(filt_regions)) &
        (df["Category"].isin(filt_categories))
    ]

    # 3. Compute Dynamic Metrics
    total_sales = filtered_df["Sales"].sum()
    total_orders = filtered_df["Order ID"].nunique()
    total_customers = filtered_df["Customer ID"].nunique()
    years_covered = filtered_df["Year"].nunique()

    # 4. Display Metrics (clean layout, no emojis)
    st.write("")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Sales", f"${total_sales:,.2f}")
    with m_col2:
        st.metric("Total Orders", f"{total_orders:,}")
    with m_col3:
        st.metric("Total Customers", f"{total_customers:,}")
    with m_col4:
        st.metric("Years Covered", f"{years_covered}")

    st.write("---")

    # 5. Display Charts Grid
    st.subheader("Sales Visualizations")
    grid_col1, grid_col2 = st.columns(2)
    
    with grid_col1:
        # Dynamic bar chart of Sales by Year
        plot_sales_by_year(filtered_df)
        
        # Static Monthly sales trend
        display_static_chart("monthly sales trend.png", "Monthly Sales Trend (Full History)")

    with grid_col2:
        # Static Sales by Region
        display_static_chart("sales growth by region.png", "Sales Growth by Region (Full History)")
        
        # Static Sales by Category
        display_static_chart("revenue by category.png", "Revenue by Product Category (Full History)")
