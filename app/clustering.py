"""
Product Demand Segments page for the Superstore Sales Streamlit application.

This module aggregates sub-category sales data, runs inference using pre-saved
KMeans and PCA models, and renders cluster metrics and stocking strategies.
"""

import os
import streamlit as st
import pandas as pd
from app.utils import load_data, load_ml_model
from app.charts import display_static_chart

def render_clustering_page() -> None:
    """Render the Product Demand Segments page showing K-Means cluster analysis, PCA projections, and stocking rules."""
    st.title("Product Demand Segments")
    st.write(
        "Segment product sub-categories into demand classes based on total sales volume, "
        "average transaction values, sales volatility, and year-over-year growth rates."
    )

    # 1. Load Data and Models
    try:
        df = load_data()
        scaler = load_ml_model("scaler.pkl")
        kmeans = load_ml_model("kmeans.pkl")
        pca = load_ml_model("pca.pkl")
    except Exception as e:
        st.error(f"Error loading models or dataset: {e}")
        return

    # 2. Aggregations (matches the notebook clustering logic)
    total_sales = df.groupby("Sub-Category")["Sales"].sum().rename("Total Sales")
    average_order = df.groupby("Sub-Category")["Sales"].mean().rename("Average Order Value")
    
    monthly_sales = (
        df.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="ME")])["Sales"]
        .sum()
        .reset_index()
    )
    volatility = monthly_sales.groupby("Sub-Category")["Sales"].std().rename("Sales Volatility")
    
    yearly_sales = (
        df.groupby(["Sub-Category", "Year"])["Sales"]
        .sum()
        .reset_index()
    )
    growth = yearly_sales.pivot(index="Sub-Category", columns="Year", values="Sales")
    growth["Sales Growth Rate"] = (
        (growth[growth.columns[-1]] - growth[growth.columns[0]]) / growth[growth.columns[0]]
    ) * 100
    
    cluster_df = pd.concat(
        [total_sales, average_order, volatility, growth["Sales Growth Rate"]],
        axis=1
    )
    cluster_df.fillna(0, inplace=True)
    
    # 3. Model Inference (No retraining/fitting)
    features = ["Total Sales", "Average Order Value", "Sales Volatility", "Sales Growth Rate"]
    scaled_data = scaler.transform(cluster_df[features])
    cluster_df["Cluster"] = kmeans.predict(scaled_data)
    
    # Map cluster assignments to segment names
    cluster_names = {
        0: "Growing High-Value Products",
        1: "Low Volume, Stable Demand",
        2: "High Volume, Mature Products"
    }
    cluster_df["Demand Segment"] = cluster_df["Cluster"].map(cluster_names)

    # 4. Display Evaluation Curves (Elbow and Silhouette)
    st.subheader("Model Evaluation and Cluster Selection")
    col1, col2 = st.columns(2)
    with col1:
        display_static_chart("elbow method.png", "Elbow Curve (Optimal Clusters: k = 3)")
    with col2:
        display_static_chart("silhouette analysis.png", "Silhouette Analysis (Highest Score at k = 3: 0.4516)")

    st.write("---")

    # 5. PCA Projection
    st.subheader("PCA Cluster Projection")
    st.write("Two-dimensional visualization of K-Means clusters projected using Principal Component Analysis (PCA):")
    display_static_chart("clusters.png", "Sub-Category Clusters in PCA Space")

    st.write("---")

    # 6. Cluster Summary Table
    st.subheader("Segment Characterization")
    summary_df = (
        cluster_df.groupby("Demand Segment")[features]
        .mean()
        .reset_index()
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.write("---")

    # 7. Stocking Strategy Notes (Data-first Markdown formatting)
    st.subheader("Stocking Strategy Notes")
    
    st.markdown(
        """
        - **Growing High-Value Products (Copiers)**:
          - *Characteristics*: Exceptional average order values and sales growth, coupled with high volatility.
          - *Replenishment Rule*: Scale inventory levels gradually; secure supplier agreements and keep safety buffer stocks to avoid stockouts.
          
        - **High Volume, Mature Products (Phones, Chairs, Storage, Binders, Paper, Accessories)**:
          - *Characteristics*: Consistently high total sales with established market demands.
          - *Replenishment Rule*: Maintain high standard stock levels and deploy automated replenishment alerts.
          
        - **Low Volume, Stable Demand (Bookcases, Tables, Appliances, Art, Envelopes, Labels, Fasteners, etc.)**:
          - *Characteristics*: Slow turnover with highly stable and predictable demand.
          - *Replenishment Rule*: Adopt a lean Just-In-Time (JIT) stock policy to minimize carrying costs.
        """
    )

    st.write("---")

    # 8. Category Assignments
    st.subheader("Sub-Category Assignments")
    display_cols = ["Demand Segment", "Total Sales", "Average Order Value", "Sales Volatility", "Sales Growth Rate"]
    st.dataframe(cluster_df[display_cols].sort_values("Demand Segment"), use_container_width=True)
