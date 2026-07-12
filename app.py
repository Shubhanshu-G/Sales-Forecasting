"""
Superstore Sales Analytics Dashboard - Main Entry Point.

This module coordinates page navigation using the flat sidebar menu,
sets minimal styling options, and routes execution to respective page modules.
"""

import streamlit as st
from app.utils import inject_minimalist_style
from app.sidebar import render_sidebar

# Import page modules
from app.home import render_home_page
from app.forecasting import render_forecasting_page
from app.anomaly import render_anomaly_page
from app.clustering import render_clustering_page
from app.dataset import render_dataset_page
from app.about import render_about_page

def main() -> None:
    """Main dashboard entry point: configures page settings, sidebar navigation, and routes page rendering."""
    # Configure page settings (minimalist, wide layout, no emojis)
    st.set_page_config(
        page_title="Superstore Sales Analytics Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject clean design tokens
    inject_minimalist_style()

    # Render sidebar and get selected page name
    selected_page = render_sidebar()

    # Route execution to the chosen page module
    try:
        if selected_page == "Sales Overview":
            render_home_page()
            
        elif selected_page == "Forecast Explorer":
            render_forecasting_page()
            
        elif selected_page == "Anomaly Report":
            render_anomaly_page()
            
        elif selected_page == "Product Demand Segments":
            render_clustering_page()
            
        elif selected_page == "Dataset Explorer":
            render_dataset_page()
            
        elif selected_page == "About Project":
            render_about_page()
            
        else:
            st.error("Invalid page selection.")
            
    except Exception as e:
        st.error("An unexpected error occurred while rendering the dashboard.")
        st.exception(e)

if __name__ == "__main__":
    main()
