"""
Sidebar navigation component for the Superstore Sales Streamlit application.

This module renders the side navigation control without any decorative icons or emojis.
"""

import streamlit as st
from typing import List

def render_sidebar() -> str:
    """Render a flat, clean navigation sidebar and return the selected page name."""
    st.sidebar.markdown("### Superstore Analytics")
    
    # Renders clean, emoji-free navigation pages
    pages = [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments",
        "Dataset Explorer",
        "About Project"
    ]
    
    selected_page = st.sidebar.radio(
        label="Navigation Menu",
        options=pages,
        label_visibility="collapsed"
    )
    
    return selected_page
