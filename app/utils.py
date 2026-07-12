"""
Utility functions for the Superstore Sales Streamlit application.

This module provides functions for loading datasets, caching pre-trained models,
and defining model evaluation metrics. It adheres to a clean, minimalist, data-first style.
"""

import os
import pickle
import streamlit as st
import pandas as pd
from typing import Any, Dict

# Dynamic path determinations
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "train.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ==========================================
# MODEL EVALUATION METRICS
# ==========================================
MODEL_METRICS: Dict[str, Dict[str, float]] = {
    "SARIMA": {
        "MAE": 25612.15,
        "RMSE": 32649.94,
        "MAPE (%)": 24.82,
        "R² Score": -2.321
    },
    "Prophet": {
        "MAE": 20250.79,
        "RMSE": 22318.41,
        "MAPE (%)": 21.86,
        "R² Score": -0.552
    },
    "Tuned XGBoost": {
        "MAE": 13907.70,
        "RMSE": 16262.98,
        "MAPE (%)": 14.10,
        "R² Score": 0.176
    }
}

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and preprocess the Superstore Sales dataset from train.csv."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
    
    # Read the raw CSV file
    df = pd.read_csv(DATA_PATH)
    
    # Preprocess date columns to datetime formats (format matches the notebook: %d/%m/%Y)
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    
    # Extract date features
    df["Year"] = df["Order Date"].dt.year
    df["Month Number"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.strftime("%B")
    
    # Calculate shipping days
    df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    
    return df

@st.cache_resource
def load_ml_model(model_filename: str) -> Any:
    """Load a serialized model or scaler pickle from the models directory."""
    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def inject_minimalist_style() -> None:
    """Inject custom minimalist CSS styles to style layout and metrics containers."""
    minimal_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Minimalist Metric block styling using Streamlit native theme variables */
        div[data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.2);
            padding: 12px 18px;
            border-radius: 4px;
        }
    </style>
    """
    st.markdown(minimal_css, unsafe_allow_html=True)
