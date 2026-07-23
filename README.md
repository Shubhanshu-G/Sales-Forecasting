# Superstore Sales Performance & Demand Forecasting Dashboard

[Live Demo](https://slaesforecasting.streamlit.app/)

An end-to-end operational sales intelligence and demand forecasting pipeline for a commercial retail superstore. This system converts historical transaction records (train.csv) into a modular, production-ready Streamlit analytics application, containerized with Docker for portable deployment.

---

## Executive Summary & Business Relevance

In commercial retail, optimizing supply chains requires balancing stock levels against customer demand. Under-stocking high-value items leads to immediate revenue loss, while over-stocking slow-moving items ties up working capital.

This project addresses these challenges by applying statistical decomposition, multi-model forecasting, anomaly isolation, and K-Means product segmentation on four years (2015–2018) of historical transactional records. The results are deployed in a Tableau/Power BI style, flat-text analytics dashboard designed for inventory planners and executive decision-makers.

---

## Evaluation Criteria & System Implementation

### 1. Time Series Analysis & Decomposition

To identify structural demand components, raw transaction sales are aggregated to weekly and monthly frequencies. We apply **multiplicative classical decomposition**:
$$\text{Sales}_t = \text{Trend}_t \times \text{Seasonality}_t \times \text{Residual}_t$$

- **Trend**: Long-term upward movement, showing consistent annual revenue growth.
- **Seasonality**: Highly recurrent Q4 spike (peaking in September, November, and December) driven by corporate holiday shopping cycles.
- **Residuals**: Random fluctuations and supply shocks, which are analyzed in the anomaly module.

### 2. Time Series Forecasting (All 3 Models)

We implement and serialize three distinct modeling methodologies to capture different signal types:

1. **SARIMA (Seasonal Autoregressive Integrated Moving Average)**:
   - **Parameters**: $\text{SARIMAX}(1,0,0) \times (0,0,0)_{12}$ fit on monthly sales.
   - **Role**: Serves as a linear statistical baseline modeling auto-regressive properties of the sales timeline.
2. **Meta Prophet**:
   - **Role**: An additive regression model decomposing non-linear trends with yearly seasonality. Highly resilient to missing intervals.
3. **Tuned XGBoost (Gradient Boosted Trees)**:
   - **Feature Engineering**: Formulates forecasting as a supervised learning task. Features include 3 historical monthly lags ($t-1$, $t-2$, $t-3$), 3-month rolling averages, and extracted temporal indicators (Month, Quarter, Season).
   - **Role**: Captures non-linear feature interactions and localized shocks.

### 3. Model Comparison & Justified Recommendation

Models are evaluated on a 3-month holdout test set (October – December 2018):

| Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | MAPE (%) | $R^2$ Score | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SARIMA** | $25,612.15 | $32,649.94 | 24.82% | -2.321 | Baseline |
| **Prophet** | $20,250.79 | $22,318.41 | 21.86% | -0.552 | Runner-Up |
| **Tuned XGBoost** | **$13,907.70** | **$16,262.98** | **14.10%** | **0.176** | **Production Winner** |

- **Winner Selection**: **Tuned XGBoost** is recommended for production. It achieves the lowest forecast error (MAE of $\$13,907.70$, representing a $45\%$ improvement over SARIMA) and is the only model achieving a positive $R^2$ score ($0.176$), proving it successfully captures the high-volatility seasonal demand spikes in Q4.

### 4. Double-Method Anomaly Detection

To isolate revenue surges and operational disruptions, we employ two concurrent methods:

1. **Isolation Forest (Global)**: An unsupervised algorithm isolating anomalous weekly sales sums by randomly partitioning feature values across the full dataset distribution.
2. **Rolling Z-Score (Local)**: Identifies week-over-week surges deviating by $>2$ standard deviations from the rolling 8-week moving average.

- **Key Finding (Overlap on 22-Mar-2015)**: The week ending **22 March 2015** is the only date flagged by both models. This intersection isolates a major sales spike ($\$11,543.60$) that is both a global extreme and a sharp local departure from the surrounding trend.

### 5. Product Demand Segmentation & K-Means Clustering

To replace arbitrary category boundaries, we aggregate four operational features for each product sub-category: Total Sales, Order Volume, Sales Volatility (Standard Deviation), and Year-over-Year (YoY) Sales Growth.

Data is scaled using `StandardScaler` and projected into 2D space using **Principal Component Analysis (PCA)**. We apply **K-Means clustering** ($k=3$, validated by Elbow and Silhouette methods) to segment products:

1. **Cluster 0: Low-Volume, Stable Demand** (e.g., Envelopes, Labels, Fasteners):
   - *Strategy*: Apply Just-in-Time (JIT) replenishment to minimize holding costs.
2. **Cluster 1: High-Volume, Growing Demand** (e.g., Copiers, Phones, Accessories):
   - *Strategy*: Establish high safety stock levels to capture maximum market share and prevent stockouts.
3. **Cluster 2: High-Volatility, Cyclical Demand** (e.g., Tables, Chairs, Machines):
   - *Strategy*: Maintain flexible, demand-driven scheduling and coordinate promotions with Q4 seasonal peaks.

### 6. Streamlit Dashboard Architecture & Custom Theme Adaptability

The web app is structured modularly for production-grade maintenance:

```
project/
│
├── app.py                    # Main dashboard router
├── app/
│   ├── __init__.py           # Package indicator
│   ├── sidebar.py            # Flat text sidebar navigation
│   ├── home.py               # Sales Overview: dynamic KPIs and Plotly charts
│   ├── forecasting.py        # Model evaluations: Tuned XGBoost, SARIMA, Prophet
│   ├── anomaly.py            # Outlier tables & overlap analysis
│   ├── clustering.py         # K-Means PCA projection & replenishment rules
│   ├── dataset.py            # Data profile schema, summaries, and CSV downloads
│   ├── about.py              # Operational workflows and key insights
│   ├── charts.py             # Plotly and static chart rendering helpers
│   └── utils.py              # Data caching, model deserialization, and styling
├── models/                   # Serialized ML artifacts (.pkl)
├── charts/                   # Pre-rendered static analysis plots
├── train.csv                 # Core transaction dataset
├── requirements.txt          # Production environment specifications
├── Dockerfile                # Container build definition
├── .dockerignore             # Files excluded from the Docker build context
└── README.md                 # System report
```

- **Usability & Theme Adaptability**: Custom CSS handles typeface rendering. Instead of hardcoded background colors, metrics panels reference Streamlit's native CSS variable `var(--secondary-background-color)`. This ensures boxes render with high contrast and readable text in both the default Dark and Light/White themes.

---

## Docker Deployment

The application is fully containerized using a `Dockerfile` and `.dockerignore`.

Docker Hub repository: [dropper135/sales-forecasting-app](https://hub.docker.com/repository/docker/dropper135/sales-forecasting-app/general)

### Pull the Docker Image

```bash
docker pull dropper135/sales-forecasting-app:latest
```

### Run the Container

```bash
docker run -p 8501:8501 dropper135/sales-forecasting-app:latest
```

Open your browser and navigate to:

```
http://localhost:8501
```

---

## Local Setup (Without Docker)

### 1. Environment Setup

We recommend using Python 3.10. Install all system dependencies specified in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Run the Application

Run the Streamlit server from the root directory:

```bash
streamlit run app.py
```

Open your web browser and navigate to `http://localhost:8501`.

### 3. Model Retraining (Optional)

The application loads pre-trained model artifacts from the `models/` directory on startup. If you need to retrain or modify the underlying machine learning logic, you can execute the exploratory notebook:

```bash
jupyter nbconvert --to notebook --execute analysis.ipynb
```

*Note: Do not call `.fit()` or modify models during live app sessions.*

---

# Sales-Forecasting
