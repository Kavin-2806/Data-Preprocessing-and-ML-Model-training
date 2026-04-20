import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
    silhouette_score,
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib
import tempfile
import os
import io
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="🚀 ML Dashboard Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern, impressive UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --warning-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        --card-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        --glass-bg: rgba(255, 255, 255, 0.95);
        --glass-border: rgba(255, 255, 255, 0.2);
    }

    * {
        box-sizing: border-box;
    }

    body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .css-1lcbmhc.e1fqkh3o2 {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        color: #1e293b !important;
        font-family: 'Inter', sans-serif !important;
        overflow-x: hidden;
    }

    /* Modern Header */
    .main-header {
        background: var(--primary-gradient);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--card-shadow);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
    }

    .main-header h1 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.02em;
    }

    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-weight: 400;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.15);
    }

    /* Enhanced Sidebar */
    .sidebar-header {
        background: var(--primary-gradient);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        text-align: center;
    }

    .sidebar-header h2 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.5rem !important;
        margin: 0 !important;
        font-weight: 600;
    }

    /* Modern Buttons */
    .stButton>button {
        background: var(--primary-gradient) !important;
        color: #ffffff !important;
        border-radius: 15px !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important;
    }

    .stButton>button:active {
        transform: translateY(0px) !important;
    }

    /* Success/Warning Buttons */
    .success-btn>button {
        background: var(--success-gradient) !important;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4) !important;
    }

    .warning-btn>button {
        background: var(--warning-gradient) !important;
        box-shadow: 0 4px 15px rgba(250, 112, 154, 0.4) !important;
    }

    /* Enhanced Form Elements */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>div>div,
    .stTextArea>div>div>textarea,
    .stNumberInput>div>div>input {
        background: var(--glass-bg) !important;
        color: #1e293b !important;
        border: 2px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }

    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div>div>div:focus-within,
    .stTextArea>div>div>textarea:focus,
    .stNumberInput>div>div>input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        transform: translateY(-1px);
    }

    /* Multiselect Enhancement */
    .stMultiSelect div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] input,
    .stMultiSelect div[data-baseweb="select"] [data-baseweb="control"] {
        background: var(--glass-bg) !important;
        color: #1e293b !important;
        box-shadow: none !important;
        border: 2px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }

    .stMultiSelect div[data-baseweb="select"] input {
        background: transparent !important;
        color: #1e293b !important;
        caret-color: #1e293b !important;
        outline: none !important;
        -webkit-box-shadow: 0 0 0px 1000px var(--glass-bg) inset !important;
        box-shadow: 0 0 0px 1000px var(--glass-bg) inset !important;
    }

    .stMultiSelect div[data-baseweb="select"] [data-baseweb="tag"] {
        background: var(--primary-gradient) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }

    /* Enhanced Metrics */
    .metric-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--card-shadow);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }

    /* Progress Bars */
    .progress-container {
        background: rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 1rem 0;
    }

    .progress-bar {
        height: 100%;
        background: var(--primary-gradient);
        border-radius: 10px;
        transition: width 0.5s ease;
    }

    /* Section Headers */
    .section-header {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
        margin: 2rem 0 1rem 0 !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-header::before {
        content: '';
        width: 4px;
        height: 2rem;
        background: var(--primary-gradient);
        border-radius: 2px;
    }

    /* Subheaders */
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
        margin: 1.5rem 0 1rem 0 !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Enhanced Tables */
    .dataframe {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: var(--card-shadow) !important;
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
    }

    .dataframe th {
        background: var(--primary-gradient) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.9rem !important;
    }

    .dataframe td {
        padding: 0.75rem 1rem !important;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2) !important;
    }

    .dataframe tr:nth-child(even) {
        background: rgba(241, 245, 249, 0.5) !important;
    }

    .dataframe tr:hover {
        background: rgba(102, 126, 234, 0.1) !important;
    }

    /* Status Messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 12px !important;
        border: none !important;
        box-shadow: var(--card-shadow) !important;
        backdrop-filter: blur(10px) !important;
    }

    .stSuccess {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(34, 197, 94, 0.05)) !important;
        border-left: 4px solid #22c55e !important;
    }

    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.05)) !important;
        border-left: 4px solid #3b82f6 !important;
    }

    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05)) !important;
        border-left: 4px solid #f59e0b !important;
    }

    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05)) !important;
        border-left: 4px solid #ef4444 !important;
    }

    /* Sidebar Navigation */
    .sidebar-nav-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 10px;
        transition: all 0.3s ease;
        cursor: pointer;
        font-weight: 500;
    }

    .sidebar-nav-item:hover {
        background: rgba(102, 126, 234, 0.1);
        transform: translateX(5px);
    }

    .sidebar-nav-item.active {
        background: var(--primary-gradient);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .sidebar-nav-item i {
        width: 20px;
        text-align: center;
    }

    /* Loading Animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
    }

    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 4px solid rgba(102, 126, 234, 0.2);
        border-left: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem !important;
        }

        .glass-card {
            padding: 1rem !important;
        }

        .section-header {
            font-size: 1.5rem !important;
        }
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(148, 163, 184, 0.1);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .css-1rs6os {visibility: hidden;}
    .css-17ziqus {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper functions
def init_session_state():
    if "experiment_history" not in st.session_state:
        st.session_state["experiment_history"] = []
    if "trained_pipeline" not in st.session_state:
        st.session_state["trained_pipeline"] = None
    if "training_features" not in st.session_state:
        st.session_state["training_features"] = []
    if "training_schema" not in st.session_state:
        st.session_state["training_schema"] = {}
    if "model_download_blob" not in st.session_state:
        st.session_state["model_download_blob"] = None
    if "best_model_name" not in st.session_state:
        st.session_state["best_model_name"] = None
    if "optimized_model_name" not in st.session_state:
        st.session_state["optimized_model_name"] = None


def detect_task_type(y):
    return not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20


def get_model_options(is_classification):
    if is_classification:
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
        }
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
    }


def build_model_with_params(model_name, params, is_classification):
    params = params.copy()
    if model_name == "Logistic Regression":
        if "max_iter" not in params:
            params["max_iter"] = 1000
        return LogisticRegression(**params)
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42, **params)
    if model_name == "Random Forest":
        return RandomForestClassifier(random_state=42, **params)
    if model_name == "KNN":
        return KNeighborsClassifier(**params)
    if model_name == "Linear Regression":
        return LinearRegression(**params)
    if model_name == "Decision Tree Regressor":
        return DecisionTreeRegressor(random_state=42, **params)
    if model_name == "Random Forest Regressor":
        return RandomForestRegressor(random_state=42, **params)
    return None


def get_manual_hyperparameters(model_name):
    params = {}
    if model_name == "Logistic Regression":
        params["C"] = st.number_input("Logistic Regression: C", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
        params["solver"] = st.selectbox("Solver", ["lbfgs", "liblinear"], index=0)
        params["max_iter"] = st.number_input("Max Iterations", min_value=100, max_value=5000, value=1000, step=100)
    elif model_name == "Decision Tree":
        max_depth = st.selectbox("Max Depth", [None, 3, 5, 10, 20], index=0)
        params["max_depth"] = max_depth
        params["min_samples_split"] = st.slider("Min Samples Split", 2, 10, 2)
    elif model_name == "Random Forest":
        params["n_estimators"] = st.select_slider("Number of Trees", [50, 100, 150, 200], value=100)
        params["max_depth"] = st.selectbox("Max Depth", [None, 5, 10, 20], index=0)
        params["min_samples_split"] = st.slider("Min Samples Split", 2, 10, 2)
    elif model_name == "KNN":
        params["n_neighbors"] = st.slider("Neighbors (k)", 1, 20, 5)
        params["weights"] = st.selectbox("Weights", ["uniform", "distance"], index=0)
    elif model_name == "Linear Regression":
        params = {}
    elif model_name == "Decision Tree Regressor":
        params["max_depth"] = st.selectbox("Max Depth", [None, 3, 5, 10, 20], index=0)
        params["min_samples_split"] = st.slider("Min Samples Split", 2, 10, 2)
    elif model_name == "Random Forest Regressor":
        params["n_estimators"] = st.select_slider("Number of Trees", [50, 100, 150, 200], value=100)
        params["max_depth"] = st.selectbox("Max Depth", [None, 5, 10, 20], index=0)
        params["min_samples_split"] = st.slider("Min Samples Split", 2, 10, 2)
    return params


def build_tuning_grid(model_name):
    grid = {}
    if model_name == "Random Forest":
        grid = {
            "model__n_estimators": [50, 100, 150],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 4, 6],
        }
    elif model_name == "Random Forest Regressor":
        grid = {
            "model__n_estimators": [50, 100, 150],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 4, 6],
        }
    elif model_name == "Decision Tree":
        grid = {
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 4, 6],
        }
    elif model_name == "Decision Tree Regressor":
        grid = {
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 4, 6],
        }
    elif model_name == "KNN":
        grid = {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
        }
    elif model_name == "Logistic Regression":
        grid = {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs", "liblinear"],
        }
    return grid


def display_tuning_controls(model_name):
    grid = build_tuning_grid(model_name)
    selected_grid = {}

    if not grid:
        st.info("No hyperparameter grid available for this model.")
        return selected_grid

    st.write("Define hyperparameter search ranges:")
    for param, options in grid.items():
        if isinstance(options[0], str) or options == [None, 5, 10, 20]:
            selected_values = st.multiselect(param, options, default=options)
        else:
            selected_values = st.multiselect(param, options, default=options[:min(3, len(options))])
        if selected_values:
            selected_grid[param] = selected_values
    return selected_grid


def build_tuning_grid_from_inputs(model_name):
    if model_name == "Logistic Regression":
        return {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs", "liblinear"],
        }
    return build_tuning_grid(model_name)


def build_preprocessing_pipeline(X, is_classification, feature_selection_k=None):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    if not transformers:
        return None

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)
    steps = [("preprocessor", preprocessor)]

    if feature_selection_k is not None and feature_selection_k > 0:
        score_func = f_classif if is_classification else f_regression
        steps.append(("feature_selection", SelectKBest(score_func=score_func, k=feature_selection_k)))

    return Pipeline(steps)


def create_prediction_schema(X):
    schema = {}
    for col in X.columns:
        dtype = X[col].dtype
        is_numeric = pd.api.types.is_numeric_dtype(dtype)
        options = []
        if not is_numeric:
            unique_vals = X[col].dropna().unique().tolist()
            options = sorted([str(x) for x in unique_vals])
        schema[col] = {"is_numeric": is_numeric, "options": options}
    return schema


def get_prediction_input(schema):
    input_data = {}
    for col, meta in schema.items():
        if meta["is_numeric"]:
            input_data[col] = st.number_input(col, value=0.0, format="%.6f", key=f"pred_{col}")
        else:
            if len(meta["options"]) > 0 and len(meta["options"]) <= 100:
                input_data[col] = st.selectbox(col, meta["options"], key=f"pred_{col}")
            else:
                input_data[col] = st.text_input(col, key=f"pred_{col}")
    return pd.DataFrame([input_data])


def save_model_blob(pipeline, feature_columns):
    payload = {"pipeline": pipeline, "feature_columns": feature_columns}
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    tmp_file.close()
    joblib.dump(payload, tmp_file.name)
    with open(tmp_file.name, "rb") as f:
        blob = f.read()
    os.remove(tmp_file.name)
    return blob


def get_highly_correlated_pairs(df, threshold=0.8):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr_matrix = df[numeric_cols].corr().abs()
    high_corr = []
    for i, col_i in enumerate(numeric_cols):
        for j, col_j in enumerate(numeric_cols):
            if j <= i:
                continue
            value = corr_matrix.iloc[i, j]
            if not np.isnan(value) and value > threshold:
                high_corr.append((col_i, col_j, float(value)))
    return high_corr


def format_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ===== FEATURE 1: BUSINESS PROBLEM MODE =====
BUSINESS_PROBLEMS = {
    "Classification": {
        "Churn Prediction": {
            "description": "Predict if a customer will leave",
            "target_suggestions": ["churn", "left", "churned", "cancel"],
            "insights": "Focus on customer lifetime value and engagement metrics"
        },
        "Fraud Detection": {
            "description": "Identify fraudulent transactions",
            "target_suggestions": ["fraud", "is_fraud", "fraudulent", "suspicious"],
            "insights": "Class imbalance is common; use SMOTE or class weights"
        },
        "Disease Diagnosis": {
            "description": "Classify if a patient has a disease",
            "target_suggestions": ["disease", "diagnosis", "condition", "positive"],
            "insights": "Focus on sensitivity and specificity metrics"
        },
    },
    "Regression": {
        "Sales Forecasting": {
            "description": "Predict future sales",
            "target_suggestions": ["sales", "revenue", "amount", "total"],
            "insights": "Consider seasonality and time-series patterns"
        },
        "Price Prediction": {
            "description": "Estimate product/house price",
            "target_suggestions": ["price", "cost", "value", "rate"],
            "insights": "Identify price drivers and outliers early"
        },
        "Demand Forecasting": {
            "description": "Predict product demand",
            "target_suggestions": ["demand", "quantity", "units", "volume"],
            "insights": "External factors (weather, seasonality) matter"
        },
    }
}


# ===== FEATURE 2: SMART DATA RECOMMENDATIONS =====
def get_data_recommendations(df):
    """Generate AI-powered data quality suggestions"""
    recommendations = []
    
    # Check skewness
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        skewness = abs(stats.skew(df[col].dropna()))
        if skewness > 1:
            recommendations.append({
                "type": "Transform",
                "column": col,
                "issue": f"High skewness ({skewness:.2f})",
                "suggestion": "Apply log transform or Box-Cox",
                "priority": "Medium"
            })
    
    # Check correlations
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.9:
                    recommendations.append({
                        "type": "Feature",
                        "column": f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]}",
                        "issue": f"High correlation ({corr_matrix.iloc[i, j]:.2f})",
                        "suggestion": "Drop one highly correlated feature",
                        "priority": "High"
                    })
    
    # Check missing values
    missing_pct = (df.isnull().sum() / len(df) * 100)
    for col, pct in missing_pct[missing_pct > 0].items():
        if pct > 30:
            recommendations.append({
                "type": "Missing",
                "column": col,
                "issue": f"Missing values ({pct:.1f}%)",
                "suggestion": "Drop column or use advanced imputation",
                "priority": "High"
            })
        elif pct > 10:
            recommendations.append({
                "type": "Missing",
                "column": col,
                "issue": f"Missing values ({pct:.1f}%)",
                "suggestion": "Use mean/median imputation",
                "priority": "Medium"
            })
    
    return pd.DataFrame(recommendations) if recommendations else None


# ===== FEATURE 3: FEATURE ENGINEERING MODULE =====
def apply_polynomial_features(X, degree=2):
    """Generate polynomial features"""
    # Drop NaN values before polynomial features
    X_clean = X.dropna()
    if len(X_clean) == 0:
        st.error("No valid data after removing NaN values")
        return X
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_clean)
    feature_names = poly.get_feature_names_out(input_features=X_clean.columns.tolist())
    return pd.DataFrame(X_poly, columns=feature_names, index=X_clean.index)


def extract_date_features(df, date_col):
    """Extract date features"""
    if date_col not in df.columns:
        return df
    
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    
    # Drop rows with NaT values in date column
    df_copy = df_copy.dropna(subset=[date_col])
    
    if len(df_copy) == 0:
        return df.drop(date_col, axis=1)
    
    df_copy[f"{date_col}_year"] = df_copy[date_col].dt.year
    df_copy[f"{date_col}_month"] = df_copy[date_col].dt.month
    df_copy[f"{date_col}_day"] = df_copy[date_col].dt.day
    df_copy[f"{date_col}_dayofweek"] = df_copy[date_col].dt.dayofweek
    df_copy[f"{date_col}_quarter"] = df_copy[date_col].dt.quarter
    
    return df_copy.drop(date_col, axis=1)


def apply_binning(X, column, method='equal_width', n_bins=5):
    """Bin continuous features"""
    X_copy = X.copy()
    # Drop NaN values in the column to bin
    X_copy = X_copy.dropna(subset=[column])
    if len(X_copy) == 0:
        return X_copy
    try:
        if method == 'equal_width':
            X_copy[f"{column}_binned"] = pd.cut(X_copy[column], bins=n_bins)
        else:  # quantile
            X_copy[f"{column}_binned"] = pd.qcut(X_copy[column], q=n_bins, duplicates='drop')
    except Exception as e:
        st.error(f"Binning error: {str(e)}")
    return X_copy


def create_feature_interactions(X, col1, col2):
    """Create interaction features"""
    X_copy = X.copy()
    if col1 in X.columns and col2 in X.columns:
        X_copy[f"{col1}_x_{col2}"] = X_copy[col1] * X_copy[col2]
    return X_copy


# ===== FEATURE 4: ADVANCED MODEL EVALUATION =====
def get_classification_metrics(y_test, y_pred, y_pred_proba=None):
    """Comprehensive classification evaluation"""
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }
    
    if y_pred_proba is not None and len(np.unique(y_test)) == 2:
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    return metrics


def get_regression_metrics(y_test, y_pred, n_features=1):
    """Comprehensive regression evaluation"""
    n_samples = len(y_test)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    
    return {
        "R² Score": r2,
        "Adjusted R²": adjusted_r2,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    }


# ===== FEATURE 5: IMBALANCE DETECTION & HANDLING =====
def detect_class_imbalance(y, threshold=0.6):
    """Detect class imbalance in target variable"""
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 1:
        return None, None
    
    value_counts = y.value_counts()
    if len(value_counts) > 2:
        return None, None  # Multiclass not supported for simple detection
    
    if len(value_counts) == 2:
        ratio = value_counts.iloc[1] / value_counts.iloc[0]
        imbalance_ratio = min(ratio, 1/ratio)
        is_imbalanced = imbalance_ratio < threshold
        return is_imbalanced, imbalance_ratio
    
    return None, None


def apply_smote(X, y, random_state=42):
    """Apply SMOTE to handle class imbalance"""
    try:
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    except:
        return X, y


# ===== FEATURE 6: OVERFITTING/UNDERFITTING DETECTION =====
def detect_overfitting(train_score, test_score, threshold=0.15):
    """Detect overfitting based on train-test gap"""
    gap = train_score - test_score
    
    if gap > threshold:
        return "Overfitting", f"Train-test gap: {gap:.3f}"
    elif train_score < 0.6 and test_score < 0.6:
        return "Underfitting", "Both scores are low"
    else:
        return "Healthy", "Model is generalizing well"


# ===== FEATURE 7: IMPROVED AUTOML LOGIC =====
def improved_automl_selection(model_scores_dict, is_classification):
    """Select best model based on score AND stability"""
    best_model = None
    best_score = -np.inf
    best_stability = np.inf
    best_name = None
    
    for model_name, scores in model_scores_dict.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Prefer high score with low variance
        if mean_score > best_score:
            best_score = mean_score
            best_stability = std_score
            best_name = model_name
    
    return best_name, best_score, best_stability


# ===== FEATURE 9: CLUSTERING IMPROVEMENTS =====
def get_clustering_metrics(X, labels):
    """Get clustering evaluation metrics"""
    sil_score = silhouette_score(X, labels)
    return {"Silhouette Score": sil_score}


# ===== FEATURE 11: REPORT GENERATION =====
def generate_experiment_report(df, selected_features, model_name, metrics, timestamp):
    """Generate markdown report"""
    report = f"""
# ML Model Experiment Report
**Generated:** {timestamp}

## Dataset Summary
- **Total Records:** {len(df):,}
- **Total Features:** {df.shape[1]}
- **Selected Features:** {', '.join(selected_features)}

## Model Configuration
- **Model Used:** {model_name}
- **Training Time:** {timestamp}

## Performance Metrics
"""
    for metric, value in metrics.items():
        if isinstance(value, float):
            report += f"- **{metric}:** {value:.4f}\n"
        else:
            report += f"- **{metric}:** {value}\n"
    
    return report


# ===== FEATURE 12: API MODE CODE GENERATION =====
def generate_fastapi_code(model_name, feature_list):
    """Generate FastAPI code snippet"""
    api_code = f'''
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

class PredictionInput(BaseModel):
    {chr(10).join([f'    {feat}: float' for feat in feature_list[:5]])}

@app.post("/predict")
async def predict(data: PredictionInput):
    features = np.array([
        {chr(10).join([f'        data.{feat},' for feat in feature_list[:5]])}
    ]).reshape(1, -1)
    
    prediction = model.predict(features)
    
    return {{"prediction": float(prediction[0])}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    return api_code


# ===== FEATURE 13: MODEL JUSTIFICATION =====
MODEL_EXPLANATIONS = {
    "Logistic Regression": {
        "strengths": "Interpretable, fast, good for linear relationships",
        "weaknesses": "Struggles with non-linear patterns",
        "use_cases": "Credit scoring, marketing response"
    },
    "Decision Tree": {
        "strengths": "Interpretable, handles non-linearity, no scaling needed",
        "weaknesses": "Prone to overfitting, unstable",
        "use_cases": "Business rules, fast decisions"
    },
    "Random Forest": {
        "strengths": "Robust, handles interactions, good default",
        "weaknesses": "Less interpretable, slower predictions",
        "use_cases": "General purpose, competition"
    },
    "KNN": {
        "strengths": "Simple, effective for local patterns",
        "weaknesses": "Slow, memory intensive, sensitive to scaling",
        "use_cases": "Anomaly detection, similarity search"
    },
    "Linear Regression": {
        "strengths": "Interpretable, fast, good baseline",
        "weaknesses": "Assumes linearity, sensitive to outliers",
        "use_cases": "Trend analysis, forecasting"
    },
}


init_session_state()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

    body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .css-1lcbmhc.e1fqkh3o2, .css-1v3fvcr {
        background-color: #f8fafc !important;
        color: #1e293b !important;
        font-family: 'Inter', sans-serif !important;
    }

    .block-container {
        padding: 2rem 2rem 2rem 2rem !important;
        border-radius: 22px;
        background: #ffffff !important;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.1);
    }

    .stButton>button {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border-radius: 14px !important;
        border: none !important;
        padding: 0.75rem 1.2rem !important;
        font-weight: 600 !important;
    }

    .stButton>button:hover {
        background-color: #2563eb !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #0f172a !important;
    }

    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stNumberInput>div>div>input {
        background: #f1f5f9 !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 0.85rem 1rem !important;
        min-height: 3rem !important;
        line-height: 1.4 !important;
        transition: all 0.2s ease !important;
    }

    .stSelectbox>div>div>div,
    .stSelectbox>div>div>div>div {
        background: #f1f5f9 !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 0.85rem 1rem !important;
        min-height: 3rem !important;
        line-height: 1.4 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        box-shadow: none !important;
    }

    .stSelectbox div[data-baseweb="select"] [role="button"],
    .stSelectbox div[data-baseweb="select"] [role="combobox"],
    .stSelectbox div[data-baseweb="select"] [data-baseweb="control"] {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
        min-height: 1.9rem !important;
        padding: 0 !important;
    }

    .stSelectbox div[data-baseweb="select"] [data-baseweb="value"] {
        color: #1e293b !important;
        line-height: 1.4 !important;
    }

    .stSelectbox>div>div>div>div:focus-within,
    .stSelectbox div[data-baseweb="select"] [role="button"]:focus,
    .stSelectbox div[data-baseweb="select"] [role="combobox"]:focus {
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.12) !important;
    }

    /* Multiselect specific styling */
    .stMultiSelect div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] input,
    .stMultiSelect div[data-baseweb="select"] [data-baseweb="control"],
    .stMultiSelect div[data-baseweb="select"] [role="combobox"] {
        background: transparent !important;
        color: #1e293b !important;
        box-shadow: none !important;
        border: none !important;
    }

    .stMultiSelect div[data-baseweb="select"] input {
        background: transparent !important;
        color: #1e293b !important;
        caret-color: #1e293b !important;
        outline: none !important;
        -webkit-box-shadow: 0 0 0px 1000px #f1f5f9 inset !important;
        box-shadow: 0 0 0px 1000px #f1f5f9 inset !important;
    }

    .stMultiSelect div[data-baseweb="select"] [data-baseweb="tag"] {
        background: #3b82f6 !important;
        color: #ffffff !important;
    }

    .stMultiSelect div[data-baseweb="select"] [data-baseweb="tag"] span {
        color: #ffffff !important;
    }

    .stMultiSelect div[data-baseweb="popover"] {
        background: #f1f5f9 !important;
    }

    .stMultiSelect div[data-baseweb="popover"] div {
        color: #1e293b !important;
    }

    .stMultiSelect div[data-baseweb="popover"] div:hover {
        background: #e2e8f0 !important;
    }

    /* Selectbox dropdown styling */
    .stSelectbox div[data-baseweb="select"] [data-baseweb="popover"] {
        background: #f1f5f9 !important;
    }

    .stSelectbox div[data-baseweb="select"] [data-baseweb="popover"] div {
        color: #1e293b !important;
    }

    .stSelectbox div[data-baseweb="select"] [data-baseweb="popover"] div:hover {
        background: #e2e8f0 !important;
    }

    .css-1n76uvr { background-color: #f1f5f9 !important; }
    .css-1v3fvcr { background-color: #f1f5f9 !important; }
    .css-18e3th9 { background-color: #ffffff !important; }
    .css-1aw1k6o { color: #0f172a !important; }
    .css-1kyxreq { color: #475569 !important; }

    .stMarkdown h1 {
        letter-spacing: 0.03em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Enhanced Header
st.markdown("""
<div class="main-header">
    <h1>🚀 ML Dashboard Pro</h1>
    <p>Advanced Machine Learning & Data Science Platform</p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR NAVIGATION
# =========================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>🧭 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)

    # Navigation options with icons
    nav_options = {
        "🎯 Problem Mode": "🎯",
        "📊 Dataset Overview": "📊",
        "💡 Insights & Recommendations": "💡",
        "🧬 Feature Engineering": "🧬",
        "🔍 Missing Values": "🔍",
        "🧹 Data Cleaning": "🧹",
        "🔤 Encoding": "🔤",
        "⚖️ Scaling": "⚖️",
        "📈 EDA": "📈",
        "🎯 Outlier Detection": "🎯",
        "🔄 PCA": "🔄",
        "🤖 Model Training": "🤖",
        "📊 Advanced Evaluation": "📊",
        "🎨 Clustering": "🎨",
        "📋 Experiment Tracking": "📋",
    }

    option = None
    for nav_item, icon in nav_options.items():
        clean_name = nav_item.split(" ", 1)[1]  # Remove emoji from nav_item
        if st.button(f"{icon} {clean_name}", key=f"nav_{nav_item}", use_container_width=True):
            option = nav_item

    # If no button clicked, default to first option or keep current
    if option is None:
        if "current_option" not in st.session_state:
            st.session_state["current_option"] = "Dataset Overview"
        option = st.session_state["current_option"]
    else:
        st.session_state["current_option"] = option

    # File uploader in sidebar
    st.markdown("---")
    uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"], help="Upload your dataset to get started")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ===== FEATURE 1: PROBLEM MODE SELECTION =====
    if option and "Problem Mode" in option:
        st.markdown('<h1 class="section-header">🎯 Business Problem Mode</h1>', unsafe_allow_html=True)
        
        problem_category = st.selectbox("Select Problem Category", list(BUSINESS_PROBLEMS.keys()))
        
        if problem_category:
            problem_type = st.selectbox(
                "Select Problem Type",
                list(BUSINESS_PROBLEMS[problem_category].keys())
            )
            
            if problem_type:
                problem_info = BUSINESS_PROBLEMS[problem_category][problem_type]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### {problem_type}")
                    st.write(problem_info["description"])
                
                with col2:
                    st.markdown("### Suggested Target Variables")
                    for target in problem_info["target_suggestions"]:
                        st.write(f"• `{target}`")
                
                st.markdown("---")
                st.markdown(f"### 💡 Key Insight")
                st.info(problem_info["insights"])
                
                # Store problem context
                st.session_state["problem_category"] = problem_category
                st.session_state["problem_type"] = problem_type

    # =========================
    # DATASET OVERVIEW
    # =========================
    if "Dataset Overview" in option:
        st.markdown('<h1 class="section-header">📊 Dataset Overview</h1>', unsafe_allow_html=True)

        # Key metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df.shape[0]:,}</div>
                <div class="metric-label">Total Rows</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df.shape[1]}</div>
                <div class="metric-label">Total Columns</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            numeric_cols = len(df.select_dtypes(include=np.number).columns)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{numeric_cols}</div>
                <div class="metric-label">Numeric Columns</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            categorical_cols = len(df.select_dtypes(exclude=np.number).columns)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{categorical_cols}</div>
                <div class="metric-label">Categorical Columns</div>
            </div>
            """, unsafe_allow_html=True)

        # Data preview in glass card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">📋 Data Preview</h3>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Dataset info in columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">📏 Shape & Dtypes</h3>', unsafe_allow_html=True)
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write("**Data Types:**")
            dtype_counts = df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"- {dtype}: {count} columns")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">📊 Summary Statistics</h3>', unsafe_allow_html=True)
            st.dataframe(df.describe().round(3), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # INSIGHTS & SMART RECOMMENDATIONS
    # =========================
    elif "Insights" in option:
        st.markdown('<h1 class="section-header">💡 Data Insights & Recommendations</h1>', unsafe_allow_html=True)
        
        # ===== FEATURE 2: SMART DATA RECOMMENDATIONS =====
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">🤖 AI-Powered Recommendations</h3>', unsafe_allow_html=True)
        
        recommendations_df = get_data_recommendations(df)
        if recommendations_df is not None and len(recommendations_df) > 0:
            st.dataframe(recommendations_df, use_container_width=True)
        else:
            st.success("✅ No critical data quality issues detected!")
        st.markdown('</div>', unsafe_allow_html=True)

        # Missing values analysis
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">🔍 Missing Values Summary</h3>', unsafe_allow_html=True)
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            st.success("✅ No missing values detected in your dataset!")
        else:
            # Progress bar for missing data percentage
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = missing.sum()
            missing_percentage = (missing_cells / total_cells) * 100

            st.markdown(f"""
            <div style="margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span>Missing Data: {missing_cells:,} cells ({missing_percentage:.1f}%)</span>
                </div>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {min(missing_percentage, 100)}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(missing.rename("Missing Count").to_frame(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Numeric features analysis
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">📈 Numeric Feature Analysis</h3>', unsafe_allow_html=True)

            skewness = df[numeric_cols].skew().round(3)
            skew_df = skewness.to_frame(name="Skewness")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Skewness Values:**")
                st.dataframe(skew_df, use_container_width=True)

            with col2:
                # Visualize skewness
                fig = px.bar(skew_df, x=skew_df.index, y='Skewness',
                           title="Feature Skewness Distribution",
                           color='Skewness',
                           color_continuous_scale=['red', 'yellow', 'green'])
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            skewed = skewness[skewness.abs() > 0.5]
            if not skewed.empty:
                st.info(f"⚠️ **Highly skewed features:** {', '.join([f'{col} ({val:.2f})' for col, val in skewed.items()])}")
            else:
                st.success("✅ All numeric features have acceptable skewness levels!")

            st.markdown('</div>', unsafe_allow_html=True)

        # Correlation analysis
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">🔗 Feature Correlations</h3>', unsafe_allow_html=True)

        correlated_pairs = get_highly_correlated_pairs(df, threshold=0.8)
        if correlated_pairs:
            st.warning(f"⚠️ Found {len(correlated_pairs)} highly correlated feature pairs (r > 0.8)")

            corr_data = []
            for x, y_col, corr_value in correlated_pairs:
                corr_data.append({"Feature 1": x, "Feature 2": y_col, "Correlation": f"{corr_value:.3f}"})

            st.dataframe(pd.DataFrame(corr_data), use_container_width=True)

            # Correlation heatmap
            numeric_df = df.select_dtypes(include=np.number)
            if numeric_df.shape[1] > 1:
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix,
                              text_auto=True,
                              color_continuous_scale='RdBu_r',
                              title="Correlation Heatmap")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No highly correlated feature pairs found!")
        st.markdown('</div>', unsafe_allow_html=True)

    # ===== FEATURE 3: FEATURE ENGINEERING MODULE =====
    elif "Feature Engineering" in option:
        st.markdown('<h1 class="section-header">🧬 Feature Engineering</h1>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">🔧 Engineering Tools</h3>', unsafe_allow_html=True)
        
        fe_tabs = st.tabs(["📊 Polynomial Features", "📅 Date Features", "📦 Binning", "⚙️ Interactions"])
        
        with fe_tabs[0]:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                degree = st.slider("Polynomial Degree", 1, 5, 2)
                if st.button("Apply Polynomial Features"):
                    X_poly = apply_polynomial_features(df[numeric_cols], degree=degree)
                    st.success(f"✅ Created {X_poly.shape[1]} polynomial features")
                    st.dataframe(X_poly.head(), use_container_width=True)
                    st.session_state["X_poly"] = X_poly
        
        with fe_tabs[1]:
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if not date_cols:
                # Try to find columns that might be dates
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            if date_cols:
                date_col = st.selectbox("Select Date Column", date_cols)
                if st.button("Extract Date Features"):
                    df_dates = extract_date_features(df, date_col)
                    st.success(f"✅ Extracted date features from {date_col}")
                    st.dataframe(df_dates.head(), use_container_width=True)
                    st.session_state["df_with_dates"] = df_dates
            else:
                st.info("No date columns detected in your dataset")
        
        with fe_tabs[2]:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                col_to_bin = st.selectbox("Select Column to Bin", numeric_cols)
                method = st.selectbox("Binning Method", ["equal_width", "quantile"])
                n_bins = st.slider("Number of Bins", 2, 20, 5)
                
                if st.button("Apply Binning"):
                    df_binned = apply_binning(df, col_to_bin, method=method, n_bins=n_bins)
                    st.success(f"✅ Applied {method} binning to {col_to_bin}")
                    st.dataframe(df_binned[[col_to_bin, f"{col_to_bin}_binned"]].head(10), use_container_width=True)
                    st.session_state["df_binned"] = df_binned
        
        with fe_tabs[3]:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) >= 2:
                col1_interact = st.selectbox("First Feature", numeric_cols, key="int_col1")
                col2_interact = st.selectbox("Second Feature", numeric_cols, key="int_col2")
                
                if st.button("Create Interaction"):
                    if col1_interact == col2_interact:
                        st.warning("⚠️ Please select two different features for interaction")
                    else:
                        df_interact = create_feature_interactions(df, col1_interact, col2_interact)
                        st.success(f"✅ Created {col1_interact} × {col2_interact}")
                        display_cols = [col1_interact, col2_interact, f"{col1_interact}_x_{col2_interact}"]
                        st.dataframe(df_interact[display_cols].head(), use_container_width=True)
                        st.session_state["df_interact"] = df_interact
        
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # MISSING VALUES
    # =========================
    elif "Missing Values" in option:
        st.header("Missing Values Analysis")

        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if len(missing) > 0:
            st.write(missing)

            fig = px.bar(
                x=missing.index,
                y=missing.values,
                labels={'x': 'Columns', 'y': 'Missing Count'},
                title="Missing Values"
            )
            st.plotly_chart(fig)
        else:
            st.success("No Missing Values!")

    # =========================
    # DATA CLEANING
    # =========================
    elif "Data Cleaning" in option:
        st.header("Handle Missing Values")

        df_clean = df.copy()
        num_cols = df_clean.select_dtypes(include=np.number).columns
        cat_cols = df_clean.select_dtypes(exclude=np.number).columns

        st.subheader("Select Columns to Impute")
        selected_num = st.multiselect("Numeric columns", list(num_cols), default=list(num_cols))
        selected_cat = st.multiselect("Categorical columns", list(cat_cols), default=list(cat_cols))

        method = st.selectbox("Numeric Imputation", ["mean", "median"])

        if st.button("Apply Cleaning"):
            if len(selected_num) > 0:
                num_imputer = SimpleImputer(strategy=method)
                df_clean[selected_num] = num_imputer.fit_transform(df_clean[selected_num])

            if len(selected_cat) > 0:
                cat_imputer = SimpleImputer(strategy="most_frequent")
                df_clean[selected_cat] = cat_imputer.fit_transform(df_clean[selected_cat])

            st.success("Missing values handled for selected columns!")
            st.write(df_clean.head())

    # =========================
    # ENCODING
    # =========================
    elif "Encoding" in option:
        st.header("Categorical Encoding")

        df_enc = df.copy()
        cat_cols = df_enc.select_dtypes(exclude=np.number).columns

        if len(cat_cols) > 0:
            st.write("Categorical Columns:", list(cat_cols))

            if st.button("Apply One-Hot Encoding"):
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(df_enc[cat_cols])

                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
                df_enc = df_enc.drop(cat_cols, axis=1)
                df_enc = pd.concat([df_enc, encoded_df], axis=1)

                st.success("Encoding Applied!")
                st.write(df_enc.head())
        else:
            st.warning("No categorical columns found!")

    # =========================
    # SCALING
    # =========================
    elif "Scaling" in option:
        st.header("Feature Scaling")

        df_scale = df.copy()
        num_cols = df_scale.select_dtypes(include=np.number).columns

        scaler_type = st.selectbox("Choose Scaler", ["StandardScaler", "MinMaxScaler"])

        if st.button("Apply Scaling"):
            if scaler_type == "StandardScaler":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()

            df_scale[num_cols] = scaler.fit_transform(df_scale[num_cols])

            st.success(f"{scaler_type} Applied!")
            st.write(df_scale.head())

    # =========================
    # EDA
    # =========================
    elif "EDA" in option:
        st.header("Exploratory Data Analysis")

        col = st.selectbox("Select Column", df.columns)

        if df[col].dtype == 'object':
            st.subheader("Count Plot")
            count_df = df[col].value_counts().rename_axis(col).reset_index(name='count')
            fig = px.bar(count_df,
                         x=col, y='count',
                         labels={col: col, 'count': 'Count'})
            st.plotly_chart(fig)

        else:
            st.subheader("Histogram")
            fig = px.histogram(df, x=col)
            st.plotly_chart(fig)

            st.subheader("Box Plot")
            fig = px.box(df, y=col)
            st.plotly_chart(fig)

        st.subheader("Correlation Heatmap")
        corr = df.select_dtypes(include=np.number).corr()
        fig = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig)

    # =========================
    # PCA
    # =========================
    elif "PCA" in option:
        st.header("Principal Component Analysis")

        df_pca = df.copy()
        numeric_cols = df_pca.select_dtypes(include=np.number).columns

        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for PCA.")
        else:
            selected_cols = st.multiselect("Numeric columns for PCA", list(numeric_cols), default=list(numeric_cols))

            if len(selected_cols) == 0:
                st.warning("Please select at least one numeric column.")
            else:
                max_comps = min(len(selected_cols), 10)
                n_components = st.slider("Number of PCA components", min_value=1, max_value=max_comps, value=min(2, max_comps))

                if st.button("Run PCA"):
                    df_selected = df_pca[selected_cols].dropna()

                    if df_selected.empty:
                        st.warning("The selected data contains only missing values after dropna().")
                    else:
                        scaler = StandardScaler()
                        scaled = scaler.fit_transform(df_selected)
                        pca = PCA(n_components=n_components)
                        components = pca.fit_transform(scaled)

                        explained_variance = pca.explained_variance_ratio_
                        cumulative_variance = np.cumsum(explained_variance)
                        result_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)])

                        st.subheader("Explained Variance")
                        variance_df = pd.DataFrame({
                            "Principal Component": [f"PC{i+1}" for i in range(n_components)],
                            "Explained Variance Ratio": explained_variance,
                            "Cumulative Variance": cumulative_variance
                        })
                        st.write(variance_df)

                        fig = px.bar(variance_df,
                                     x="Principal Component",
                                     y="Explained Variance Ratio",
                                     title="PCA Explained Variance Ratio")
                        st.plotly_chart(fig)

                        if n_components >= 2:
                            st.subheader("PCA Projection (2D)")
                            fig2 = px.scatter(result_df, x="PC1", y="PC2",
                                              title="PCA 2D Projection",
                                              labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"})
                            st.plotly_chart(fig2)

                        if n_components >= 3:
                            st.subheader("PCA Projection (3D)")
                            fig3 = px.scatter_3d(result_df, x="PC1", y="PC2", z="PC3",
                                                 title="PCA 3D Projection")
                            st.plotly_chart(fig3)

                        st.subheader("PCA Components Preview")
                        st.write(result_df.head())

    # =========================
    # Model Training
    # =========================
    elif "Model Training" in option:
        st.markdown('<h1 class="section-header">🤖 Advanced Model Training</h1>', unsafe_allow_html=True)

        if df.empty:
            st.warning("Uploaded dataset is empty.")
        else:
            target_column = st.selectbox("Select Target Column", df.columns)

            if target_column:
                # Feature selection
                available_features = [col for col in df.columns if col != target_column]
                selected_features = st.multiselect(
                    "Select Feature Columns",
                    available_features,
                    default=available_features,
                    help="Choose which columns to use as features for training"
                )

                if not selected_features:
                    st.warning("Please select at least one feature column.")
                else:
                    X = df[selected_features].copy()
                    y = df[target_column].copy()

                    st.subheader("Data Preparation")
                    st.write(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")

                    if X.empty:
                        st.warning("No feature columns available.")
                    else:
                        if X.isnull().any().any() or y.isnull().any():
                            st.info("Missing values detected and will be imputed automatically.")

                        if y.isnull().any():
                            if pd.api.types.is_numeric_dtype(y):
                                y = y.fillna(y.median())
                            else:
                                y = y.fillna(y.mode().iloc[0] if not y.mode().empty else "")

                        is_classification = detect_task_type(y)
                        task_type = "Classification" if is_classification else "Regression"
                        st.markdown(f"**Detected task:** {task_type}")

                        model_options = get_model_options(is_classification)

                        st.subheader("🛠️ Model & Hyperparameter Configuration")
                        chosen_name = st.selectbox("Select Model", list(model_options.keys()), index=0)
                        manual_params = get_manual_hyperparameters(chosen_name)
                        chosen_model = build_model_with_params(chosen_name, manual_params, is_classification)

                        if manual_params:
                            st.markdown("**Manual hyperparameters selected:**")
                            st.write(manual_params)
                        else:
                            st.info("Using default hyperparameters for the selected model.")

                        enable_feature_selection = st.checkbox("Enable Feature Selection")
                        k_features = None
                        preview_preprocessor = build_preprocessing_pipeline(X, is_classification, feature_selection_k=None)
                        if enable_feature_selection:
                            try:
                                transformed_feature_count = len(preview_preprocessor.named_steps["preprocessor"].get_feature_names_out())
                            except Exception:
                                transformed_feature_count = X.shape[1]

                            if transformed_feature_count <= 1:
                                st.warning("Feature selection requires at least 2 features. Using all available features.")
                                k_features = transformed_feature_count
                            else:
                                max_k = min(transformed_feature_count, 20)
                                k_features = st.slider("Number of features to select", 1, max_k, value=min(max_k, 10))
                                st.write(f"Feature selection will select the top {k_features} transformed features.")
                        else:
                            st.write("Feature selection is disabled.")

                        pipeline_preprocessor = build_preprocessing_pipeline(
                            X,
                            is_classification,
                            feature_selection_k=k_features if enable_feature_selection else None,
                        )

                        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

                        if pipeline_preprocessor is None:
                            st.warning("No valid preprocessing pipeline could be built. Check your feature set.")
                        else:
                            col1, col2, col3 = st.columns(3)
                            compare_models = col1.button("Compare All Models", use_container_width=True)
                            find_best_model = col2.button("Find Best Model Automatically", use_container_width=True)
                            optimize_model = col3.button("Optimize Model", use_container_width=True)

                        if compare_models:
                            st.subheader("📊 Model Comparison")
                            st.info("Cross-validation helps compare model performance.")
                            comparison_results = []

                            for model_name, model in model_options.items():
                                try:
                                    pipeline = Pipeline([("preprocessing", pipeline_preprocessor), ("model", model)])
                                    scoring = 'accuracy' if is_classification else 'r2'
                                    scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring)
                                    comparison_results.append({
                                        "Model Name": model_name,
                                        "Score": f"{scores.mean():.3f}",
                                        "CV Score": f"{scores.mean():.3f} ± {scores.std():.3f}",
                                    })
                                except Exception as e:
                                    comparison_results.append({
                                        "Model Name": model_name,
                                        "Score": "Error",
                                        "CV Score": str(e),
                                    })

                            comparison_df = pd.DataFrame(comparison_results)
                            numeric_scores = pd.to_numeric(comparison_df["Score"], errors='coerce')
                            best_idx = numeric_scores.idxmax() if not numeric_scores.dropna().empty else None

                            def highlight_best(row):
                                if best_idx is not None and row.name == best_idx:
                                    return ['background-color: #d4edda'] * len(row)
                                return [''] * len(row)

                            st.dataframe(comparison_df.style.apply(highlight_best, axis=1))

                        if find_best_model:
                            st.subheader("🤖 AutoML: Best Model Selection")
                            best_score = -np.inf
                            best_model_name = None
                            best_model = None

                            for model_name, model in model_options.items():
                                try:
                                    pipeline = Pipeline([("preprocessing", pipeline_preprocessor), ("model", model)])
                                    scoring = 'accuracy' if is_classification else 'r2'
                                    scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring)
                                    score = scores.mean()
                                    if score > best_score:
                                        best_score = score
                                        best_model_name = model_name
                                        best_model = model
                                except Exception:
                                    continue

                            if best_model is not None:
                                st.session_state["best_model"] = best_model
                                st.session_state["best_model_name"] = best_model_name
                                st.session_state["best_score"] = best_score
                                st.success(f"✅ Best model selected: {best_model_name} ({best_score:.3f})")
                                st.metric("Best Model", best_model_name)
                                st.metric("Best Score", f"{best_score:.3f}")
                            else:
                                st.error("AutoML could not identify a valid model.")

                        if optimize_model:
                            st.subheader("🔧 Hyperparameter Tuning")
                            tuning_model_name = st.selectbox("Select model to optimize", list(model_options.keys()), key="optimize_model")

                            with st.expander("Tuning grid options", expanded=True):
                                tuning_grid = display_tuning_controls(tuning_model_name)

                            if st.button("Start Optimization"):
                                if tuning_grid:
                                    pipeline = Pipeline([("preprocessing", pipeline_preprocessor), ("model", model_options[tuning_model_name])])
                                    scoring = 'accuracy' if is_classification else 'r2'
                                    grid_search = GridSearchCV(
                                        pipeline,
                                        tuning_grid,
                                        cv=3,
                                        scoring=scoring,
                                        n_jobs=-1,
                                    )
                                    try:
                                        grid_search.fit(X, y)
                                        st.session_state["optimized_model"] = grid_search.best_estimator_.named_steps["model"]
                                        st.session_state["optimized_model_name"] = f"{tuning_model_name} (optimized)"
                                        st.session_state["best_params"] = grid_search.best_params_
                                        st.session_state["best_cv_score"] = grid_search.best_score_
                                        st.success("Optimization completed!")
                                        st.write("**Best Parameters:**", grid_search.best_params_)
                                        st.metric("Best CV Score", f"{grid_search.best_score_:.3f}")
                                    except Exception as e:
                                        st.error(f"Optimization failed: {e}")
                                else:
                                    st.warning("Please select at least one hyperparameter value range for tuning.")

                        # ===== FEATURE 5: CLASS IMBALANCE DETECTION & HANDLING =====
                        st.subheader("⚖️ Data Balance Check")
                        if is_classification:
                            is_imbalanced, imbalance_ratio = detect_class_imbalance(y, threshold=0.6)
                            
                            if is_imbalanced is not None:
                                if is_imbalanced:
                                    st.warning(f"⚠️ Class imbalance detected (ratio: {imbalance_ratio:.3f})")
                                    apply_smote_flag = st.checkbox("Apply SMOTE to handle class imbalance?")
                                    if apply_smote_flag:
                                        st.info("SMOTE will be applied during training to oversample minority class")
                                        st.session_state["apply_smote"] = True
                                    else:
                                        st.session_state["apply_smote"] = False
                                else:
                                    st.success("✅ Classes are well-balanced")
                                    st.session_state["apply_smote"] = False
                            else:
                                st.info("Dataset has only one class or multiclass (imbalance check skipped)")
                                st.session_state["apply_smote"] = False
                        else:
                            st.session_state["apply_smote"] = False

                        st.subheader("🎯 Training")
                        if st.session_state.get("optimized_model") is not None:
                            chosen_name = st.session_state.get("optimized_model_name", "Optimized Model")
                            chosen_model = st.session_state["optimized_model"]
                            st.info(f"Using optimized model: {chosen_name}")
                        elif st.session_state.get("best_model") is not None:
                            chosen_name = st.session_state.get("best_model_name", "Best Model")
                            chosen_model = st.session_state["best_model"]
                            st.info(f"Using best model: {chosen_name}")
                        else:
                            # Use the manually selected model and hyperparameters.
                            st.info(f"Using manual model: {chosen_name}")
                            chosen_model = chosen_model

                        if st.button("Train Model"):
                            if X.shape[0] < 2 or X.shape[1] == 0 or y.nunique() < 2:
                                st.warning("Not enough data to train a reliable model.")
                            else:
                                try:
                                    split_args = {"test_size": test_size, "random_state": 42}
                                    if is_classification:
                                        split_args["stratify"] = y
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_args)
                                except ValueError:
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                                # ===== FEATURE 5: APPLY SMOTE IF ENABLED =====
                                if st.session_state.get("apply_smote", False):
                                    st.info("Applying SMOTE to training data...")
                                    X_train_before = X_train.copy()
                                    X_train, y_train = apply_smote(X_train, y_train)
                                    st.success(f"✅ SMOTE applied: {len(X_train_before)} → {len(X_train)} training samples")

                                pipeline = Pipeline([
                                    ("preprocessing", pipeline_preprocessor),
                                    ("model", chosen_model),
                                ])
                                try:
                                    pipeline.fit(X_train, y_train)
                                except Exception as err:
                                    st.error(f"Model training failed: {err}")
                                else:
                                    st.session_state["trained_pipeline"] = pipeline
                                    st.session_state["training_features"] = X.columns.tolist()
                                    st.session_state["training_schema"] = create_prediction_schema(X)
                                    st.session_state["trained_model_name"] = chosen_name
                                    st.session_state["model_type"] = task_type
                                    st.session_state["last_model_name"] = chosen_name
                                    y_pred = pipeline.predict(X_test)
                                    score = accuracy_score(y_test, y_pred) if is_classification else r2_score(y_test, y_pred)
                                    timestamp = format_timestamp()
                                    st.session_state["last_score"] = score
                                    st.session_state["last_timestamp"] = timestamp
                                    st.session_state["experiment_history"].append({
                                        "Model": chosen_name,
                                        "Score": f"{score:.3f}",
                                        "Time": timestamp,
                                    })
                                    st.session_state["model_download_blob"] = save_model_blob(pipeline, X.columns.tolist())
                                    st.success("Model trained successfully!")

                                    st.subheader("📈 Evaluation")
                                    if is_classification:
                                        # ===== FEATURE 4: ADVANCED CLASSIFICATION METRICS =====
                                        metrics_dict = get_classification_metrics(y_test, y_pred, pipeline.predict_proba(X_test) if hasattr(pipeline.named_steps["model"], 'predict_proba') else None)
                                        
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Accuracy", f"{metrics_dict['Accuracy']:.3f}")
                                        with col2:
                                            st.metric("Precision", f"{metrics_dict['Precision']:.3f}")
                                        with col3:
                                            st.metric("Recall", f"{metrics_dict['Recall']:.3f}")
                                        with col4:
                                            st.metric("F1-Score", f"{metrics_dict['F1-Score']:.3f}")
                                        
                                        if "ROC-AUC" in metrics_dict:
                                            st.metric("ROC-AUC", f"{metrics_dict['ROC-AUC']:.3f}")
                                        
                                        # Confusion Matrix
                                        cm = confusion_matrix(y_test, y_pred)
                                        fig = px.imshow(
                                            cm,
                                            labels={"x": "Predicted", "y": "Actual"},
                                            x=np.arange(cm.shape[1]).astype(str),
                                            y=np.arange(cm.shape[0]).astype(str),
                                            text_auto=True,
                                            color_continuous_scale="Blues",
                                            title="Confusion Matrix",
                                        )
                                        st.plotly_chart(fig)
                                        
                                        # ROC and Precision-Recall curves for binary classification
                                        if len(np.unique(y_test)) == 2 and hasattr(pipeline.named_steps["model"], 'predict_proba'):
                                            try:
                                                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                                                model = pipeline.named_steps["model"]
                                                pos_label = None
                                                if hasattr(model, 'classes_') and len(model.classes_) == 2:
                                                    pos_label = model.classes_[1]
                                                
                                                if pos_label is None:
                                                    y_test_series = pd.Series(y_test).reset_index(drop=True)
                                                    unique_labels = y_test_series.unique()
                                                    if len(unique_labels) == 2:
                                                        pos_label = unique_labels[1]
                                                
                                                fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=pos_label)
                                                precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba, pos_label=pos_label)
                                                
                                                col_roc, col_pr = st.columns(2)
                                                with col_roc:
                                                    fig_roc = px.line(x=fpr, y=tpr, title="ROC Curve", 
                                                                     labels={"x": "False Positive Rate", "y": "True Positive Rate"})
                                                    fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', 
                                                                       line=dict(dash='dash', color='gray'))
                                                    st.plotly_chart(fig_roc, use_container_width=True)
                                                
                                                with col_pr:
                                                    fig_pr = px.line(x=recall_curve, y=precision_curve, title="Precision-Recall Curve",
                                                                    labels={"x": "Recall", "y": "Precision"})
                                                    st.plotly_chart(fig_pr, use_container_width=True)
                                            except Exception as e:
                                                st.warning(f"⚠️ Could not generate ROC/PR curves: {str(e)}")
                                    else:
                                        # ===== FEATURE 4: ADVANCED REGRESSION METRICS =====
                                        metrics_dict = get_regression_metrics(y_test, y_pred, n_features=len(selected_features))
                                        
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("R² Score", f"{metrics_dict['R² Score']:.3f}")
                                        with col2:
                                            st.metric("Adjusted R²", f"{metrics_dict['Adjusted R²']:.3f}")
                                        with col3:
                                            st.metric("MAE", f"{metrics_dict['MAE']:.3f}")
                                        with col4:
                                            st.metric("RMSE", f"{metrics_dict['RMSE']:.3f}")
                                        
                                        # Residuals plot
                                        residuals = y_test - y_pred
                                        fig_residuals = px.scatter(x=y_pred, y=residuals, title="Residual Plot",
                                                                   labels={"x": "Predicted Values", "y": "Residuals"})
                                        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                                        st.plotly_chart(fig_residuals)
                                    
                                    # ===== FEATURE 6: OVERFITTING DETECTION =====
                                    st.markdown("---")
                                    train_score = accuracy_score(y_train, pipeline.predict(X_train)) if is_classification else r2_score(y_train, pipeline.predict(X_train))
                                    test_score = score
                                    
                                    status, message = detect_overfitting(train_score, test_score, threshold=0.15)
                                    
                                    if status == "Overfitting":
                                        st.warning(f"⚠️ {status} detected: {message}")
                                        st.info("Suggestions: Reduce model complexity, add regularization, collect more data, or apply early stopping")
                                    elif status == "Underfitting":
                                        st.warning(f"⚠️ {status} detected: {message}")
                                        st.info("Suggestions: Increase model complexity, add more features, reduce regularization, or improve data quality")
                                    else:
                                        st.success(f"✅ {status}: {message}")
                                    
                                    col_train, col_test = st.columns(2)
                                    with col_train:
                                        st.metric("Train Score", f"{train_score:.3f}")
                                    with col_test:
                                        st.metric("Test Score", f"{test_score:.3f}")


                                    st.subheader("📚 Learning Curve")
                                    try:
                                        scoring = 'accuracy' if is_classification else 'r2'
                                        train_sizes, train_scores, val_scores = learning_curve(
                                            pipeline,
                                            X,
                                            y,
                                            cv=3,
                                            scoring=scoring,
                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                        )
                                        train_mean = np.mean(train_scores, axis=1)
                                        train_std = np.std(train_scores, axis=1)
                                        val_mean = np.mean(val_scores, axis=1)
                                        val_std = np.std(val_scores, axis=1)
                                        fig_lc = px.line(
                                            x=train_sizes,
                                            y=train_mean,
                                            labels={"x": "Training Set Size", "y": "Score"},
                                            title="Learning Curve",
                                        )
                                        fig_lc.add_scatter(x=train_sizes, y=val_mean, mode='lines', name='Validation Score')
                                        fig_lc.add_scatter(x=train_sizes, y=train_mean + train_std, mode='lines', line=dict(width=0), showlegend=False)
                                        fig_lc.add_scatter(x=train_sizes, y=train_mean - train_std, mode='lines', line=dict(width=0), fill='tonexty', showlegend=False)
                                        fig_lc.add_scatter(x=train_sizes, y=val_mean + val_std, mode='lines', line=dict(width=0), showlegend=False)
                                        fig_lc.add_scatter(x=train_sizes, y=val_mean - val_std, mode='lines', line=dict(width=0), fill='tonexty', showlegend=False)
                                        st.plotly_chart(fig_lc)
                                    except Exception as e:
                                        st.warning(f"Could not generate learning curve: {e}")

                                    importance_df = None
                                    model_object = pipeline.named_steps["model"]
                                    try:
                                        feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
                                    except Exception:
                                        feature_names = []
                                    if hasattr(model_object, "feature_importances_") and len(feature_names) == len(model_object.feature_importances_):
                                        importance_df = pd.DataFrame({
                                            "feature": feature_names,
                                            "importance": model_object.feature_importances_,
                                        }).sort_values("importance", ascending=False)
                                    elif hasattr(model_object, "coef_") and len(feature_names) == len(np.atleast_1d(model_object.coef_).reshape(-1)):
                                        coefs = model_object.coef_
                                        if coefs.ndim == 1:
                                            importance_df = pd.DataFrame({
                                                "feature": feature_names,
                                                "importance": np.abs(coefs),
                                            }).sort_values("importance", ascending=False)
                                        else:
                                            importance_df = pd.DataFrame({
                                                "feature": feature_names,
                                                "importance": np.abs(coefs).mean(axis=0),
                                            }).sort_values("importance", ascending=False)
                                    if importance_df is not None:
                                        st.subheader("Feature Importance")
                                        st.write(importance_df.head(10))
                                        fig_imp = px.bar(
                                            importance_df.head(10),
                                            x="importance",
                                            y="feature",
                                            orientation="h",
                                            title="Top Feature Importances",
                                        )
                                        st.plotly_chart(fig_imp)

                        st.subheader("🔮 Make Predictions")
                        if st.session_state.get("trained_pipeline") is None:
                            st.info("Train a model first to enable predictions and downloads.")
                        else:
                            prediction_input = get_prediction_input(st.session_state["training_schema"])
                            if st.button("Predict"):
                                try:
                                    prediction = st.session_state["trained_pipeline"].predict(prediction_input)
                                    st.success(f"Prediction: {prediction[0]}")
                                except Exception as e:
                                    st.error(f"Prediction failed: {e}")

                            if st.session_state.get("model_download_blob") is not None:
                                st.download_button(
                                    "Download Model",
                                    st.session_state["model_download_blob"],
                                    file_name="model.pkl",
                                    mime="application/octet-stream",
                                )

                        st.subheader("📜 History")
                        if st.session_state["experiment_history"]:
                            history_df = pd.DataFrame(st.session_state["experiment_history"])
                            st.table(history_df)
                        else:
                            st.info("No experiments tracked yet.")

                        if st.session_state.get("trained_pipeline") is not None:
                            trained_model_object = st.session_state["trained_pipeline"].named_steps["model"]
                            if isinstance(trained_model_object, (RandomForestClassifier, RandomForestRegressor)):
                                enable_shap = st.checkbox("Enable SHAP (may be slow)")
                                if enable_shap:
                                    try:
                                        import shap
                                        with st.spinner("Computing SHAP..."):
                                            sample_size = min(100, len(X))
                                            X_sample = X.sample(sample_size, random_state=42)
                                            transformed = st.session_state["trained_pipeline"].named_steps["preprocessing"].transform(X_sample)
                                            explainer = shap.TreeExplainer(trained_model_object)
                                            shap_values = explainer.shap_values(transformed)
                                            st.write("**SHAP Summary Plot**")
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            if is_classification and isinstance(shap_values, list) and len(shap_values) > 1:
                                                shap.summary_plot(shap_values[1], transformed, show=False)
                                            else:
                                                shap.summary_plot(shap_values, transformed, show=False)
                                            st.pyplot(fig)
                                            if is_classification and isinstance(shap_values, list) and len(shap_values) > 1:
                                                shap_importance = np.abs(shap_values[1]).mean(axis=0)
                                            else:
                                                shap_importance = np.abs(shap_values).mean(axis=0)
                                            try:
                                                shap_features = st.session_state["trained_pipeline"].named_steps["preprocessing"].get_feature_names_out()
                                            except Exception:
                                                shap_features = [f"feature_{i}" for i in range(len(shap_importance))]
                                            shap_df = pd.DataFrame({
                                                "feature": shap_features,
                                                "shap_importance": shap_importance,
                                            }).sort_values("shap_importance", ascending=False)
                                            st.write(shap_df.head(10))
                                    except ImportError:
                                        st.warning("SHAP library not installed. Install with: pip install shap")
                                    except Exception as e:
                                        st.warning(f"SHAP analysis failed: {e}")

    # ===== FEATURE 11: EXPERIMENT TRACKING WITH REPORT & API =====
    elif "Advanced Evaluation" in option or "Experiment Tracking" in option:
        st.markdown('<h1 class="section-header">📋 Advanced Evaluation & Experiment Tracking</h1>', unsafe_allow_html=True)
        
        if not st.session_state.get("experiment_history"):
            st.info("No experiments tracked yet. Train models in the Model Training section to populate history.")
        else:
            # Display experiment history
            st.subheader("📊 Experiment History")
            history_df = pd.DataFrame(st.session_state["experiment_history"])
            st.dataframe(history_df, use_container_width=True)
            
            # Sort and filter options
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox("Sort by", ["Model", "Score", "Time"])
            with col2:
                sort_desc = st.checkbox("Descending", value=True)
            
            history_df_sorted = history_df.sort_values(by=sort_by, ascending=not sort_desc)
            st.dataframe(history_df_sorted, use_container_width=True)
            
            # Download history
            csv_history = history_df_sorted.to_csv(index=False)
            st.download_button(
                label="📥 Download Experiment History (CSV)",
                data=csv_history,
                file_name="experiment_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # ===== FEATURE 11: REPORT GENERATION =====
        if st.session_state.get("trained_pipeline") is not None:
            st.markdown("---")
            st.subheader("📄 Generate Report")
            
            if st.button("Generate Full Report"):
                report = generate_experiment_report(
                    df,
                    st.session_state.get("training_features", []),
                    st.session_state.get("trained_model_name", "Unknown"),
                    {"Score": st.session_state.get("last_score", 0)},
                    st.session_state.get("last_timestamp", "")
                )
                
                st.markdown(report)
                
                # Download report
                st.download_button(
                    label="📥 Download Report (Markdown)",
                    data=report,
                    file_name="ml_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            # ===== FEATURE 12: API CODE GENERATION =====
            st.markdown("---")
            st.subheader("⚡ FastAPI Code Generator")
            
            if st.button("Generate API Code"):
                api_code = generate_fastapi_code(
                    st.session_state.get("trained_model_name", "model"),
                    st.session_state.get("training_features", [])
                )
                
                st.code(api_code, language="python")
                
                st.download_button(
                    label="📥 Download API Code",
                    data=api_code,
                    file_name="api.py",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # ===== FEATURE 13: MODEL JUSTIFICATION =====
            st.markdown("---")
            st.subheader("🎯 Model Explanation & Justification")
            
            model_name = st.session_state.get("trained_model_name", "Unknown")
            if model_name in MODEL_EXPLANATIONS:
                explanation = MODEL_EXPLANATIONS[model_name]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 💪 Strengths")
                    st.write(explanation["strengths"])
                
                with col2:
                    st.markdown("### ⚠️ Weaknesses")
                    st.write(explanation["weaknesses"])
                
                with col3:
                    st.markdown("### 🎯 Best Use Cases")
                    st.write(explanation["use_cases"])
            else:
                st.info(f"Model explanation not available for {model_name}")

    # =========================
    # OUTLIER DETECTION
    # =========================
    elif "Outlier Detection" in option:
        st.header("Outlier Detection & Handling (IQR Method)")

        df_out = df.copy()
        num_cols = df_out.select_dtypes(include=np.number).columns

        if len(num_cols) == 0:
            st.warning("No numeric columns available for outlier detection.")
        else:
            col = st.selectbox("Select Column", num_cols)

            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = df_out[(df_out[col] < lower) | (df_out[col] > upper)]

            st.write(f"🔴 Number of Outliers: {len(outliers)}")

            st.subheader("Before Handling")
            fig1 = px.box(df_out, y=col, title="Before Capping")
            st.plotly_chart(fig1)

            method = st.selectbox("Choose Handling Method", ["None", "Capping"])

            if method == "Capping":
                if st.button("Apply Capping"):
                    df_out[col] = np.where(df_out[col] < lower, lower, df_out[col])
                    df_out[col] = np.where(df_out[col] > upper, upper, df_out[col])
                    st.success("✅ Capping Applied Successfully!")
                    st.subheader("After Capping")
                    fig2 = px.box(df_out, y=col, title="After Capping")
                    st.plotly_chart(fig2)
                    st.subheader("Updated Data Preview")
                    st.write(df_out.head())

    # =========================
    # Clustering with DBSCAN & Metrics
    # =========================
    elif "Clustering" in option:
        st.markdown('<h1 class="section-header">🎯 Advanced Clustering Analysis</h1>', unsafe_allow_html=True)

        if df.empty:
            st.warning("Uploaded dataset is empty.")
        else:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            if len(numeric_cols) == 0:
                st.warning("No numeric columns available for clustering.")
            else:
                st.subheader("Data Preparation")
                df_cluster = df[numeric_cols].copy()
                if df_cluster.isnull().any().any():
                    st.info("Missing values detected and will be imputed with median.")
                    imputer = SimpleImputer(strategy="median")
                    df_cluster = pd.DataFrame(
                        imputer.fit_transform(df_cluster),
                        columns=numeric_cols,
                        index=df_cluster.index
                    )

                scaler = StandardScaler()
                df_scaled = pd.DataFrame(
                    scaler.fit_transform(df_cluster),
                    columns=numeric_cols,
                    index=df_cluster.index
                )
                
                # Choose clustering algorithm
                clustering_algo = st.radio("Select Clustering Algorithm", ["K-Means", "DBSCAN"], horizontal=True)
                
                if clustering_algo == "K-Means":
                    st.subheader("📈 Elbow Method")
                    st.write("Find the optimal number of clusters (k) using the Elbow Method:")

                    wcss = []
                    k_range = range(1, 11)

                    for k in k_range:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(df_scaled)
                        wcss.append(kmeans.inertia_)

                    elbow_df = pd.DataFrame({"k": k_range, "WCSS": wcss})
                    fig_elbow = px.line(
                        elbow_df, x="k", y="WCSS",
                        markers=True,
                        title="Elbow Method: WCSS vs Number of Clusters",
                        labels={"k": "Number of Clusters (k)", "WCSS": "Within-Cluster Sum of Squares"}
                    )
                    fig_elbow.update_traces(mode="lines+markers")
                    st.plotly_chart(fig_elbow)

                    k_clusters = st.slider("Number of Clusters (k)", 2, 10, 3)

                    if st.button("Perform K-Means Clustering"):
                        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(df_scaled)

                        df_with_clusters = df.copy()
                        df_with_clusters["Cluster"] = clusters

                        st.success(f"✅ K-Means clustering completed! Data divided into {k_clusters} clusters.")
                        
                        # ===== FEATURE 9: SILHOUETTE SCORE =====
                        sil_score = silhouette_score(df_scaled, clusters)
                        st.metric("Silhouette Score", f"{sil_score:.3f}")
                        if sil_score > 0.5:
                            st.success("✅ Excellent cluster separation")
                        elif sil_score > 0.3:
                            st.info("⚠️ Good cluster separation")
                        else:
                            st.warning("⚠️ Poor cluster separation - consider adjusting k")

                        st.subheader("📊 2D Cluster Visualization (PCA)")
                        pca_2d = PCA(n_components=2, random_state=42)
                        pca_components = pca_2d.fit_transform(df_scaled)

                        pca_df = pd.DataFrame({
                            "PC1": pca_components[:, 0],
                            "PC2": pca_components[:, 1],
                            "Cluster": clusters.astype(str)
                        })

                        fig_2d = px.scatter(
                            pca_df, x="PC1", y="PC2", color="Cluster",
                            title=f"K-Means Clustering (k={k_clusters}) - 2D PCA Projection",
                            labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
                            color_discrete_sequence=px.colors.qualitative.Set1
                        )
                        st.plotly_chart(fig_2d)

                        if df_scaled.shape[1] >= 3:
                            st.subheader("📊 3D Cluster Visualization (PCA)")
                            pca_3d = PCA(n_components=3, random_state=42)
                            pca_components_3d = pca_3d.fit_transform(df_scaled)

                            pca_df_3d = pd.DataFrame({
                                "PC1": pca_components_3d[:, 0],
                                "PC2": pca_components_3d[:, 1],
                                "PC3": pca_components_3d[:, 2],
                                "Cluster": clusters.astype(str)
                            })

                            fig_3d = px.scatter_3d(
                                pca_df_3d, x="PC1", y="PC2", z="PC3", color="Cluster",
                                title=f"K-Means Clustering (k={k_clusters}) - 3D PCA Projection",
                                color_discrete_sequence=px.colors.qualitative.Set1
                            )
                            st.plotly_chart(fig_3d)

                        st.subheader("📋 Cluster Summary")
                        cluster_summary = df_with_clusters.groupby("Cluster").agg({
                            **{col: "mean" for col in numeric_cols},
                            "Cluster": "count"
                        }).rename(columns={"Cluster": "Count"}).round(3)
                    st.write("Cluster Statistics (means and counts):")
                    st.dataframe(cluster_summary)

                    cluster_counts = df_with_clusters["Cluster"].value_counts().sort_index()
                    fig_dist = px.bar(
                        cluster_counts,
                        title="Cluster Distribution",
                        labels={"index": "Cluster", "value": "Number of Samples"},
                        color=cluster_counts.index.astype(str),
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    st.plotly_chart(fig_dist)

                    st.subheader("💾 Download Clustered Data")
                    csv_data = df_with_clusters.to_csv(index=False)
                    st.download_button(
                        label="Download Clustered Data (CSV)",
                        data=csv_data,
                        file_name=f"clustered_data_k{k_clusters}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # ===== FEATURE 9: DBSCAN CLUSTERING =====
                elif clustering_algo == "DBSCAN":
                    st.subheader("🔍 DBSCAN Parameters")
                    eps = st.slider("Epsilon (eps) - Maximum distance between points", 0.1, 5.0, 0.5, 0.1)
                    min_samples = st.slider("Minimum samples in neighborhood", 2, 20, 5)
                    
                    if st.button("Perform DBSCAN Clustering"):
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        clusters = dbscan.fit_predict(df_scaled)
                        
                        n_clusters_dbscan = len(set(clusters)) - (1 if -1 in clusters else 0)
                        n_noise = list(clusters).count(-1)
                        
                        df_with_clusters = df.copy()
                        df_with_clusters["Cluster"] = clusters
                        
                        st.success(f"✅ DBSCAN completed! Found {n_clusters_dbscan} clusters and {n_noise} noise points.")
                        
                        # ===== FEATURE 9: SILHOUETTE SCORE FOR DBSCAN =====
                        if n_clusters_dbscan > 1 and n_noise < len(clusters):
                            # Only compute silhouette for non-noise points
                            non_noise_mask = clusters != -1
                            if non_noise_mask.sum() > 0:
                                sil_score = silhouette_score(df_scaled[non_noise_mask], clusters[non_noise_mask])
                                st.metric("Silhouette Score", f"{sil_score:.3f}")
                                if sil_score > 0.5:
                                    st.success("✅ Excellent cluster separation")
                                elif sil_score > 0.3:
                                    st.info("⚠️ Good cluster separation")
                                else:
                                    st.warning("⚠️ Poor cluster separation - consider adjusting eps or min_samples")
                        
                        st.subheader("📊 2D DBSCAN Visualization (PCA)")
                        pca_2d = PCA(n_components=2, random_state=42)
                        pca_components = pca_2d.fit_transform(df_scaled)
                        
                        pca_df = pd.DataFrame({
                            "PC1": pca_components[:, 0],
                            "PC2": pca_components[:, 1],
                            "Cluster": ["Noise" if c == -1 else f"C{c}" for c in clusters]
                        })
                        
                        fig_2d = px.scatter(
                            pca_df, x="PC1", y="PC2", color="Cluster",
                            title=f"DBSCAN Clustering (eps={eps}) - 2D PCA Projection",
                            labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"}
                        )
                        st.plotly_chart(fig_2d)
                        
                        if df_scaled.shape[1] >= 3:
                            st.subheader("📊 3D DBSCAN Visualization (PCA)")
                            pca_3d = PCA(n_components=3, random_state=42)
                            pca_components_3d = pca_3d.fit_transform(df_scaled)
                            
                            pca_df_3d = pd.DataFrame({
                                "PC1": pca_components_3d[:, 0],
                                "PC2": pca_components_3d[:, 1],
                                "PC3": pca_components_3d[:, 2],
                                "Cluster": ["Noise" if c == -1 else f"C{c}" for c in clusters]
                            })
                            
                            fig_3d = px.scatter_3d(
                                pca_df_3d, x="PC1", y="PC2", z="PC3", color="Cluster",
                                title=f"DBSCAN Clustering (eps={eps}) - 3D PCA Projection"
                            )
                            st.plotly_chart(fig_3d)
                        
                        st.subheader("📊 Cluster Distribution")
                        cluster_counts = df_with_clusters["Cluster"].value_counts()
                        fig_dist = px.bar(
                            cluster_counts,
                            title="Cluster Distribution (incl. Noise Points)",
                            labels={"index": "Cluster", "value": "Number of Samples"}
                        )
                        st.plotly_chart(fig_dist)
                        
                        st.subheader("💾 Download Clustered Data")
                        csv_data = df_with_clusters.to_csv(index=False)
                        st.download_button(
                            label="Download DBSCAN Clustered Data (CSV)",
                            data=csv_data,
                            file_name=f"dbscan_clustered_eps{eps}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )