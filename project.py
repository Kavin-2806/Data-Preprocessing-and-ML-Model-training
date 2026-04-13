import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
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
)
import matplotlib.pyplot as plt
import joblib
import tempfile
import os
import io
from datetime import datetime

st.set_page_config(layout="wide")

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
    .stSelectbox>div>div>div>div,
    .stTextArea>div>div>textarea,
    .stNumberInput>div>div>input {
        background: #f1f5f9 !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
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

st.title("📊 Data Preprocessing & EDA Dashboard ")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # =========================
    # SIDEBAR NAVIGATION
    # =========================
    option = st.sidebar.radio("Navigation", [
        "Dataset Overview",
        "Insights",
        "Missing Values",
        "Data Cleaning",
        "Encoding",
        "Scaling",
        "EDA",
        "Outlier Detection",
        "PCA",
        "Model Training",
        "Clustering"
    ])

    # =========================
    # DATASET OVERVIEW
    # =========================
    if option == "Dataset Overview":
        st.header("Dataset Overview")
        st.write(df.head())

        st.subheader("Shape")
        st.write(df.shape)

        st.subheader("Data Types")
        st.write(df.dtypes)

        st.subheader("Summary Statistics")
        st.write(df.describe())

    # =========================
    # INSIGHTS
    # =========================
    elif option == "Insights":
        st.header("📊 Insights")

        st.subheader("Missing Value Summary")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values detected.")
        else:
            st.table(missing.rename("Missing Count").to_frame())

        st.subheader("Numeric Feature Skewness")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for skewness analysis.")
        else:
            skewness = df[numeric_cols].skew().round(3)
            skew_df = skewness.to_frame(name="Skewness")
            st.table(skew_df)
            skewed = skewness[skewness.abs() > 0.5]
            if not skewed.empty:
                st.info(
                    "Highly skewed numeric columns: "
                    + ", ".join([f"{col} ({val:.2f})" for col, val in skewed.items()])
                )

        st.subheader("Highly Correlated Features")
        correlated_pairs = get_highly_correlated_pairs(df, threshold=0.8)
        if correlated_pairs:
            for x, y_col, corr_value in correlated_pairs:
                st.write(f"Feature **{x}** is highly correlated with **{y_col}** (correlation = {corr_value:.2f})")
        else:
            st.success("No feature pairs with correlation above 0.8 found.")

    # =========================
    # MISSING VALUES
    # =========================
    elif option == "Missing Values":
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
    elif option == "Data Cleaning":
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
    elif option == "Encoding":
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
    elif option == "Scaling":
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
    elif option == "EDA":
        st.header("Exploratory Data Analysis")

        col = st.selectbox("Select Column", df.columns)

        if df[col].dtype == 'object':
            st.subheader("Count Plot")
            fig = px.bar(df[col].value_counts().reset_index(),
                         x='index', y=col,
                         labels={'index': col, col: 'Count'})
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
    elif option == "PCA":
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
    elif option == "Model Training":
        st.header("🚀 Advanced Model Training & Evaluation")

        if df.empty:
            st.warning("Uploaded dataset is empty.")
        else:
            target_column = st.selectbox("Select Target Column", df.columns)

            if target_column:
                X = df.drop(columns=[target_column])
                y = df[target_column].copy()

                st.subheader("Data Preparation")
                st.write(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")

                if X.empty:
                    st.warning("No feature columns available after dropping the target.")
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

                    enable_feature_selection = st.checkbox("Enable Feature Selection")
                    k_features = None
                    preview_preprocessor = build_preprocessing_pipeline(X, is_classification, feature_selection_k=None)
                    if enable_feature_selection:
                        try:
                            transformed_feature_count = len(preview_preprocessor.named_steps["preprocessor"].get_feature_names_out())
                        except Exception:
                            transformed_feature_count = X.shape[1]
                        max_k = max(1, min(transformed_feature_count, 20))
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
                            if st.button("Start Optimization"):
                                param_grids = {
                                    "Random Forest": {"model__n_estimators": [50, 100], "model__max_depth": [None, 10, 20]},
                                    "Logistic Regression": {"model__C": [0.1, 1, 10]},
                                    "Decision Tree": {"model__max_depth": [None, 5, 10]},
                                    "Linear Regression": {},
                                    "Decision Tree Regressor": {"model__max_depth": [None, 5, 10]},
                                    "Random Forest Regressor": {"model__n_estimators": [50, 100], "model__max_depth": [None, 10, 20]},
                                    "KNN": {"model__n_neighbors": [3, 5, 7]},
                                }
                                if tuning_model_name in param_grids:
                                    grid = param_grids[tuning_model_name]
                                    if grid:
                                        pipeline = Pipeline([("preprocessing", pipeline_preprocessor), ("model", model_options[tuning_model_name])])
                                        scoring = 'accuracy' if is_classification else 'r2'
                                        grid_search = GridSearchCV(
                                            pipeline,
                                            grid,
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
                                        st.info("No hyperparameters available for this model.")
                                else:
                                    st.error("Selected model is not supported for tuning.")

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
                            chosen_name = st.selectbox("Select Model", list(model_options.keys()))
                            chosen_model = model_options[chosen_name]

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
                                        accuracy = accuracy_score(y_test, y_pred)
                                        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                                        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                                        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                                        metrics_df = pd.DataFrame({
                                            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                                            "Value": [f"{accuracy:.3f}", f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"]
                                        })
                                        st.table(metrics_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([
                                            {'selector': 'th', 'props': [('text-align', 'center')]},
                                            {'selector': 'td', 'props': [('text-align', 'center')]}
                                        ]))
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
                                    else:
                                        r2 = r2_score(y_test, y_pred)
                                        mae = mean_absolute_error(y_test, y_pred)
                                        mse = mean_squared_error(y_test, y_pred)
                                        c1, c2, c3 = st.columns(3)
                                        c1.metric("R2 Score", f"{r2:.3f}")
                                        c2.metric("MAE", f"{mae:.3f}")
                                        c3.metric("MSE", f"{mse:.3f}")

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

    # =========================
    # OUTLIER DETECTION
    # =========================
    elif option == "Outlier Detection":
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
    # Clustering
    # =========================
    elif option == "Clustering":
        st.header("🎯 K-Means Clustering Analysis")

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

                if st.button("Perform Clustering"):
                    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(df_scaled)

                    df_with_clusters = df.copy()
                    df_with_clusters["Cluster"] = clusters

                    st.success(f"✅ Clustering completed! Data divided into {k_clusters} clusters.")

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
