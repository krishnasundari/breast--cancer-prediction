"""
app.py
======
Interactive Streamlit Web Application for Breast Cancer Biopsy Prediction & Explainable AI.
Allows clinicians, researchers, and interviewers to compare multiple trained ML models
(KNN, Logistic Regression, SVM, Random Forest), test real-time inference, and inspect
global and local explanations using SHAP (SHapley Additive exPlanations).
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add repo root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import load_trained_pipeline, predict
from src.data_loader import load_raw_dataset
from src.explain import explain_instance, plot_local_waterfall

st.set_page_config(
    page_title="Breast Cancer Detection & Explainable AI Platform",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #4A148C;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8F9FA;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #7B1FA2;
        margin-bottom: 10px;
    }
    .benign-badge {
        background-color: #E8F5E9;
        color: #2E7D32;
        font-weight: bold;
        padding: 10px 18px;
        border-radius: 8px;
        font-size: 1.3rem;
        display: inline-block;
        border: 2px solid #A5D6A7;
    }
    .malignant-badge {
        background-color: #FFEBEE;
        color: #C62828;
        font-weight: bold;
        padding: 10px 18px;
        border-radius: 8px;
        font-size: 1.3rem;
        display: inline-block;
        border: 2px solid #EF9A9A;
    }
    .shap-positive {
        background-color: #FFF3E0;
        border-left: 4px solid #E65100;
        padding: 8px 12px;
        margin-bottom: 6px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .shap-negative {
        background-color: #E1F5FE;
        border-left: 4px solid #0288D1;
        padding: 8px 12px;
        margin-bottom: 6px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🎗️ Breast Cancer Detection & Explainable AI (XAI)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Clinical Diagnostic & Transparent Machine Learning Platform comparing <b>KNN, Logistic Regression, SVM, and Random Forest</b> with <b>SHAP Interpretability</b> on the Wisconsin Diagnostic Dataset.</div>', unsafe_allow_html=True)

# Sidebar - Model Selection
st.sidebar.header("🤖 Model Selection")
MODEL_NAME_TO_KEY = {
    "K-Nearest Neighbors (KNN)": "knn",
    "Support Vector Machine (SVM)": "svm",
    "Random Forest": "random_forest",
    "Logistic Regression": "logistic_regression"
}

selected_model_name = st.sidebar.selectbox(
    "Choose Active Classifier:",
    options=list(MODEL_NAME_TO_KEY.keys()),
    index=2  # Default to Random Forest for instant TreeSHAP
)
selected_model_key = MODEL_NAME_TO_KEY[selected_model_name]

# Load Model with caching
@st.cache_resource
def get_model(model_key: str):
    return load_trained_pipeline(model_key)

try:
    artifact = get_model(selected_model_key)
    pipeline = artifact["pipeline"]
    feature_names = artifact["feature_names"]
    metrics = artifact.get("metrics", {})
except Exception as e:
    st.error(f"Error loading model '{selected_model_name}': {e}. Please ensure 'python -m src.train' was executed.")
    st.stop()

# Sidebar - Biopsy Sample Selection & Feature Input
st.sidebar.header("🔬 Biopsy Feature Controls")

# Sample Presets
preset_choice = st.sidebar.selectbox(
    "Load Clinical Preset Sample:",
    ["Custom Input", "Sample A — Confirmed Malignant (Patient #842302)", "Sample B — Confirmed Benign (Patient #8510426)"]
)

default_values = {}
if preset_choice == "Sample A — Confirmed Malignant (Patient #842302)":
    default_values = {
        'radius_mean': 17.99, 'texture_mean': 10.38, 'perimeter_mean': 122.8, 'area_mean': 1001.0,
        'smoothness_mean': 0.1184, 'compactness_mean': 0.2776, 'concavity_mean': 0.3001,
        'concave points_mean': 0.1471, 'symmetry_mean': 0.2419, 'fractal_dimension_mean': 0.07871,
        'radius_se': 1.095, 'texture_se': 0.9053, 'perimeter_se': 8.589, 'area_se': 153.4,
        'smoothness_se': 0.006399, 'compactness_se': 0.04904, 'concavity_se': 0.05373,
        'concave points_se': 0.01587, 'symmetry_se': 0.03003, 'fractal_dimension_se': 0.006193,
        'radius_worst': 25.38, 'texture_worst': 17.33, 'perimeter_worst': 184.6, 'area_worst': 2019.0,
        'smoothness_worst': 0.1622, 'compactness_worst': 0.6656, 'concavity_worst': 0.7119,
        'concave points_worst': 0.2654, 'symmetry_worst': 0.4601, 'fractal_dimension_worst': 0.1189
    }
elif preset_choice == "Sample B — Confirmed Benign (Patient #8510426)":
    default_values = {
        'radius_mean': 13.54, 'texture_mean': 14.36, 'perimeter_mean': 87.46, 'area_mean': 566.3,
        'smoothness_mean': 0.09779, 'compactness_mean': 0.08129, 'concavity_mean': 0.06664,
        'concave points_mean': 0.04781, 'symmetry_mean': 0.1885, 'fractal_dimension_mean': 0.05766,
        'radius_se': 0.2699, 'texture_se': 0.7886, 'perimeter_se': 2.058, 'area_se': 23.56,
        'smoothness_se': 0.008462, 'compactness_se': 0.0146, 'concavity_se': 0.02387,
        'concave points_se': 0.01315, 'symmetry_se': 0.0198, 'fractal_dimension_se': 0.0023,
        'radius_worst': 15.11, 'texture_worst': 19.26, 'perimeter_worst': 99.7, 'area_worst': 711.2,
        'smoothness_worst': 0.144, 'compactness_worst': 0.1773, 'concavity_worst': 0.239,
        'concave points_worst': 0.1288, 'symmetry_worst': 0.2977, 'fractal_dimension_worst': 0.07259
    }

# Input Sliders
st.sidebar.subheader("Key Mean Features:")
input_data = {}
for feat in feature_names:
    def_val = default_values.get(feat, 14.0 if 'radius' in feat else (500.0 if 'area' in feat else 0.1))
    if 'mean' in feat:
        input_data[feat] = st.sidebar.number_input(
            f"{feat.replace('_', ' ').title()}",
            value=float(def_val),
            format="%.4f"
        )
    else:
        # Default rest from preset or baseline
        input_data[feat] = float(def_val)

# Main Grid Layout
col_pred, col_metrics = st.columns([1.1, 0.9])

with col_pred:
    st.subheader(f"🩺 Diagnostic Prediction ({selected_model_name})")
    
    # Run Inference with selected model
    result = predict(input_data, model_name_or_path=selected_model_key)
    diag = result["diagnosis"]
    prob_m = result["probability_malignant"]
    prob_b = result["probability_benign"]
    conf = result["confidence"] * 100

    if diag == "Malignant":
        st.markdown(f'<div class="malignant-badge">⚠️ MALIGNANT ({conf:.1f}% Confidence)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="benign-badge">✅ BENIGN ({conf:.1f}% Confidence)</div>', unsafe_allow_html=True)

    st.write("")
    st.write(f"**Clinical Risk Assessment:** {result['clinical_risk_assessment']}")
    
    # Probability Bars
    st.write("#### Predicted Probabilities:")
    st.progress(prob_m, text=f"Malignant Probability: {prob_m * 100:.1f}%")
    st.progress(prob_b, text=f"Benign Probability: {prob_b * 100:.1f}%")

    st.info("""
    **Clinical Guidance:**
    In oncological screening, **Recall** is prioritized to ensure near-zero False Negatives (FN). 
    A Malignant classification triggers immediate histopathology confirmation and targeted imaging.
    """)

with col_metrics:
    st.subheader(f"📊 {selected_model_name} Benchmark")
    acc = metrics.get('accuracy', 0.0)
    prec = metrics.get('precision', 0.0)
    rec = metrics.get('recall_sensitivity', 0.0)
    f1 = metrics.get('f1_score', 0.0)
    roc_auc = metrics.get('roc_auc', 0.0)

    m1, m2 = st.columns(2)
    m1.metric("Test Accuracy", f"{acc * 100:.2f}%")
    m2.metric("Malignant Precision", f"{prec * 100:.2f}%")
    
    m3, m4 = st.columns(2)
    m3.metric("Malignant Recall", f"{rec * 100:.2f}%")
    m4.metric("F1-Score", f"{f1 * 100:.2f}%")

    arch_descriptions = {
        "knn": "• Algorithm: Distance-Weighted K-Nearest Neighbors (KNN)<br>• Optimal K: 6 neighbors<br>• Preprocessing: StandardScaler (Z-score normalization)<br>• Distance Metric: Euclidean Distance",
        "svm": "• Algorithm: Support Vector Classifier (RBF Kernel)<br>• Regularization (C): 1.0<br>• Preprocessing: StandardScaler<br>• Probability Calibration: Platt Scaling",
        "random_forest": "• Algorithm: Random Forest Classifier (100 Trees)<br>• Splitting Criterion: Gini Impurity<br>• Preprocessing: Raw continuous features (scale-invariant)<br>• Random Seed: 42",
        "logistic_regression": "• Algorithm: Logistic Regression (L2 Regularization)<br>• Solver: L-BFGS (max_iter=1000)<br>• Preprocessing: StandardScaler<br>• Random Seed: 42"
    }

    st.markdown(f"""
    <div class="metric-card">
    <b>Model Architecture & Setup:</b><br>
    {arch_descriptions.get(selected_model_key, 'Standard Scikit-Learn Pipeline')}
    </div>
    """, unsafe_allow_html=True)

# Tabs for Diagnostic Visualizations & Explainable AI
st.write("---")
st.subheader("📈 Diagnostic Analytics & Explainable AI")
tab_xai, tab_comp, tab1, tab2, tab3 = st.tabs([
    "🧠 Explainable AI (SHAP)",
    "🏆 4-Model Comparison",
    "📉 Confusion Matrix",
    "⚡ Error Rate vs. K (Elbow Plot)",
    "🎯 ROC-AUC Curves"
])

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assets_dir = os.path.join(base_dir, "assets")
reports_dir = os.path.join(base_dir, "reports")

with tab_xai:
    st.write(f"### 🔍 Model Transparency: SHAP Interpretability for {selected_model_name}")
    st.caption("SHAP (SHapley Additive exPlanations) attributes predictive credit to each nuclear morphology feature based on cooperative game theory.")

    # 1. Local Explanation Section
    st.write("#### 1. Local Patient Biopsy Feature Attribution")
    
    with st.spinner(f"Computing local SHAP explanations using {selected_model_name}..."):
        try:
            explanation = explain_instance(input_data, model_name=selected_model_key, pipeline=pipeline)
            
            col_xai1, col_xai2 = st.columns([1.1, 0.9])
            
            with col_xai1:
                st.write(f"**Explainer Architecture:** `{explanation['explainer_type']}`")
                st.write(f"**Predicted Target:** `{explanation['diagnosis']}` | **{explanation['score_name']}:** `{explanation['model_score']:.4f}`")
                st.write(f"**Baseline Rate (Expected Value):** `{explanation['base_value']:.4f}`")

                # Direction of contribution cards
                st.write("**Top Factors Elevating Risk (Pushing toward Malignant):**")
                if explanation["top_positive_features"]:
                    for item in explanation["top_positive_features"]:
                        st.markdown(
                            f'<div class="shap-positive"><b>{item["feature"].replace("_", " ").title()}:</b> '
                            f'Value = {item["feature_value"]:.4f} &nbsp;|&nbsp; '
                            f'<b>+{item["shap_value"]:.4f}</b> contribution</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.write("None: No features strongly pushed toward malignancy.")

                st.write("**Top Factors Lowering Risk (Pushing toward Benign):**")
                if explanation["top_negative_features"]:
                    for item in explanation["top_negative_features"]:
                        st.markdown(
                            f'<div class="shap-negative"><b>{item["feature"].replace("_", " ").title()}:</b> '
                            f'Value = {item["feature_value"]:.4f} &nbsp;|&nbsp; '
                            f'<b>{item["shap_value"]:.4f}</b> contribution</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.write("None: No features strongly pushed toward benign.")

            with col_xai2:
                # Render Waterfall plot
                fig_waterfall = plot_local_waterfall(explanation["shap_explanation"], max_display=9)
                st.pyplot(fig_waterfall)
                plt.close(fig_waterfall)

        except Exception as err:
            st.warning(f"Could not compute local SHAP explanation: {err}")

    st.write("---")
    # 2. Global Explanations Section
    st.write("#### 2. Global Clinical Biomarker Reliance")
    st.write("Global feature attribution computed across the training cohort, highlighting which cell nucleus properties dominate classification decisions:")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        biomarker_img = os.path.join(assets_dir, "shap_feature_importance.png")
        if os.path.exists(biomarker_img):
            st.image(biomarker_img, caption="Comparative Clinical Biomarker Importance Across 4 Classifiers", use_container_width=True)
    with col_g2:
        rf_summary_img = os.path.join(assets_dir, "shap_summary_random_forest.png")
        if os.path.exists(rf_summary_img):
            st.image(rf_summary_img, caption="TreeSHAP Beeswarm Plot (Random Forest Malignant Attribution)", use_container_width=True)

    # Global importance table for active model
    shap_csv = os.path.join(reports_dir, "shap_feature_importance.csv")
    if os.path.exists(shap_csv):
        df_all_shap = pd.read_csv(shap_csv)
        df_model_shap = df_all_shap[df_all_shap["model"] == selected_model_key].head(10)
        st.write(f"**Top 10 Biomarkers for `{selected_model_name}` (Ranked by Mean Absolute SHAP):**")
        st.dataframe(df_model_shap[["importance_rank", "feature", "mean_abs_shap"]], use_container_width=True)

    st.warning("""
    **Educational & Clinical Decision Support Notice:**
    SHAP values quantify statistical feature attribution and predictive reliance within this machine learning system.
    They do **not** represent biological causality or clinical pathology findings. All diagnostic determinations must be corroborated by licensed pathologists via histologic biopsy review.
    """)

with tab_comp:
    st.write("### Benchmark Comparison across 4 Classification Algorithms")
    csv_path = os.path.join(reports_dir, "model_comparison.csv")
    if os.path.exists(csv_path):
        df_comp = pd.read_csv(csv_path)
        st.dataframe(df_comp, use_container_width=True)
    
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        comp_img = os.path.join(assets_dir, "model_comparison.png")
        if os.path.exists(comp_img):
            st.image(comp_img, caption="Multi-Metric Model Comparison (Test vs CV)", use_container_width=True)
    with col_img2:
        roc_multi_img = os.path.join(assets_dir, "multi_roc_curves.png")
        if os.path.exists(roc_multi_img):
            st.image(roc_multi_img, caption="Multi-Model ROC Curves Comparison", use_container_width=True)

with tab1:
    cm_path = os.path.join(assets_dir, "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix on Held-Out Test Set (114 samples)", use_container_width=True)
    else:
        st.write("Confusion Matrix image not found.")

with tab2:
    elbow_path = os.path.join(assets_dir, "error_rate_k.png")
    if os.path.exists(elbow_path):
        st.image(elbow_path, caption="Elbow Method Curve: Misclassification Error vs. Number of Neighbors (K)", use_container_width=True)

with tab3:
    roc_path = os.path.join(assets_dir, "roc_curve.png")
    if os.path.exists(roc_path):
        st.image(roc_path, caption="Receiver Operating Characteristic (ROC) Curve", use_container_width=True)
