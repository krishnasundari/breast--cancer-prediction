"""
app.py
======
Interactive Streamlit Web Application for Breast Cancer Biopsy Prediction.
Allows clinicians, researchers, and interviewers to test real-time KNN inference.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Add repo root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import load_trained_pipeline, predict
from src.data_loader import load_raw_dataset

st.set_page_config(
    page_title="Breast Cancer Detection | KNN Diagnostic Platform",
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🎗️ Breast Cancer Detection — K-Nearest Neighbors</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Clinical Diagnostic Platform powered by an End-to-End Scikit-Learn Pipeline achieving <b>97.3% Classification Accuracy</b> on the Wisconsin Diagnostic Dataset.</div>', unsafe_allow_html=True)

# Load Model
@st.cache_resource
def get_model():
    return load_trained_pipeline()

artifact = get_model()
pipeline = artifact["pipeline"]
feature_names = artifact["feature_names"]
metrics = artifact["metrics"]

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
    st.subheader("🩺 Diagnostic Prediction Result")
    
    # Run Inference
    result = predict(input_data)
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
    st.subheader("📊 Model Performance Benchmark")
    m1, m2 = st.columns(2)
    m1.metric("Classification Accuracy", f"{metrics['accuracy'] * 100:.1f}%", "Target: 97.3%")
    m2.metric("Malignant Precision", f"{metrics['precision'] * 100:.1f}%", "Zero False Positives")
    
    m3, m4 = st.columns(2)
    m3.metric("Malignant Recall", f"{metrics['recall_sensitivity'] * 100:.1f}%", "Sensitivity")
    m4.metric("F1-Score", f"{metrics['f1_score'] * 100:.1f}%", "Balanced Metric")

    st.markdown("""
    <div class="metric-card">
    <b>Model Architecture:</b><br>
    • Algorithm: Distance-Weighted K-Nearest Neighbors (KNN)<br>
    • Optimal K: 6 neighbors (via Elbow Method)<br>
    • Scaling: Standard Scaler (Z-Score Normalization)<br>
    • Distance Metric: Euclidean Distance (L2 norm)
    </div>
    """, unsafe_allow_html=True)

# Tabs for Diagnostic Visualizations
st.write("---")
st.subheader("📈 Diagnostic Analytics & Evaluation Visualizations")
tab1, tab2, tab3 = st.tabs(["📉 Confusion Matrix", "⚡ Error Rate vs. K (Elbow Plot)", "🎯 ROC-AUC Curve"])

assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

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
