"""
app.py
======
Interactive Streamlit Web Application for Breast Cancer Biopsy Prediction & Explainable AI.
Structured into four clear clinical and machine learning modules:
1. Biopsy Prediction
2. Model Comparison
3. Explainable AI (SHAP)
4. About the Project
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
from src.data_loader import load_raw_dataset, FEATURE_NAMES
from src.explain import explain_instance, plot_local_waterfall

st.set_page_config(
    page_title="Breast Cancer Classification & Explainable AI Platform",
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
        color: #311B92;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #455A64;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #F8F9FA;
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 5px solid #5E35B1;
        margin-bottom: 10px;
    }
    .benign-badge {
        background-color: #E8F5E9;
        color: #1B5E20;
        font-weight: bold;
        padding: 10px 18px;
        border-radius: 8px;
        font-size: 1.25rem;
        display: inline-block;
        border: 2px solid #81C784;
    }
    .malignant-badge {
        background-color: #FFEBEE;
        color: #B71C1C;
        font-weight: bold;
        padding: 10px 18px;
        border-radius: 8px;
        font-size: 1.25rem;
        display: inline-block;
        border: 2px solid #E57373;
    }
    .shap-positive {
        background-color: #FFF3E0;
        border-left: 4px solid #E65100;
        padding: 8px 12px;
        margin-bottom: 6px;
        border-radius: 4px;
        font-size: 0.88rem;
    }
    .shap-negative {
        background-color: #E1F5FE;
        border-left: 4px solid #0288D1;
        padding: 8px 12px;
        margin-bottom: 6px;
        border-radius: 4px;
        font-size: 0.88rem;
    }
    .disclaimer-box {
        background-color: #FFFDE7;
        border-left: 4px solid #FBC02D;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 15px;
        font-size: 0.88rem;
        color: #5D4037;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🎗️ Breast Cancer Classification & Explainable AI System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A Clinical Decision-Support & Diagnostic Benchmark Platform powered by <b>KNN, Logistic Regression, SVM, and Random Forest</b> with <b>SHAP Model Transparency</b>.</div>', unsafe_allow_html=True)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assets_dir = os.path.join(base_dir, "assets")
reports_dir = os.path.join(base_dir, "reports")

# Sidebar - Model Selection & Biopsy Controls
st.sidebar.header("🤖 Model Selection")
MODEL_NAME_TO_KEY = {
    "K-Nearest Neighbors (KNN)": "knn",
    "Support Vector Machine (SVM)": "svm",
    "Random Forest": "random_forest",
    "Logistic Regression": "logistic_regression"
}

selected_model_name = st.sidebar.selectbox(
    "Active Classifier:",
    options=list(MODEL_NAME_TO_KEY.keys()),
    index=2,  # Default to Random Forest
    help="Select which machine learning architecture to evaluate and explain."
)
selected_model_key = MODEL_NAME_TO_KEY[selected_model_name]

# Load Selected Model
@st.cache_resource
def get_model(model_key: str):
    return load_trained_pipeline(model_key)

try:
    artifact = get_model(selected_model_key)
    pipeline = artifact["pipeline"]
    feature_names = artifact["feature_names"]
    metrics = artifact.get("metrics", {})
except Exception as e:
    st.error(f"Error loading model '{selected_model_name}': {e}. Please ensure 'python -m src.train' has been run.")
    st.stop()

# Biopsy Sample Presets
st.sidebar.header("🔬 Patient Biopsy Features")
preset_choice = st.sidebar.selectbox(
    "Load Clinical Preset Sample:",
    [
        "Custom Patient Input",
        "Sample A — Confirmed Malignant (Patient #842302)",
        "Sample B — Confirmed Benign (Patient #8510426)"
    ],
    help="Preload confirmed clinical cases from the Wisconsin dataset or enter custom values."
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

# Organized Sidebar Feature Inputs
input_data = {}
with st.sidebar.expander("🔹 Mean Nuclear Measurements", expanded=True):
    for feat in [f for f in feature_names if "_mean" in f]:
        def_val = default_values.get(feat, 14.0 if 'radius' in feat else (500.0 if 'area' in feat else 0.1))
        input_data[feat] = st.number_input(
            f"{feat.replace('_mean', '').replace('_', ' ').title()}",
            value=float(def_val),
            format="%.4f",
            key=f"input_{feat}"
        )

with st.sidebar.expander("🔸 Standard Error (SE) Measurements", expanded=False):
    for feat in [f for f in feature_names if "_se" in f]:
        def_val = default_values.get(feat, 1.0 if 'radius' in feat else (40.0 if 'area' in feat else 0.01))
        input_data[feat] = st.number_input(
            f"{feat.replace('_se', '').replace('_', ' ').title()} SE",
            value=float(def_val),
            format="%.4f",
            key=f"input_{feat}"
        )

with st.sidebar.expander("🔺 Worst / Largest Measurements", expanded=False):
    for feat in [f for f in feature_names if "_worst" in f]:
        def_val = default_values.get(feat, 16.0 if 'radius' in feat else (800.0 if 'area' in feat else 0.15))
        input_data[feat] = st.number_input(
            f"{feat.replace('_worst', '').replace('_', ' ').title()} Worst",
            value=float(def_val),
            format="%.4f",
            key=f"input_{feat}"
        )

# Four Clear Sections via Tabs
tab_pred, tab_comp, tab_xai, tab_about = st.tabs([
    "🔮 1. Prediction",
    "📊 2. Model Comparison",
    "🧠 3. Explainable AI",
    "ℹ️ 4. About the Project"
])

# ==============================================================================
# SECTION 1: PREDICTION
# ==============================================================================
with tab_pred:
    st.subheader(f"Diagnostic Assessment using {selected_model_name}")
    col_pred1, col_pred2 = st.columns([1.1, 0.9])

    with col_pred1:
        st.markdown("#### Patient Biopsy Classification")
        try:
            pred_result = predict(input_data, model_name_or_path=selected_model_key)
            diag = pred_result["diagnosis"]
            prob_m = pred_result["probability_malignant"]
            prob_b = pred_result["probability_benign"]
            conf = pred_result["confidence"] * 100

            if diag == "Malignant":
                st.markdown(f'<div class="malignant-badge">⚠️ MALIGNANT INDICATION ({conf:.1f}% Confidence)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="benign-badge">✅ BENIGN INDICATION ({conf:.1f}% Confidence)</div>', unsafe_allow_html=True)

            st.write("")
            st.write(f"**Clinical Risk Stratification:** {pred_result['clinical_risk_assessment']}")

            # Probability Bars
            st.write("##### Predicted Probability Distribution:")
            st.progress(prob_m, text=f"Malignancy Probability: {prob_m * 100:.2f}%")
            st.progress(prob_b, text=f"Benign Probability: {prob_b * 100:.2f}%")

            st.info("""
            **Oncology Screening Protocol:**
            In fine needle aspirate (FNA) evaluations, sensitivity (Recall) is prioritized to ensure malignant lesions are not missed (minimizing False Negatives). A Malignant prediction warrants immediate core needle biopsy and targeted radiographic correlation.
            """)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

    with col_pred2:
        st.markdown(f"#### Active Classifier Performance ({selected_model_name})")
        m_acc = metrics.get('accuracy', 0.0)
        m_prec = metrics.get('precision', 0.0)
        m_rec = metrics.get('recall_sensitivity', 0.0)
        m_f1 = metrics.get('f1_score', 0.0)
        m_auc = metrics.get('roc_auc', 0.0)
        m_pr_auc = metrics.get('pr_auc', 0.0)

        c1, c2 = st.columns(2)
        c1.metric("Test Accuracy", f"{m_acc * 100:.2f}%")
        c2.metric("Malignant Precision", f"{m_prec * 100:.2f}%")

        c3, c4 = st.columns(2)
        c3.metric("Malignant Recall (Sens.)", f"{m_rec * 100:.2f}%")
        c4.metric("Malignant F1-Score", f"{m_f1 * 100:.2f}%")

        c5, c6 = st.columns(2)
        c5.metric("ROC-AUC Score", f"{m_auc:.4f}" if m_auc else "N/A")
        c6.metric("PR-AUC Score", f"{m_pr_auc:.4f}" if m_pr_auc else "N/A")

        arch_descriptions = {
            "knn": "• Model: Distance-Weighted K-Nearest Neighbors (K=6)<br>• Scaling: StandardScaler (Z-score normalization)<br>• Distance Metric: Euclidean (L2 Norm)",
            "svm": "• Model: Support Vector Classifier (RBF Kernel, C=1.0)<br>• Scaling: StandardScaler<br>• Probability Calibration: Platt Scaling",
            "random_forest": "• Model: Random Forest (100 Trees, Gini Impurity)<br>• Scaling: Scale-Invariant (Raw Features)<br>• Seed: 42",
            "logistic_regression": "• Model: Logistic Regression (L2 Penalty, L-BFGS)<br>• Scaling: StandardScaler<br>• Max Iterations: 1000"
        }
        st.markdown(f"""
        <div class="metric-card">
        <b>Pipeline Configuration:</b><br>
        {arch_descriptions.get(selected_model_key, 'Scikit-Learn Pipeline')}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
    <b>⚠️ Important Educational & Research Notice:</b>
    This system is intended strictly for academic benchmarking and algorithmic interpretability research. It is NOT an FDA-approved medical diagnostic device and does NOT provide definitive medical diagnoses. Clinical treatment decisions must always be directed by qualified medical pathologists.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: MODEL COMPARISON
# ==============================================================================
with tab_comp:
    st.subheader("Cross-Architecture Benchmark & Clinical Evaluation")
    st.markdown("All 4 models were evaluated under identical conditions: a stratified 80/20 train/test partition (`random_state=42`) with 5-fold stratified cross-validation on the training set.")

    # Load Comparison Report
    csv_comp_path = os.path.join(reports_dir, "model_comparison.csv")
    if os.path.exists(csv_comp_path):
        df_comp = pd.read_csv(csv_comp_path)
        display_df = df_comp.copy()
        
        # Format percentage columns
        pct_cols = [
            "CV_Accuracy_Mean", "CV_Recall_Mean", "CV_F1_Mean", "CV_ROC_AUC_Mean",
            "Test_Accuracy", "Test_Malignant_Precision", "Test_Malignant_Recall",
            "Test_Malignant_F1", "Test_ROC_AUC", "Test_PR_AUC", "Test_Malignant_FNR"
        ]
        for col in pct_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda v: f"{v*100:.2f}%" if pd.notnull(v) else "N/A")

        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("Comparison report CSV not found. Please run 'python -m src.train' to regenerate.")

    st.write("---")
    st.markdown("#### Diagnostic Evaluation Visualizations")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        comp_img = os.path.join(assets_dir, "model_comparison.png")
        if os.path.exists(comp_img):
            st.image(comp_img, caption="Four-Model Performance Comparison (Held-Out Test Set)", use_container_width=True)

        pr_img = os.path.join(assets_dir, "multi_pr_curves.png")
        if os.path.exists(pr_img):
            st.image(pr_img, caption="Multi-Model Precision-Recall (PR) Curves (Malignant Class)", use_container_width=True)

    with col_chart2:
        roc_img = os.path.join(assets_dir, "multi_roc_curves.png")
        if os.path.exists(roc_img):
            st.image(roc_img, caption="Multi-Model Receiver Operating Characteristic (ROC) Curves", use_container_width=True)

        cm_img = os.path.join(assets_dir, "confusion_matrix.png")
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Test Set Confusion Matrix Heatmap (114 samples)", use_container_width=True)

    st.markdown("#### Clinical Error & False-Negative Analysis")
    st.markdown("""
    - **False Negatives (FN):** In oncological screening, False Negatives represent patients with undiagnosed malignancies. Both **Support Vector Machine (SVM)** and **Random Forest** achieved the lowest False Negative count (**FN = 3**, FNR = 7.14%) alongside **100% Precision** (Zero False Positives).
    - **False Positives (FP):** False alarms cause patient anxiety and unnecessary invasive procedures. SVM and Random Forest generated **0 False Positives**, while Logistic Regression and KNN generated 1 False Positive each.
    - **Precision-Recall (PR-AUC):** Because malignant prevalence is 37%, PR curves provide critical validation of high positive predictive value across all sensitivity thresholds.
    """)

# ==============================================================================
# SECTION 3: EXPLAINABLE AI (SHAP)
# ==============================================================================
with tab_xai:
    st.subheader(f"Model Transparency & Interpretability ({selected_model_name})")
    st.caption("SHAP (SHapley Additive exPlanations) mathematically allocates credit to each cellular morphology measurement using cooperative game theory.")

    # Clear distinction of units
    if selected_model_key == "logistic_regression":
        st.info("""
        **ℹ️ Explanation Unit: Log-Odds Margin (Logit)**
        For Logistic Regression, SHAP values explain the change in the linear decision function ($\Delta \text{logit}$).
        Positive values push log-odds toward Malignant ($> 0$); negative values push toward Benign ($< 0$).
        """)
    else:
        st.info("""
        **ℹ️ Explanation Unit: Probability of Malignancy (0.0 to 1.0)**
        For Random Forest, SVM, and KNN, SHAP values explain the direct percentage-point shift in predicted risk of malignancy ($\Delta P(\text{Malignant})$) away from the population baseline.
        """)

    st.markdown("#### 1. Local Patient Biopsy Feature Attribution")

    with st.spinner(f"Computing local SHAP explanations using {selected_model_name}..."):
        try:
            explanation = explain_instance(input_data, model_name=selected_model_key, pipeline=pipeline)
            
            col_x1, col_x2 = st.columns([1.05, 0.95])

            with col_x1:
                st.write(f"**Explainer Architecture:** `{explanation['explainer_type']}`")
                st.write(f"**Predicted Diagnosis:** `{explanation['diagnosis']}` | **Score:** `{explanation['model_score']:.4f}` ({explanation['unit_label']})")
                st.write(f"**Population Baseline Rate ($E[f(X)]$):** `{explanation['base_value']:.4f}`")

                st.write("**Top Factors Elevating Risk (Pushing toward Malignant):**")
                if explanation["top_positive_features"]:
                    for item in explanation["top_positive_features"]:
                        st.markdown(
                            f'<div class="shap-positive"><b>{item["feature"].replace("_", " ").title()}:</b> '
                            f'Measured = {item["feature_value"]:.4f} &nbsp;|&nbsp; '
                            f'<b>+{item["shap_value"]:.4f}</b> contribution</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.write("None: No features contributed positively toward malignancy.")

                st.write("**Top Factors Lowering Risk (Pushing toward Benign):**")
                if explanation["top_negative_features"]:
                    for item in explanation["top_negative_features"]:
                        st.markdown(
                            f'<div class="shap-negative"><b>{item["feature"].replace("_", " ").title()}:</b> '
                            f'Measured = {item["feature_value"]:.4f} &nbsp;|&nbsp; '
                            f'<b>{item["shap_value"]:.4f}</b> contribution</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.write("None: No features contributed toward benign.")

            with col_x2:
                # Dynamic Waterfall plot
                fig_waterfall = plot_local_waterfall(
                    explanation["shap_explanation"],
                    max_display=9,
                    unit_label=explanation.get("unit_label")
                )
                st.pyplot(fig_waterfall)
                plt.close(fig_waterfall)

        except Exception as e:
            st.error(f"Error computing local SHAP explanation: {e}")

    st.write("---")
    st.markdown("#### 2. Global Clinical Biomarker Reliance")
    st.write("Aggregated feature attribution computed across the training cohort, highlighting which morphological dimensions dominate classification:")

    col_gx1, col_gx2 = st.columns(2)
    with col_gx1:
        biomarker_img = os.path.join(assets_dir, "shap_feature_importance.png")
        if os.path.exists(biomarker_img):
            st.image(biomarker_img, caption="Comparative Clinical Biomarker Importance Across 4 Classifiers", use_container_width=True)

    with col_gx2:
        rf_summary_img = os.path.join(assets_dir, "shap_summary_random_forest.png")
        if os.path.exists(rf_summary_img):
            st.image(rf_summary_img, caption="TreeSHAP Beeswarm Summary Plot (Random Forest Malignant Attribution)", use_container_width=True)

    # Active Model Ranked Importance Table
    shap_csv = os.path.join(reports_dir, "shap_feature_importance.csv")
    if os.path.exists(shap_csv):
        df_all_shap = pd.read_csv(shap_csv)
        df_model_shap = df_all_shap[df_all_shap["model"] == selected_model_key].head(10).copy()
        df_model_shap["feature"] = df_model_shap["feature"].apply(lambda f: f.replace("_", " ").title())
        st.write(f"**Top 10 Biomarkers for `{selected_model_name}` (Ranked by Mean Absolute SHAP):**")
        st.dataframe(df_model_shap[["importance_rank", "feature", "mean_abs_shap"]], use_container_width=True)

    st.warning("""
    **Methodological Attribution Notice:**
    SHAP values quantify statistical feature attribution and predictive reliance within the trained machine learning pipeline. They do **not** establish biological causality or physical etiology.
    """)

# ==============================================================================
# SECTION 4: ABOUT THE PROJECT
# ==============================================================================
with tab_about:
    st.subheader("About the Breast Cancer Classification & Explainable AI Project")

    col_ab1, col_ab2 = st.columns([1.1, 0.9])

    with col_ab1:
        st.markdown("""
        #### 📖 Project Overview
        This project provides an end-to-end, scientifically rigorous Machine Learning and Explainable AI (XAI) pipeline for breast cancer diagnosis and malignancy detection. Built on the **Wisconsin Breast Cancer Diagnostic Dataset (WDBC)**, the system evaluates and benchmarks four core classification algorithms:
        
        1. **K-Nearest Neighbors (KNN):** Distance-weighted local neighborhood classifier ($w = 1/d$).
        2. **Logistic Regression:** Linear probabilistic classifier with L2 regularization.
        3. **Support Vector Machine (SVM):** Maximum-margin classifier with Radial Basis Function (RBF) kernel and Platt probability calibration.
        4. **Random Forest:** Non-parametric ensemble of 100 de-correlated decision trees.

        #### 🔬 Clinical Dataset & Cytological Features
        Features were computed from digitized images of Fine Needle Aspirate (FNA) biopsies of breast masses, describing characteristics of the cell nuclei present in the image:
        - **Radius:** Mean of distances from center to points on the perimeter.
        - **Texture:** Standard deviation of gray-scale values.
        - **Perimeter & Area:** Nuclear boundary dimensions and surface coverage.
        - **Smoothness:** Local variation in radius lengths.
        - **Compactness:** Defined as $\\frac{\\text{perimeter}^2}{\\text{area}} - 1.0$.
        - **Concavity:** Severity of concave portions of the nuclear contour.
        - **Concave Points:** Number of concave portions of the contour.
        - **Symmetry & Fractal Dimension:** Nuclear symmetry and "coastline approximation" ($FD = \\text{dim} - 1$).
        
        Each characteristic is measured as **Mean**, **Standard Error (SE)**, and **Worst (largest)** across cells, yielding 30 continuous clinical features.
        """)

    with col_ab2:
        st.markdown("""
        #### ⚙️ Engineering Principles
        - **Zero Data Leakage:** Preprocessing (`StandardScaler`) is encapsulated strictly inside scikit-learn pipelines and fit solely on training folds.
        - **Stratified Partitioning:** Strict 80/20 train/test splitting (`random_state=42`) preserves real-world class prevalence.
        - **Reproducibility:** Seeded deterministic pipelines ensure consistent benchmark metrics across environments.
        - **Axiomatic Interpretability:** SHAP TreeExplainer, LinearExplainer, and KernelExplainer implementations guarantee theoretical consistency.

        #### 👤 Author & Repository
        - **Author:** Sundari Krishna (IIT Patna, B.Tech EEE)
        - **GitHub Repository:** [krishnasundari/breast--cancer-prediction](https://github.com/krishnasundari/breast--cancer-prediction)
        - **License:** Open Source under the [MIT License](https://opensource.org/licenses/MIT)
        """)

    st.markdown("""
    <div class="disclaimer-box">
    <b>⚕️ Medical & Educational Disclaimer:</b><br>
    This software is created solely for educational, research, and algorithmic benchmarking purposes. It is not an FDA-cleared diagnostic medical device and must never replace the clinical judgment of a licensed medical pathologist or oncologist.
    </div>
    """, unsafe_allow_html=True)
