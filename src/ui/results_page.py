"""
results_page.py
===============
Page 3: Prediction and Explainable AI Results.
Divided into:
- Section A: Multi-Model Prediction Summary (4 models, probabilities, model agreement)
- Section B: Model Benchmark Comparison (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
- Section C: Explain Why the Model Predicted This Result (Local SHAP waterfall, directional factors, global reliance)
- Section D: Clinical Next Actions & Navigation
"""

import os
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.predict import predict, load_trained_pipeline
from src.explain import explain_instance, plot_local_waterfall
from src.data_loader import FEATURE_NAMES
from src.ui.nav import PAGE_INTAKE, PAGE_ABOUT, navigate_to


def render_results_page():
    st.markdown('<div class="hero-title">🔬 Diagnostic Assessment & Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Step 2 of 2: Multi-Model Consensus & SHAP Attribution Analysis</div>', unsafe_allow_html=True)

    # Validate that features exist in session state
    patient_features = st.session_state.get("patient_features")
    if not patient_features or len(patient_features) != 30:
        st.error("No valid patient feature signature detected. Please return to the Patient Information page to enter measurements.")
        if st.button("⬅️ Go to Patient Intake"):
            navigate_to(PAGE_INTAKE)
        return

    patient_meta = st.session_state.get("patient_meta", {})
    patient_symptoms = st.session_state.get("patient_symptoms", {})

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assets_dir = os.path.join(base_dir, "assets")
    reports_dir = os.path.join(base_dir, "reports")

    # Patient Case Header Pill
    st.markdown(f"""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:12px 18px; margin-bottom:18px; display:flex; justify-content:space-between; align-items:center;">
        <div>
            <b>Patient ID:</b> <code>{patient_meta.get('patient_id', 'N/A')}</code> &nbsp;|&nbsp;
            <b>Age:</b> {patient_meta.get('age', 'N/A')} yrs &nbsp;|&nbsp;
            <b>Indication:</b> {patient_meta.get('history', 'Routine screening')}
        </div>
        <div style="font-size:0.85rem; color:#5A6E85;">
            Cytology: 30 Quantitative Wisconsin Metrics Verified
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Run predictions across all 4 models
    models_list = ["knn", "logistic_regression", "svm", "random_forest"]
    model_display_names = {
        "knn": "K-Nearest Neighbors (KNN)",
        "logistic_regression": "Logistic Regression",
        "svm": "Support Vector Machine (SVM)",
        "random_forest": "Random Forest"
    }

    results = {}
    with st.spinner("Executing inference across all 4 trained machine learning pipelines..."):
        for m in models_list:
            try:
                results[m] = predict(patient_features, model_name_or_path=m)
            except Exception as e:
                st.error(f"Error evaluating {m}: {e}")

    # =========================================================================
    # SECTION A: Prediction Summary Card
    # =========================================================================
    st.markdown('<div class="section-title">🩺 Section A: Prediction Summary & Model Consensus</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Independent diagnostic inference from all four classifiers:</div>', unsafe_allow_html=True)

    # Tally Consensus
    mal_count = sum(1 for res in results.values() if res["diagnosis"] == "Malignant")
    ben_count = len(results) - mal_count

    col_sum_badge, col_sum_details = st.columns([1.1, 0.9])

    with col_sum_badge:
        if mal_count >= 3:
            st.markdown(f'<div class="med-badge-malignant">⚠️ MALIGNANT INDICATION ({mal_count}/4 Models Agree)</div>', unsafe_allow_html=True)
            consensus_msg = "Elevated Malignancy Risk: Significant concordance among classifiers. Immediate histological verification advised."
        elif ben_count >= 3:
            st.markdown(f'<div class="med-badge-benign">✅ BENIGN INDICATION ({ben_count}/4 Models Agree)</div>', unsafe_allow_html=True)
            consensus_msg = "Consistent Benign Profile: Majority of classifiers classify nuclear metrics as non-malignant."
        else:
            st.markdown(f'<div style="background:#FFF3E0; color:#E65100; font-weight:800; padding:12px 24px; border-radius:10px; font-size:1.4rem; display:inline-block; border:2px solid #FFCC80;">⚖️ BORDERLINE / SPLIT CONSENSUS (2 vs 2)</div>', unsafe_allow_html=True)
            consensus_msg = "Equivocal Nuclear Morphometry: Split decision between algorithms. Cytopathologist expert review required."

        st.write("")
        st.write(f"**Clinical Interpretation Note:** {consensus_msg}")

    with col_sum_details:
        st.markdown("""
        <div class="med-card">
            <h5 style="margin-top:0; color:#0D1B2A;">Model Agreement Summary:</h5>
            <div style="font-size:0.9rem; line-height:1.6;">
        """, unsafe_allow_html=True)
        for m in models_list:
            res = results.get(m, {})
            diag = res.get("diagnosis", "N/A")
            color = "#C62828" if diag == "Malignant" else "#2E7D32"
            prob_m = res.get("probability_malignant", 0.0)
            st.markdown(f"• <b>{model_display_names[m]}:</b> <span style='color:{color}; font-weight:700;'>{diag}</span> (P(Malignant) = {prob_m*100:.1f}%)", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Individual Model Scorecards
    st.markdown("##### Individual Classifier Probability Distribution:")
    card_cols = st.columns(4)
    for idx, m in enumerate(models_list):
        res = results.get(m, {})
        with card_cols[idx]:
            diag = res.get("diagnosis", "N/A")
            prob_m = res.get("probability_malignant", 0.0)
            prob_b = res.get("probability_benign", 0.0)
            risk = res.get("clinical_risk_assessment", "")

            badge_color = "#C62828" if diag == "Malignant" else "#2E7D32"
            st.markdown(f"""
            <div class="med-card" style="padding:14px; text-align:center;">
                <div style="font-weight:700; font-size:0.95rem; color:#0D1B2A; margin-bottom:4px;">{model_display_names[m].split('(')[0]}</div>
                <div style="font-size:1.2rem; font-weight:800; color:{badge_color}; margin-bottom:8px;">{diag}</div>
                <div style="font-size:0.8rem; color:#5A6E85;">P(Malignant): <b>{prob_m*100:.1f}%</b></div>
                <div style="font-size:0.8rem; color:#5A6E85;">P(Benign): <b>{prob_b*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

    st.write("---")

    # =========================================================================
    # SECTION B: Model Comparison Table & Charts
    # =========================================================================
    st.markdown('<div class="section-title">📊 Section B: Model Benchmark Comparison & Clinical Trade-offs</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Verified performance metrics on the held-out Wisconsin test set (114 samples) and 5-fold cross-validation:</div>', unsafe_allow_html=True)

    csv_comp_path = os.path.join(reports_dir, "model_comparison.csv")
    if os.path.exists(csv_comp_path):
        df_comp = pd.read_csv(csv_comp_path)
        display_df = df_comp.copy()
        for col in ["CV_Accuracy_Mean", "CV_Recall_Mean", "Test_Accuracy", "Test_Malignant_Precision", "Test_Malignant_Recall", "Test_Malignant_F1", "Test_ROC_AUC", "Test_PR_AUC", "Test_Malignant_FNR"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda v: f"{v*100:.2f}%" if pd.notnull(v) else "N/A")
        st.dataframe(display_df, use_container_width=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        comp_img = os.path.join(assets_dir, "model_comparison.png")
        if os.path.exists(comp_img):
            st.image(comp_img, caption="Four-Model Performance Comparison (Test vs CV)", use_container_width=True)
        pr_img = os.path.join(assets_dir, "multi_pr_curves.png")
        if os.path.exists(pr_img):
            st.image(pr_img, caption="Multi-Model Precision-Recall Curves (Positive Predictive Power)", use_container_width=True)

    with col_c2:
        roc_img = os.path.join(assets_dir, "multi_roc_curves.png")
        if os.path.exists(roc_img):
            st.image(roc_img, caption="Multi-Model Receiver Operating Characteristic (ROC)", use_container_width=True)
        cm_img = os.path.join(assets_dir, "confusion_matrix.png")
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Test Confusion Matrix (114 samples)", use_container_width=True)

    st.write("---")

    # =========================================================================
    # SECTION C: Explain Why the Model Predicted This Result (SHAP)
    # =========================================================================
    st.markdown('<div class="section-title">🧠 Section C: Explain Why the Model Predicted This Result (SHAP)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Shapley Additive exPlanations (SHAP) break down the prediction into simple factors pushing toward Malignant or Benign:</div>', unsafe_allow_html=True)

    # Friendly Layman / Interview Explainer Card
    with st.expander("💡 Plain-English Guide: How to Explain this in Interviews & to Doctors", expanded=True):
        st.markdown("""
        **Think of SHAP as a Tug-of-War game:**
        * 🎯 **Baseline Rate ($E[f(X)] = 37.3\%$):** The neutral starting point — the average rate of breast malignancy in the historical dataset.
        * 🔴 **Red Bars (+ Risk):** Cellular abnormalities (e.g. abnormally large nuclei or irregular notched edges) that **push the risk UP towards Malignant**.
        * 🔵 **Blue Bars (− Risk):** Normal healthy characteristics (e.g. smooth cell texture) that **pull the risk DOWN towards Benign**.
        * 🏁 **Final Score ($f(x) = 95.0\%$):** Baseline + (All Red Pulls) − (All Blue Pulls) = Patient's Final Risk Score.
        """)

    # Explainer Selector
    selected_xai_model = st.selectbox(
        "Select Model to Inspect Explainability:",
        options=models_list,
        format_func=lambda x: model_display_names[x],
        index=3  # Default to Random Forest
    )

    # Friendly human-readable feature labels dictionary
    FEATURE_HUMAN_NAMES = {
        "area_worst": "Largest Nuclear Size (Area Worst)",
        "concave points_worst": "Severe Edge Indentations (Concave Points Worst)",
        "radius_worst": "Largest Nuclear Radius",
        "perimeter_worst": "Outer Boundary Length (Perimeter Worst)",
        "concave points_mean": "Average Border Indentations",
        "concavity_worst": "Severity of Cell Border Hollows",
        "texture_mean": "Cell Surface Texture (Chromatin Smoothness)",
        "texture_worst": "Worst Texture Roughness",
        "area_mean": "Average Nuclear Area",
        "perimeter_mean": "Average Cell Perimeter",
        "radius_mean": "Average Nuclear Radius",
        "smoothness_worst": "Nuclear Smoothness Variation",
        "compactness_worst": "Cell Compactness / Density",
        "symmetry_worst": "Nuclear Asymmetry"
    }

    # Differentiate units clearly using clean styled markdown banners
    if selected_xai_model == "logistic_regression":
        st.markdown("""
        <div style="background:#E1F5FE; border-left:4px solid #0288D1; padding:12px 16px; border-radius:6px; color:#01579B; margin-bottom:14px; font-size:0.9rem; line-height:1.5;">
            <b>ℹ️ Explanation Unit: Log-Odds Margin (Logit)</b><br>
            For Logistic Regression, LinearSHAP explains the shift in the linear margin (ΔLogit).
            Positive values push log-odds toward Malignant; negative values push toward Benign.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#E1F5FE; border-left:4px solid #0288D1; padding:12px 16px; border-radius:6px; color:#01579B; margin-bottom:14px; font-size:0.9rem; line-height:1.5;">
            <b>ℹ️ Explanation Unit: Probability of Malignancy (0.0 to 1.0)</b><br>
            For Random Forest (TreeSHAP), SVM, and KNN (KernelSHAP), SHAP values directly represent the change in predicted risk (ΔP(Malignant)) away from the baseline cohort rate.
        </div>
        """, unsafe_allow_html=True)

    with st.spinner(f"Computing local SHAP waterfall attribution for {model_display_names[selected_xai_model]}..."):
        try:
            explanation = explain_instance(patient_features, model_name=selected_xai_model)
            
            x_col1, x_col2 = st.columns([1.05, 0.95])

            with x_col1:
                st.write(f"**Explainer Technique:** `{explanation['explainer_type']}`")
                st.write(f"**Target Score:** `{explanation['model_score']:.4f}` ({explanation['unit_label']})")
                st.write(f"**Baseline Cohort Rate ($E[f(X)]$):** `{explanation['base_value']:.4f}`")

                st.write("**Top Red Flags (Pushing Risk Towards Malignant):**")
                if explanation["top_positive_features"]:
                    for item in explanation["top_positive_features"]:
                        f_raw = item["feature"]
                        f_label = FEATURE_HUMAN_NAMES.get(f_raw, f_raw.replace("_", " ").title())
                        st.markdown(
                            f'<div class="shap-factor-pos"><b>{f_label}:</b> '
                            f'Measured = <code>{item["feature_value"]:.4f}</code> &nbsp;|&nbsp; '
                            f'<b>+{item["shap_value"]:.4f}</b> contribution</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.write("None: No features contributed positively toward malignancy.")

                st.write("**Top Healthy Indicators (Pulling Risk Towards Benign):**")
                if explanation["top_negative_features"]:
                    for item in explanation["top_negative_features"]:
                        f_raw = item["feature"]
                        f_label = FEATURE_HUMAN_NAMES.get(f_raw, f_raw.replace("_", " ").title())
                        st.markdown(
                            f'<div class="shap-factor-neg"><b>{f_label}:</b> '
                            f'Measured = <code>{item["feature_value"]:.4f}</code> &nbsp;|&nbsp; '
                            f'<b>{item["shap_value"]:.4f}</b> contribution</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.write("None: No features contributed toward benign.")

            with x_col2:
                fig_waterfall = plot_local_waterfall(
                    explanation["shap_explanation"],
                    max_display=9,
                    unit_label=explanation.get("unit_label")
                )
                st.pyplot(fig_waterfall)
                plt.close(fig_waterfall)

        except Exception as e:
            st.error(f"Error computing local SHAP explanation: {e}")

    # Global Biomarker Reliance
    st.markdown("##### Global Biomarker Reliance Across the Wisconsin Cohort:")
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        biomarker_img = os.path.join(assets_dir, "shap_feature_importance.png")
        if os.path.exists(biomarker_img):
            st.image(biomarker_img, caption="Comparative Clinical Biomarker Importance Across 4 Classifiers", use_container_width=True)
    with g_col2:
        rf_summary_img = os.path.join(assets_dir, "shap_summary_random_forest.png")
        if os.path.exists(rf_summary_img):
            st.image(rf_summary_img, caption="TreeSHAP Beeswarm Summary Plot (Random Forest Malignant Attribution)", use_container_width=True)

    st.markdown("""
    <div class="medical-disclaimer">
        <b>⚠️ Methodological Interpretability Boundary:</b><br>
        SHAP values reflect mathematical feature attribution and predictive reliance within this statistical learning pipeline. 
        <b>They do NOT denote biological causation or physical disease etiology.</b> A high attribution indicates the model relied heavily on that nuclear geometric property to reach its classification score.
    </div>
    """, unsafe_allow_html=True)

    st.write("---")

    # =========================================================================
    # SECTION D: Next Actions & Navigation
    # =========================================================================
    st.markdown('<div class="section-title">🔄 Section D: Next Actions</div>', unsafe_allow_html=True)

    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        if st.button("⬅️ Edit Patient Measurements", use_container_width=True):
            navigate_to(PAGE_INTAKE)

    with d_col2:
        if st.button("🆕 Start New Assessment", use_container_width=True):
            st.session_state["patient_features"] = None
            navigate_to(PAGE_INTAKE)

    with d_col3:
        if st.button("📖 Read Project Documentation", use_container_width=True):
            navigate_to(PAGE_ABOUT)
