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
    # SECTION C: Explain Why the Model Predicted This Result (Simple Points & Cards)
    # =========================================================================
    st.markdown('<div class="section-title">🧠 Section C: Why Did the Model Predict This? (AI Explanation in Simple Points)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Here is the exact breakdown of which biopsy measurements pushed the risk UP toward Malignant and which pulled it DOWN toward Benign:</div>', unsafe_allow_html=True)

    # Explainer Selector
    selected_xai_model = st.selectbox(
        "Select Model to Explain:",
        options=models_list,
        format_func=lambda x: model_display_names[x],
        index=3  # Default to Random Forest
    )

    # Friendly human-readable feature labels dictionary
    FEATURE_HUMAN_NAMES = {
        "area_worst": "Largest Cell Nucleus Size (Area Worst)",
        "concave points_worst": "Severe Edge Indentations (Concave Points Worst)",
        "radius_worst": "Largest Cell Radius",
        "perimeter_worst": "Outer Boundary Length (Perimeter Worst)",
        "concave points_mean": "Average Cell Border Indentations",
        "concavity_worst": "Severity of Cell Contour Hollows",
        "texture_mean": "Cell Surface Texture (Smoothness)",
        "texture_worst": "Cell Surface Roughness",
        "area_mean": "Average Cell Nucleus Area",
        "perimeter_mean": "Average Cell Perimeter",
        "radius_mean": "Average Cell Radius",
        "smoothness_worst": "Nuclear Smoothness Variation",
        "compactness_worst": "Cell Density & Compactness",
        "symmetry_worst": "Cell Nucleus Asymmetry"
    }

    with st.spinner(f"Analyzing key decision factors for {model_display_names[selected_xai_model]}..."):
        try:
            explanation = explain_instance(patient_features, model_name=selected_xai_model)
            
            # Simple 3-Column Summary Cards
            sum_col1, sum_col2, sum_col3 = st.columns([1, 1.4, 1.4])
            
            with sum_col1:
                st.markdown(f"""
                <div style="background:#F8F9FA; border:1px solid #DEE2E6; border-radius:8px; padding:16px; height:100%;">
                    <div style="font-size:0.85rem; color:#6C757D; font-weight:600; text-transform:uppercase;">1. Starting Point</div>
                    <div style="font-size:1.6rem; font-weight:700; color:#0D1B2A; margin:6px 0;">{explanation['base_value'] * 100:.1f}%</div>
                    <div style="font-size:0.85rem; color:#495057; line-height:1.4;">
                        <b>Baseline Population Rate:</b> The historical average risk for breast biopsy patients before inspecting this case.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with sum_col2:
                pos_count = len(explanation["top_positive_features"])
                st.markdown(f"""
                <div style="background:#FFF5F5; border:1px solid #FFC9C9; border-left:4px solid #E03131; border-radius:8px; padding:16px; height:100%;">
                    <div style="font-size:0.85rem; color:#C92A2A; font-weight:600; text-transform:uppercase;">2. Red Flags ({pos_count})</div>
                    <div style="font-size:1.2rem; font-weight:700; color:#E03131; margin:6px 0;">Pushed Risk UP ⬆️</div>
                    <div style="font-size:0.85rem; color:#495057; line-height:1.4;">
                        Measurements that showed abnormal cell enlargement or irregular shape, increasing malignancy suspicion.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with sum_col3:
                neg_count = len(explanation["top_negative_features"])
                st.markdown(f"""
                <div style="background:#EBFBEE; border:1px solid #B2F2BB; border-left:4px solid #2F9E44; border-radius:8px; padding:16px; height:100%;">
                    <div style="font-size:0.85rem; color:#2B8A3E; font-weight:600; text-transform:uppercase;">3. Healthy Signs ({neg_count})</div>
                    <div style="font-size:1.2rem; font-weight:700; color:#2F9E44; margin:6px 0;">Pulled Risk DOWN ⬇️</div>
                    <div style="font-size:0.85rem; color:#495057; line-height:1.4;">
                        Measurements that looked normal or smooth, reducing suspicion toward a benign diagnosis.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            
            # Detailed Breakdown in Clean Points
            f_col1, f_col2 = st.columns(2)

            with f_col1:
                st.markdown("#### 🚨 Top Red Flags (Pushed Risk Toward Malignant):")
                if explanation["top_positive_features"]:
                    for idx, item in enumerate(explanation["top_positive_features"][:4], 1):
                        f_raw = item["feature"]
                        f_label = FEATURE_HUMAN_NAMES.get(f_raw, f_raw.replace("_", " ").title())
                        impact_pct = item["shap_value"] * 100
                        st.markdown(
                            f"""
                            <div style="background:#FFF0F3; border-left:4px solid #D6336C; padding:10px 14px; border-radius:6px; margin-bottom:10px; font-size:0.9rem;">
                                <b>{idx}. {f_label}</b><br>
                                Measured Value: <code>{item["feature_value"]:.2f}</code> &nbsp;|&nbsp; 
                                <b style="color:#C2255C;">+{impact_pct:.1f}% risk increase</b>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("✅ No features pushed toward malignancy.")

            with f_col2:
                st.markdown("#### 🟢 Top Healthy Indicators (Pulled Risk Toward Benign):")
                if explanation["top_negative_features"]:
                    for idx, item in enumerate(explanation["top_negative_features"][:4], 1):
                        f_raw = item["feature"]
                        f_label = FEATURE_HUMAN_NAMES.get(f_raw, f_raw.replace("_", " ").title())
                        impact_pct = abs(item["shap_value"] * 100)
                        st.markdown(
                            f"""
                            <div style="background:#F4FCE3; border-left:4px solid #74B816; padding:10px 14px; border-radius:6px; margin-bottom:10px; font-size:0.9rem;">
                                <b>{idx}. {f_label}</b><br>
                                Measured Value: <code>{item["feature_value"]:.2f}</code> &nbsp;|&nbsp; 
                                <b style="color:#5C940D;">-{impact_pct:.1f}% risk reduction</b>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("ℹ️ No features significantly lowered risk.")

            # Plain-English Decision Verdict
            final_risk = explanation['model_score'] * 100
            if final_risk >= 50:
                verdict_color = "#E03131"
                verdict_icon = "⚠️"
                verdict_text = f"The model concluded <b>Malignant ({final_risk:.1f}% risk)</b> because the cellular abnormalities (especially cell enlargement and border indentations) heavily outweighed normal indicators."
            else:
                verdict_color = "#2F9E44"
                verdict_icon = "✅"
                verdict_text = f"The model concluded <b>Benign ({100 - final_risk:.1f}% confidence)</b> because the cell size and morphology remained well within normal non-cancerous thresholds."

            st.markdown(f"""
            <div style="background:#F8F9FA; border-left:4px solid {verdict_color}; padding:14px 18px; border-radius:8px; margin-top:14px; font-size:0.95rem; color:#212529;">
                <b>{verdict_icon} Plain-English Summary:</b> {verdict_text}
            </div>
            """, unsafe_allow_html=True)

            # Optional Technical Graphs (Collapsed by default so they don't overwhelm)
            st.write("")
            with st.expander("🔬 View Detailed Technical SHAP Graphs (Waterfall & Beeswarm Plots)", expanded=False):
                st.markdown("###### Local SHAP Waterfall Attribution Plot:")
                fig_waterfall = plot_local_waterfall(
                    explanation["shap_explanation"],
                    max_display=9,
                    unit_label=explanation.get("unit_label")
                )
                st.pyplot(fig_waterfall)
                plt.close(fig_waterfall)

                st.markdown("###### Global Biomarker Importance Across Wisconsin Dataset:")
                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    biomarker_img = os.path.join(assets_dir, "shap_feature_importance.png")
                    if os.path.exists(biomarker_img):
                        st.image(biomarker_img, caption="Comparative Clinical Biomarker Importance Across 4 Classifiers", use_container_width=True)
                with g_col2:
                    rf_summary_img = os.path.join(assets_dir, "shap_summary_random_forest.png")
                    if os.path.exists(rf_summary_img):
                        st.image(rf_summary_img, caption="TreeSHAP Beeswarm Summary Plot", use_container_width=True)

                st.markdown("""
                <div class="medical-disclaimer" style="margin-top:12px;">
                    <b>⚠️ Methodological Interpretability Boundary:</b><br>
                    SHAP values reflect mathematical feature attribution and predictive reliance within this statistical learning pipeline. 
                    <b>They do NOT denote biological causation or physical disease etiology.</b>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error computing local explanation: {e}")

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
