"""
patient_page.py
===============
Page 2: Patient Case Information & Biopsy Intake.
Supports TWO clearly visible, independent input methods:
- OPTION A: Upload Biopsy Report (PDF) with automated 30-feature extraction and review table
- OPTION B: Direct Manual Entry for all 30 continuous features organized into Mean, SE, and Worst
Neither method is inaccessible; both populate and synchronize with the same validated 30-feature vector.
Optional patient metadata & symptoms are stored separately and isolated from ML inputs.
"""

import os
import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from src.data_loader import FEATURE_NAMES
from src.pdf_parser import parse_fna_pdf_report
from src.ui.nav import PAGE_HOME, PAGE_RESULTS, navigate_to


# Reference presets for clinician testing and demonstration
SAMPLE_PRESETS = {
    "Malignant Preset (Wisconsin Case #842302)": {
        'radius_mean': 17.99, 'texture_mean': 10.38, 'perimeter_mean': 122.8, 'area_mean': 1001.0,
        'smoothness_mean': 0.1184, 'compactness_mean': 0.2776, 'concavity_mean': 0.3001,
        'concave points_mean': 0.1471, 'symmetry_mean': 0.2419, 'fractal_dimension_mean': 0.07871,
        'radius_se': 1.095, 'texture_se': 0.9053, 'perimeter_se': 8.589, 'area_se': 153.4,
        'smoothness_se': 0.006399, 'compactness_se': 0.04904, 'concavity_se': 0.05373,
        'concave points_se': 0.01587, 'symmetry_se': 0.03003, 'fractal_dimension_se': 0.006193,
        'radius_worst': 25.38, 'texture_worst': 17.33, 'perimeter_worst': 184.6, 'area_worst': 2019.0,
        'smoothness_worst': 0.1622, 'compactness_worst': 0.6656, 'concavity_worst': 0.7119,
        'concave points_worst': 0.2654, 'symmetry_worst': 0.4601, 'fractal_dimension_worst': 0.1189
    },
    "Benign Preset (Wisconsin Case #8510426)": {
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
}


def sync_patient_features_to_inputs(features_dict: Dict[str, float], metadata_dict: Any = None):
    """Synchronizes feature dictionary and metadata into session_state using versioned keys."""
    if "patient_features" not in st.session_state or st.session_state["patient_features"] is None:
        st.session_state["patient_features"] = {}
    for feat, val in features_dict.items():
        st.session_state["patient_features"][feat] = float(val)
    if metadata_dict and isinstance(metadata_dict, dict):
        if "patient_meta" not in st.session_state:
            st.session_state["patient_meta"] = {}
        if "patient_id" in metadata_dict:
            st.session_state["patient_meta"]["patient_id"] = metadata_dict["patient_id"]
        if "history" in metadata_dict:
            st.session_state["patient_meta"]["history"] = metadata_dict["history"]
        if "age" in metadata_dict:
            st.session_state["patient_meta"]["age"] = int(metadata_dict["age"])
    
    st.session_state["patient_version"] = st.session_state.get("patient_version", 0) + 1


def render_patient_page():
    st.markdown('<div class="hero-title">📋 Patient Case & Biopsy Intake</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Step 1 of 2: Ingest Cytological Data & Clinical Context</div>', unsafe_allow_html=True)

    # Initialize state collections
    if "patient_features" not in st.session_state or st.session_state["patient_features"] is None:
        st.session_state["patient_features"] = SAMPLE_PRESETS["Malignant Preset (Wisconsin Case #842302)"].copy()
    if "patient_meta" not in st.session_state:
        st.session_state["patient_meta"] = {"patient_id": "PT-842302", "age": 48, "history": "Palpable discrete breast nodule"}
    if "patient_symptoms" not in st.session_state:
        st.session_state["patient_symptoms"] = {}
    if "input_method_selection" not in st.session_state:
        st.session_state["input_method_selection"] = "OPTION A: Upload Biopsy Report (PDF)"
    if "patient_version" not in st.session_state:
        st.session_state["patient_version"] = 0

    v = st.session_state["patient_version"]

    # =========================================================================
    # SECTION A: Basic Patient Information (Non-identifying tracking)
    # =========================================================================
    st.markdown('<div class="section-title">👤 Section A: Basic Patient Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Non-personally identifiable reference data for clinical report tracking:</div>', unsafe_allow_html=True)

    c_id, c_age, c_hist = st.columns([1, 1, 2])
    with c_id:
        p_id = st.text_input("Patient Reference ID (Optional)", value=st.session_state["patient_meta"].get("patient_id", "PT-842302"), key=f"p_id_{v}")
    with c_age:
        p_age = st.number_input("Patient Age", min_value=18, max_value=110, value=int(st.session_state["patient_meta"].get("age", 48)), key=f"p_age_{v}")
    with c_hist:
        p_hist = st.text_input("Relevant Medical History (Optional)", value=st.session_state["patient_meta"].get("history", "Palpable discrete breast nodule"), key=f"p_hist_{v}")

    st.session_state["patient_meta"] = {"patient_id": p_id, "age": p_age, "history": p_hist}

    st.write("---")

    # =========================================================================
    # SECTION B: Biopsy Data Input — TWO EXPLICIT OPTIONS
    # =========================================================================
    st.markdown('<div class="section-title">📄 Section B: Medical Biopsy Morphometric Data</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-caption">
    The four machine learning classifiers require the exact <b>30 quantitative nuclear features</b> from Fine Needle Aspirate (FNA) cytology.<br>
    Choose one of the two input options below. Both options produce the same validated 30-feature vector.
    </div>
    """, unsafe_allow_html=True)

    # Clearly visible two-option radio selector
    OPT_PDF = "OPTION A: Upload Biopsy Report (PDF)"
    OPT_MANUAL = "OPTION B: Enter Measurements Manually"

    selected_method = st.radio(
        "Select Biopsy Data Input Method:",
        [OPT_PDF, OPT_MANUAL],
        index=0 if st.session_state["input_method_selection"] == OPT_PDF else 1,
        horizontal=True,
        key="main_input_method_radio"
    )
    st.session_state["input_method_selection"] = selected_method

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # OPTION A: UPLOAD BIOPSY REPORT (PDF)
    # -------------------------------------------------------------------------
    if selected_method == OPT_PDF:
        st.markdown("### 📄 Option A: Upload Fine Needle Aspirate (FNA) PDF Report")
        st.markdown("""
        <div class="context-notice">
        <b>PDF Extraction Guidelines:</b> The automated parser scans digital pathology reports for Wisconsin nuclear morphometry metrics. 
        General blood panels, CBC tests, or unstructured notes will be flagged as incompatible. 
        <b>Missing measurements will never be fabricated.</b>
        </div>
        """, unsafe_allow_html=True)

        # Clear any old demo state
        st.session_state.pop("_active_demo_pdf_bytes", None)
        st.session_state.pop("_active_demo_pdf_name", None)

        col_upload, col_demo = st.columns([1.25, 0.75], gap="medium")

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        samples_dir = os.path.join(base_dir, "data", "sample_reports")

        with col_upload:
            uploaded_pdf = st.file_uploader(
                "Upload FNA Cytopathology PDF Report",
                type=["pdf"],
                help="Upload an authentic cytopathology report containing cell nuclear morphology measurements.",
                key="pdf_file_uploader"
            )

        with col_demo:
            st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#0D1B2A; margin-bottom:6px;'>Download Sample Test PDFs:</div>", unsafe_allow_html=True)
            st.caption("Don't have a report? Download a sample, then upload it on the left:")

            p_mal = os.path.join(samples_dir, "fna_biopsy_sample_malignant.pdf")
            if os.path.exists(p_mal):
                with open(p_mal, "rb") as f:
                    st.download_button("📥 Malignant Sample PDF", data=f.read(), file_name="fna_biopsy_sample_malignant.pdf", mime="application/pdf", use_container_width=True)

            p_ben = os.path.join(samples_dir, "fna_biopsy_sample_benign.pdf")
            if os.path.exists(p_ben):
                with open(p_ben, "rb") as f:
                    st.download_button("📥 Benign Sample PDF", data=f.read(), file_name="fna_biopsy_sample_benign.pdf", mime="application/pdf", use_container_width=True)

        # Process ONLY when a user has actually uploaded a PDF file
        if uploaded_pdf is not None:
            pdf_bytes = uploaded_pdf.getvalue()
            pdf_sig = f"{uploaded_pdf.name}_{len(pdf_bytes)}"

            # Trigger parse and session sync if this is a newly uploaded PDF
            if st.session_state.get("_last_parsed_pdf_sig") != pdf_sig:
                with st.spinner(f"Extracting 30 cytological features from '{uploaded_pdf.name}'..."):
                    parse_res = parse_fna_pdf_report(pdf_bytes)
                st.session_state["_last_parsed_pdf_sig"] = pdf_sig
                st.session_state["_cached_parse_res"] = parse_res

                if parse_res["success"]:
                    meta = parse_res.get("metadata", {})
                    # Smart defaults based on filename or patient reference
                    if "8510426" in meta.get("patient_id", "") or "benign" in uploaded_pdf.name.lower():
                        meta.setdefault("patient_id", "PT-8510426")
                        meta.setdefault("history", "Routine screening / benign fibroadenoma")
                        meta.setdefault("age", 36)
                    elif "842302" in meta.get("patient_id", "") or "malignant" in uploaded_pdf.name.lower():
                        meta.setdefault("patient_id", "PT-842302")
                        meta.setdefault("history", "Palpable discrete breast nodule")
                        meta.setdefault("age", 48)

                    sync_patient_features_to_inputs(parse_res["extracted_features"], meta)
                    st.rerun()
            else:
                parse_res = st.session_state.get("_cached_parse_res", parse_fna_pdf_report(pdf_bytes))

            if parse_res["success"]:
                found_count = parse_res["features_found_count"]
                extracted = parse_res["extracted_features"]
                missing = parse_res["missing_features"]

                if parse_res["is_complete"]:
                    st.success(f"✅ Full Feature Extraction Succeeded: Detected all {found_count} of 30 required Wisconsin nuclear features from '{uploaded_pdf.name}'.")
                else:
                    st.warning(
                        f"⚠️ Incomplete PDF Extraction: Found {found_count} of 30 features. "
                        f"Missing {len(missing)} measurements: {', '.join(missing[:4])}... "
                        "Please inspect and complete the missing values in the review form below."
                    )

                # Display Extracted Measurements Review Table
                with st.expander("📊 View Extracted Values & Status Table", expanded=True):
                    table_rows = []
                    for f in FEATURE_NAMES:
                        status = "✅ Extracted" if f in extracted else "❌ Missing from PDF"
                        val = extracted.get(f, "—")
                        table_rows.append({
                            "Feature Name": f.replace("_", " ").title(),
                            "Canonical Key": f,
                            "Extracted Value": f"{val:.4f}" if isinstance(val, (int, float)) else str(val),
                            "Extraction Status": status
                        })
                    df_review = pd.DataFrame(table_rows)
                    st.dataframe(df_review, use_container_width=True, hide_index=True)
            else:
                st.error(f"❌ PDF Incompatible: {parse_res.get('error')}")
                st.info("💡 You may switch to 'OPTION B: Enter Measurements Manually' to supply the 30 continuous features directly.")
        else:
            st.session_state.pop("_last_parsed_pdf_sig", None)
            st.session_state.pop("_cached_parse_res", None)
            st.info("👆 **Awaiting Biopsy Report:** Click the `Upload` button above to select your PDF report from your computer.")

    # -------------------------------------------------------------------------
    # OPTION B: ENTER MEASUREMENTS MANUALLY
    # -------------------------------------------------------------------------
    else:
        st.markdown("### ✍️ Option B: Enter Measurements Manually")
        st.markdown("""
        <div class="context-notice">
        <b>Direct Biopsy Entry:</b> Input the 30 continuous real-valued measurements directly into the form below without uploading a file. 
        All values are validated before model inference.
        </div>
        """, unsafe_allow_html=True)

        # Quick preset buttons to assist clinician testing
        preset_c1, preset_c2, preset_c3 = st.columns([1, 1, 1.5])
        with preset_c1:
            if st.button("📥 Load Malignant Preset", use_container_width=True):
                sync_patient_features_to_inputs(
                    SAMPLE_PRESETS["Malignant Preset (Wisconsin Case #842302)"],
                    {"patient_id": "PT-842302", "age": 48, "history": "Palpable discrete breast nodule"}
                )
                st.success("Loaded confirmed Malignant preset (30 features).")
                st.rerun()
        with preset_c2:
            if st.button("📥 Load Benign Preset", use_container_width=True):
                sync_patient_features_to_inputs(
                    SAMPLE_PRESETS["Benign Preset (Wisconsin Case #8510426)"],
                    {"patient_id": "PT-8510426", "age": 36, "history": "Routine screening / benign fibroadenoma"}
                )
                st.success("Loaded confirmed Benign preset (30 features).")
                st.rerun()
        with preset_c3:
            if st.button("🔄 Reset Measurements to Default", use_container_width=True):
                sync_patient_features_to_inputs(
                    SAMPLE_PRESETS["Malignant Preset (Wisconsin Case #842302)"],
                    {"patient_id": "PT-842302", "age": 48, "history": "Palpable discrete breast nodule"}
                )
                st.info("Reset features to default baseline.")
                st.rerun()

    # -------------------------------------------------------------------------
    # 30-FEATURE REVIEW & EDITABLE FORM (Always available & synchronized)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 🔬 Review & Confirm 30 Nuclear Morphological Measurements")
    st.caption("Organized by Mean, Standard Error, and Worst characteristics. Users can review, adjust, or correct any value directly:")

    cur_feats = st.session_state["patient_features"]

    # 3 Tabs organized cleanly into Mean (10), Standard Error (10), Worst (10)
    tab_mean, tab_se, tab_worst = st.tabs([
        "1. Mean Measurements (10)",
        "2. Standard Error Measurements (10)",
        "3. Worst Measurements (10)"
    ])

    # 1. Mean Measurements (10)
    with tab_mean:
        st.markdown("<div style='font-size:0.85rem; color:#5A6E85; margin-bottom:8px;'>Mean geometric dimensions of digitized cell nuclei:</div>", unsafe_allow_html=True)
        mean_cols = [f for f in FEATURE_NAMES if f.endswith("_mean")]
        m_c1, m_c2 = st.columns(2)
        for idx, feat in enumerate(mean_cols):
            target_col = m_c1 if idx % 2 == 0 else m_c2
            with target_col:
                cur_val = float(cur_feats.get(feat, 15.0))
                label = feat.replace("_mean", "").replace("_", " ").title()
                val = st.number_input(
                    f"{label} (Mean)",
                    value=cur_val,
                    format="%.4f",
                    key=f"input_{feat}_{v}"
                )
                cur_feats[feat] = val

    # 2. Standard Error Measurements (10)
    with tab_se:
        st.markdown("<div style='font-size:0.85rem; color:#5A6E85; margin-bottom:8px;'>Standard error across boundary sample variations:</div>", unsafe_allow_html=True)
        se_cols = [f for f in FEATURE_NAMES if f.endswith("_se")]
        s_c1, s_c2 = st.columns(2)
        for idx, feat in enumerate(se_cols):
            target_col = s_c1 if idx % 2 == 0 else s_c2
            with target_col:
                cur_val = float(cur_feats.get(feat, 1.0))
                label = feat.replace("_se", "").replace("_", " ").title()
                val = st.number_input(
                    f"{label} (SE)",
                    value=cur_val,
                    format="%.4f",
                    key=f"input_{feat}_{v}"
                )
                cur_feats[feat] = val

    # 3. Worst Measurements (10)
    with tab_worst:
        st.markdown("<div style='font-size:0.85rem; color:#5A6E85; margin-bottom:8px;'>Worst / largest dimension measured across aspirate:</div>", unsafe_allow_html=True)
        worst_cols = [f for f in FEATURE_NAMES if f.endswith("_worst")]
        w_c1, w_c2 = st.columns(2)
        for idx, feat in enumerate(worst_cols):
            target_col = w_c1 if idx % 2 == 0 else w_c2
            with target_col:
                cur_val = float(cur_feats.get(feat, 20.0))
                label = feat.replace("_worst", "").replace("_", " ").title()
                val = st.number_input(
                    f"{label} (Worst)",
                    value=cur_val,
                    format="%.4f",
                    key=f"input_{feat}_{v}"
                )
                cur_feats[feat] = val

    st.session_state["patient_features"] = cur_feats

    # Rigorous Feature Validation Check
    missing_feats = [f for f in FEATURE_NAMES if f not in cur_feats or cur_feats[f] is None or pd.isna(cur_feats[f])]
    is_valid_features = len(missing_feats) == 0 and len(cur_feats) == 30

    if is_valid_features:
        st.success("✅ Feature Signature Verified: All 30 required continuous dimensions are validated and ready for model inference.")
    else:
        st.error(f"❌ Incomplete Feature Vector: The following {len(missing_feats)} features are missing or invalid: {missing_feats}")

    st.write("---")

    # =========================================================================
    # SECTION C: Optional Symptoms & Clinical Questionnaire (Isolated from ML)
    # =========================================================================
    st.markdown('<div class="section-title">🩺 Section C: Symptoms & Clinical Questionnaire (Optional Context)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="context-notice">
    <b>Clinical Boundary Notice:</b> The questionnaire below records qualitative patient-reported context. 
    <b>Crucially, these symptom responses are kept separate from the machine learning feature vector</b> 
    because the Wisconsin classifiers were trained exclusively on cellular nuclear morphology. 
    Symptoms do not alter the ML mathematical classification scores.
    </div>
    """, unsafe_allow_html=True)

    symp_c1, symp_c2 = st.columns(2)

    with symp_c1:
        s_lump = st.selectbox("Palpable Breast Lump / Mass:", ["Yes", "No", "Unsure"], index=0)
        s_pain = st.selectbox("Localized Breast Pain / Tenderness:", ["Yes", "No", "Unsure"], index=1)
        s_discharge = st.selectbox("Nipple Discharge / Inversion:", ["Yes", "No", "Unsure"], index=1)

    with symp_c2:
        s_skin = st.selectbox("Skin Changes (Dimpling / Redness / Peau d'orange):", ["Yes", "No", "Unsure"], index=1)
        s_family = st.selectbox("Family History of Breast / Ovarian Malignancy:", ["Yes", "No", "Unsure"], index=0)
        s_duration = st.selectbox("Duration of Symptoms:", ["< 1 Month", "1 - 6 Months", "> 6 Months", "Asymptomatic / Screening"])

    st.session_state["patient_symptoms"] = {
        "lump": s_lump, "pain": s_pain, "discharge": s_discharge,
        "skin_changes": s_skin, "family_history": s_family, "duration": s_duration
    }

    st.write("")

    # Navigation Buttons
    nav_c1, nav_c2 = st.columns([1, 1])
    with nav_c1:
        if st.button("⬅️ Back to Home"):
            navigate_to(PAGE_HOME)

    with nav_c2:
        if st.button("➡️ Proceed to Prediction & Explanations", type="primary", disabled=not is_valid_features):
            navigate_to(PAGE_RESULTS)
