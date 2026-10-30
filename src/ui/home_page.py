"""
home_page.py
============
Page 1: Minimal, clean, and professional healthcare landing page.
Features:
- Header: "Breast Cancer Detection" with clean, crisp typography
- Left Section: Clear description and prominent "Start Diagnosis" button
- Right Section: Doctor holding a stethoscope photo/illustration (from reference image)
- Bottom Section: "Powered by Four Machine Learning Models" with 4 compact cards
- Educational Notice: Clear clinical decision support disclaimer
"""

import os
import base64
import streamlit as st
from src.ui.nav import PAGE_INTAKE, navigate_to


def get_doctor_image_base64() -> str:
    """Reads assets/doctor_holding_stethoscope.png and returns base64 encoded string."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    img_path = os.path.join(base_dir, "assets", "doctor_holding_stethoscope.png")
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


def render_home_page():
    # Top Header Title
    st.markdown("""
    <div style="text-align: center; margin-top: 0.5rem; margin-bottom: 2.2rem;">
        <h1 style="font-size: 2.9rem; font-weight: 800; color: #0D1B2A; letter-spacing: -0.02em; margin-bottom: 0.25rem;">
            Breast Cancer Detection
        </h1>
        <div style="font-size: 1.25rem; font-weight: 600; color: #E91E63; letter-spacing: -0.01em;">
            AI-Powered Breast Cancer Classification
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Two-Column Grid: Left (Description + CTA), Right (Doctor Photo)
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.markdown("""
        <div style="padding-top: 1.2rem; margin-bottom: 1.8rem;">
            <p style="font-size: 1.35rem; font-weight: 600; color: #0D1B2A; line-height: 1.6; margin: 0;">
                An intelligent system that analyzes validated biopsy measurements using four machine-learning models and provides explainable results.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Start Diagnosis", type="primary", use_container_width=True):
            navigate_to(PAGE_INTAKE)

    with col_right:
        doc_b64 = get_doctor_image_base64()
        if doc_b64:
            doctor_html = f"""
            <div style="text-align: center; display: flex; justify-content: center; align-items: center;">
                <div class="doctor-floating" style="background: #FFFFFF; border-radius: 18px; padding: 12px; border: 1px solid #E2E8F0; box-shadow: 0 8px 24px rgba(233, 30, 99, 0.08);">
                    <img src="data:image/png;base64,{doc_b64}" 
                         alt="Doctor holding stethoscope" 
                         style="max-width: 280px; width: 100%; height: auto; display: block; border-radius: 10px;" />
                </div>
            </div>
            """
            st.markdown(doctor_html, unsafe_allow_html=True)
        else:
            # Fallback if image path is unavailable
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            img_path = os.path.join(base_dir, "assets", "doctor_holding_stethoscope.png")
            if os.path.exists(img_path):
                st.image(img_path, width=280)

    # Section Below Hero: Powered by Four Machine Learning Models
    st.markdown("""
    <div style="text-align: center; margin-top: 2.8rem; margin-bottom: 1.2rem;">
        <div style="font-size: 1.15rem; font-weight: 700; color: #0D1B2A; letter-spacing: -0.01em;">
            ⚡ Powered by Four Machine Learning Models
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="compact-model-card">
            <span class="compact-model-icon">📍</span>
            <div class="compact-model-title">K-Nearest Neighbors</div>
            <div class="compact-model-tag">97.4% Acc • K=6</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="compact-model-card">
            <span class="compact-model-icon">📈</span>
            <div class="compact-model-title">Logistic Regression</div>
            <div class="compact-model-tag">96.5% Acc • L2 Log-Odds</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="compact-model-card">
            <span class="compact-model-icon">⚡</span>
            <div class="compact-model-title">Support Vector Machine</div>
            <div class="compact-model-tag">97.4% Acc • RBF Kernel</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="compact-model-card">
            <span class="compact-model-icon">🌲</span>
            <div class="compact-model-title">Random Forest</div>
            <div class="compact-model-tag">97.4% Acc • 100 Trees</div>
        </div>
        """, unsafe_allow_html=True)

    # Educational Disclaimer Notice
    st.markdown("""
    <div class="medical-disclaimer" style="margin-top: 2.2rem; text-align: center;">
        <b>⚕️ Educational & Clinical Decision Support Notice:</b><br>
        This platform analyzes 30 quantitative morphometric features from digitized Fine Needle Aspirate (FNA) biopsies. 
        It is designed strictly for algorithm benchmarking and explainable AI evaluation, and does not replace certified pathologist review.
    </div>
    """, unsafe_allow_html=True)
