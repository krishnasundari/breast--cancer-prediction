"""
app.py
======
Professional Multi-Page Breast Cancer Detection & Explainable AI System.
Organized into structured, dedicated clinical views:
- Home: Landing, doctor illustration, model highlights
- Patient Intake: Non-identifying metadata, PDF pathology report upload, feature validation
- Prediction & Explanations: 4-model consensus, evaluation comparisons, local & global SHAP
- About: Cytological background, mathematical formulation, clinical disclaimers
"""

import os
import sys
import streamlit as st

# Add repo root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ui.styles import apply_custom_styles
from src.ui.nav import PAGES, PAGE_HOME, PAGE_INTAKE, PAGE_RESULTS, PAGE_ABOUT, navigate_to
from src.ui.home_page import render_home_page
from src.ui.patient_page import render_patient_page
from src.ui.results_page import render_results_page
from src.ui.about_page import render_about_page

st.set_page_config(
    page_title="Breast Cancer Detection | AI Diagnostic System",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global medical styling
apply_custom_styles()

# Consume any queued navigation BEFORE radio widget instantiation
if st.session_state.get("_pending_nav"):
    target = st.session_state.pop("_pending_nav")
    st.session_state["nav_radio_selection"] = target
    st.session_state["current_page"] = target
elif "nav_radio_selection" not in st.session_state:
    st.session_state["nav_radio_selection"] = PAGE_HOME
    st.session_state["current_page"] = PAGE_HOME

# Sidebar Brand Header
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 22px; padding-bottom: 12px; border-bottom: 1px solid #E2E8F0;">
    <div style="font-size: 2.2rem; margin-bottom: 4px;">🎗️</div>
    <div style="font-size: 1.15rem; font-weight: 800; color: #0D1B2A; letter-spacing: -0.01em;">Breast Cancer Detection</div>
    <div style="font-size: 0.8rem; color: #E91E63; font-weight: 600; margin-top: 2px;">Explainable AI Diagnostic Platform</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #5A6E85; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;'>Navigation Menu</div>", unsafe_allow_html=True)


def on_nav_change():
    st.session_state["current_page"] = st.session_state["nav_radio_selection"]


selected_page = st.sidebar.radio(
    "Navigation Menu",
    PAGES,
    key="nav_radio_selection",
    on_change=on_nav_change,
    label_visibility="collapsed"
)
st.session_state["current_page"] = selected_page

st.sidebar.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="background:#F8F9FA; border-left:3px solid #E91E63; padding:12px; border-radius:6px; font-size:0.78rem; color:#5A6E85; line-height:1.45;">
    <b>Clinical Cytology System</b><br>
    Validated on Wisconsin Diagnostic Cohort (WDBC). Verified across 4 machine learning pipelines with SHAP explainability.
</div>
""", unsafe_allow_html=True)

# Dispatch to active page
current = st.session_state["current_page"]

if current == PAGE_HOME:
    render_home_page()
elif current == PAGE_INTAKE:
    render_patient_page()
elif current == PAGE_RESULTS:
    render_results_page()
elif current == PAGE_ABOUT:
    render_about_page()
