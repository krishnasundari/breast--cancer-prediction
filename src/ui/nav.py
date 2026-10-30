"""
nav.py
======
Unified navigation state and page routing constants for the Streamlit app.
Ensures seamless bidirectional navigation between:
Home -> Patient Intake -> Prediction & Explanations -> About
without widget instantiation errors or desynchronization.
"""

import streamlit as st

PAGE_HOME = "🏠 Home"
PAGE_INTAKE = "📋 Patient Intake"
PAGE_RESULTS = "🔬 Prediction & Explanations"
PAGE_ABOUT = "ℹ️ About"

PAGES = [PAGE_HOME, PAGE_INTAKE, PAGE_RESULTS, PAGE_ABOUT]


def navigate_to(page_name: str) -> None:
    """
    Safely queues target page navigation and triggers a rerun.
    The pending navigation is consumed in app.py before the radio widget is instantiated,
    avoiding StreamlitWidgetAlreadyInstantiatedError.
    """
    if page_name in PAGES:
        st.session_state["_pending_nav"] = page_name
        st.session_state["current_page"] = page_name
        st.rerun()
