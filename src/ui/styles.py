"""
styles.py
=========
Global design tokens, medical color palette, and custom styling for the
Breast Cancer Detection & Explainable AI platform.
"""

# Palette definition
COLOR_BG_WHITE = "#FFFFFF"
COLOR_BG_LIGHT = "#FAFBFD"
COLOR_NAVY_HEADING = "#0D1B2A"
COLOR_NAVY_BODY = "#1B263B"
COLOR_TEXT_MUTED = "#5A6E85"
COLOR_ACCENT_PINK = "#E91E63"
COLOR_ACCENT_PINK_LIGHT = "#FCE4EC"
COLOR_ACCENT_PINK_BORDER = "#F8BBD0"
COLOR_PRIMARY_BLUE = "#1976D2"
COLOR_SUCCESS_GREEN = "#2E7D32"
COLOR_SUCCESS_BG = "#E8F5E9"
COLOR_WARNING_RED = "#C62828"
COLOR_WARNING_BG = "#FFEBEE"

CUSTOM_CSS = f"""
<style>
    /* Global Page Styling */
    .stApp {{
        background-color: {COLOR_BG_LIGHT};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: {COLOR_NAVY_BODY};
    }}

    /* Main Container Padding */
    .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1200px;
    }}

    /* Sidebar Dimensions & Layout */
    section[data-testid="stSidebar"] {{
        width: 275px !important;
        min-width: 260px !important;
        max-width: 285px !important;
        background-color: {COLOR_BG_WHITE} !important;
        border-right: 1px solid #E2E8F0;
    }}

    section[data-testid="stSidebar"] .block-container {{
        padding-top: 2rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }}

    /* Sidebar Navigation Radio - Scoped strictly to Sidebar */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {{
        gap: 8px !important;
        display: flex;
        flex-direction: column;
    }}

    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label {{
        background: #F8F9FA;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 4px;
        transition: all 0.2s ease;
        cursor: pointer;
        width: 100%;
        box-sizing: border-box;
    }}

    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
        background: {COLOR_ACCENT_PINK_LIGHT};
        border-color: {COLOR_ACCENT_PINK_BORDER};
    }}

    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label p {{
        font-size: 16.5px !important;
        font-weight: 600 !important;
        color: {COLOR_NAVY_HEADING} !important;
        line-height: 1.4 !important;
        margin: 0 !important;
    }}

    /* Sidebar Active Highlighted Radio State */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {{
        background: {COLOR_ACCENT_PINK_LIGHT} !important;
        border: 1.5px solid {COLOR_ACCENT_PINK} !important;
        box-shadow: 0 2px 8px rgba(233, 30, 99, 0.12);
    }}

    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] p {{
        color: #C2185B !important;
        font-weight: 700 !important;
    }}

    /* Main Content Input Option Selector Cards */
    .stApp div[data-testid="stRadio"] label p {{
        font-size: 1.05rem;
        font-weight: 600;
        color: {COLOR_NAVY_HEADING};
    }}

    /* Input Field Clarity & Theme Consistency */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {{
        background-color: #FFFFFF !important;
        color: #0D1B2A !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stNumberInput"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label {{
        color: #1B263B !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }}

    /* Header & Titles */
    .hero-title {{
        font-size: 2.7rem;
        font-weight: 800;
        color: {COLOR_NAVY_HEADING};
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin-bottom: 0.35rem;
    }}

    .hero-subtitle {{
        font-size: 1.25rem;
        font-weight: 600;
        color: {COLOR_ACCENT_PINK};
        margin-bottom: 1.1rem;
        letter-spacing: -0.01em;
    }}

    .hero-desc {{
        font-size: 1.05rem;
        line-height: 1.65;
        color: {COLOR_TEXT_MUTED};
        margin-bottom: 1.8rem;
    }}

    .section-title {{
        font-size: 1.45rem;
        font-weight: 700;
        color: {COLOR_NAVY_HEADING};
        margin-top: 1.8rem;
        margin-bottom: 0.7rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .section-caption {{
        font-size: 0.95rem;
        color: {COLOR_TEXT_MUTED};
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }}

    /* Floating Doctor Animation */
    @keyframes doctorFloat {{
        0% {{
            transform: translateY(0px);
        }}
        50% {{
            transform: translateY(-8px);
        }}
        100% {{
            transform: translateY(0px);
        }}
    }}

    .doctor-floating {{
        animation: doctorFloat 3.8s ease-in-out infinite;
        display: inline-block;
        filter: drop-shadow(0 8px 24px rgba(233, 30, 99, 0.12));
    }}

    /* Compact Model Highlight Cards */
    .compact-model-card {{
        background: {COLOR_BG_WHITE};
        border: 1px solid #E2E8F0;
        border-top: 3.5px solid {COLOR_ACCENT_PINK};
        border-radius: 12px;
        padding: 16px 14px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(13, 27, 42, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }}

    .compact-model-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(233, 30, 99, 0.12);
    }}

    .compact-model-icon {{
        font-size: 1.9rem;
        margin-bottom: 6px;
    }}

    .compact-model-title {{
        font-size: 1.02rem;
        font-weight: 700;
        color: {COLOR_NAVY_HEADING};
        margin-bottom: 6px;
        line-height: 1.25;
    }}

    .compact-model-tag {{
        font-size: 0.8rem;
        font-weight: 600;
        color: #C2185B;
        background: {COLOR_ACCENT_PINK_LIGHT};
        padding: 3px 10px;
        border-radius: 12px;
        display: inline-block;
    }}

    /* Standard Cards */
    .med-card {{
        background: {COLOR_BG_WHITE};
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(13, 27, 42, 0.03);
        margin-bottom: 1.2rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}

    .med-card:hover {{
        box-shadow: 0 6px 18px rgba(13, 27, 42, 0.06);
    }}

    .med-card-highlight {{
        background: {COLOR_BG_WHITE};
        border-top: 4px solid {COLOR_ACCENT_PINK};
        border-radius: 12px;
        border-left: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        padding: 1.4rem;
        box-shadow: 0 4px 14px rgba(233, 30, 99, 0.05);
        height: 100%;
    }}

    .med-badge-malignant {{
        background-color: {COLOR_WARNING_BG};
        color: {COLOR_WARNING_RED};
        font-weight: 800;
        padding: 12px 24px;
        border-radius: 10px;
        font-size: 1.4rem;
        display: inline-block;
        border: 2px solid #EF9A9A;
        letter-spacing: -0.01em;
    }}

    .med-badge-benign {{
        background-color: {COLOR_SUCCESS_BG};
        color: {COLOR_SUCCESS_GREEN};
        font-weight: 800;
        padding: 12px 24px;
        border-radius: 10px;
        font-size: 1.4rem;
        display: inline-block;
        border: 2px solid #A5D6A7;
        letter-spacing: -0.01em;
    }}

    /* Workflow Steps */
    .step-number {{
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background-color: {COLOR_ACCENT_PINK};
        color: {COLOR_BG_WHITE};
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }}

    .step-card {{
        background: {COLOR_BG_WHITE};
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.3rem;
        height: 100%;
    }}

    /* Disclaimers */
    .medical-disclaimer {{
        background-color: #FFFDE7;
        border-left: 4px solid #FBC02D;
        border-radius: 8px;
        padding: 14px 18px;
        color: #5D4037;
        font-size: 0.88rem;
        line-height: 1.55;
        margin-top: 1.5rem;
    }}

    .context-notice {{
        background-color: {COLOR_ACCENT_PINK_LIGHT};
        border-left: 4px solid {COLOR_ACCENT_PINK};
        border-radius: 8px;
        padding: 12px 16px;
        color: #880E4F;
        font-size: 0.88rem;
        line-height: 1.5;
        margin-bottom: 1.2rem;
    }}

    /* SHAP factor cards */
    .shap-factor-pos {{
        background-color: #FFF3E0;
        border-left: 4px solid #E65100;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }}

    .shap-factor-neg {{
        background-color: #E1F5FE;
        border-left: 4px solid #0288D1;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }}

    /* Primary Action Buttons */
    div.stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #E91E63 0%, #C2185B 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        padding: 0.75rem 1.75rem !important;
        box-shadow: 0 4px 14px rgba(233, 30, 99, 0.28) !important;
        transition: all 0.2s ease !important;
    }}

    div.stButton > button[kind="primary"]:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(233, 30, 99, 0.4) !important;
        background: linear-gradient(135deg, #EC407A 0%, #AD1457 100%) !important;
    }}
</style>
"""


def apply_custom_styles():
    """Injects custom CSS styling into Streamlit app."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
