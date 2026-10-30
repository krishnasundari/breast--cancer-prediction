"""
about_page.py
=============
Page 4: About the Project, Dataset Background, Mathematical Formulations,
Engineering Principles, and Educational Disclaimer.
"""

import streamlit as st


def render_about_page():
    st.markdown('<div class="hero-title">ℹ️ About the Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Comprehensive Technical, Cytological, and Methodological Overview</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.markdown("""
        #### 1. Wisconsin Breast Cancer Diagnostic Dataset (WDBC)
        The dataset was created by **Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian** at the University of Wisconsin Hospitals, Madison. 
        Features are computed from high-resolution digitized images of a **Fine Needle Aspirate (FNA)** of breast tissue masses:
        - **Total Biopsy Cases:** 569 patients
        - **Class Distribution:** 357 Benign (62.74%), 212 Malignant (37.26%)
        - **Feature Attributes:** 30 continuous real-valued dimensions
        - **Target Encoding:** Binary diagnosis (Malignant = 1, Benign = 0)

        #### 2. The 30 Cytological Morphometric Features
        Ten baseline morphological characteristics of cell nuclei are measured across each digitized aspirate, with three statistics reported for each (**Mean**, **Standard Error (SE)**, and **Worst / Largest**):
        1. **Radius:** Mean of distances from center of mass to points on the cell nuclear perimeter.
        2. **Texture:** Standard deviation of gray-scale intensities in the nuclear region.
        3. **Perimeter:** Total length of the outer nuclear membrane boundary.
        4. **Area:** Enclosed nuclear area in square pixels.
        5. **Smoothness:** Local variation in radius lengths ($R_{\text{diff}} / R_{\text{mean}}$).
        6. **Compactness:** Measure of contour complexity: $\\frac{\\text{perimeter}^2}{\\text{area}} - 1.0$.
        7. **Concavity:** Severity and depth of concave indentations along the contour.
        8. **Concave Points:** Total count of individual concave indentations.
        9. **Symmetry:** Relative length difference between perpendicular nuclear axes.
        10. **Fractal Dimension:** Measure of boundary irregularity using coastline approximation ($FD = \\text{dim} - 1$).
        """)

    with col2:
        st.markdown("""
        #### 3. Classification Algorithms & Pipelines
        - **K-Nearest Neighbors (KNN):**
          Distance-weighted neighborhood classification ($w = 1/d$) with $K=6$, standardizing continuous dimensions via `StandardScaler`.
        - **Logistic Regression:**
          Probabilistic generalized linear model with L2 regularization ($C=1.0$), optimizing logistic loss via the L-BFGS solver.
        - **Support Vector Machine (SVM):**
          Maximum-margin hyperplane with non-linear Radial Basis Function (RBF) kernel and Platt scaling for calibrated posterior probability estimates.
        - **Random Forest:**
          Ensemble of 100 decorrelated decision trees using Gini impurity splits. Trained on raw feature scales to exploit tree scale-invariance.

        #### 4. Preprocessing & Leakage Prevention
        - **Strict Train/Test Partitioning:** Stratified 80/20 train/test split (`random_state=42`) preserves real-world class prevalence.
        - **Zero Data Leakage:** Preprocessing (`StandardScaler`) is encapsulated strictly inside scikit-learn pipelines and fit solely on training folds.
        - **Feature Invariance:** Canonical column sequences (`FEATURE_NAMES`) prevent ordering errors during deployment.

        #### 5. Explainable AI with SHAP
        - **TreeSHAP (Random Forest):** Exact polynomial-time Shapley values without sampling approximations.
        - **LinearSHAP (Logistic Regression):** Closed-form linear margin attributions with independent training background maskers.
        - **KernelSHAP (SVM & KNN):** Weighted coalitional linear regression on predicted probabilities with representative training backgrounds.
        """)

    st.write("---")

    st.markdown("""
    #### 6. Known Methodological Limitations
    1. **Observational Attribution vs. Causality:** SHAP attributions reflect how models make mathematical decisions; they do not represent biological causes of oncogenesis.
    2. **Cytological Collinearity:** Geometric dimensions (e.g. radius, perimeter, and area) share natural collinearity ($r > 0.9$). SHAP divides credit across allied features.
    3. **Generalization Scope:** Models were trained specifically on Wisconsin FNA biopsy aspirates. They cannot process mammograms, MRI images, blood panels, or genomic sequencing.

    <div class="medical-disclaimer">
        <b>⚕️ Educational & Non-Diagnostic Disclaimer:</b><br>
        This software application, associated source code, and trained machine learning models are developed strictly for academic evaluation, algorithm benchmarking, and Explainable AI research. 
        They do NOT constitute medical advice, nor are they cleared as medical diagnostic devices by any regulatory agency. 
        All clinical interpretations and patient care decisions must be conducted by certified healthcare professionals.
    </div>
    """, unsafe_allow_html=True)
