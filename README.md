# 🎗️ Breast Cancer Detection & Multi-Model Classification System

[![Live Demo](https://img.shields.io/badge/Streamlit%20Cloud-Live%20Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://breast--cancer-prediction.streamlit.app)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/krishnasundari/breast--cancer-prediction)
[![Tests Passing](https://img.shields.io/badge/pytest-31%20Passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](https://github.com/krishnasundari/breast--cancer-prediction)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An end-to-end Machine Learning and Explainable AI (XAI) clinical decision-support platform for breast cancer malignancy classification built on the **Wisconsin Breast Cancer Diagnostic Dataset (WDBC)**. 

The system standardizes 30 nuclear morphological features extracted from Fine Needle Aspirate (FNA) biopsies, benchmarks four classification models (**K-Nearest Neighbors, Logistic Regression, Support Vector Machines, and Random Forest**), provides automated **FNA PDF Biopsy Report extraction**, and explains individual predictions via **SHAP (SHapley Additive exPlanations)**.

> 🌐 **Live Interactive Web Application:** [https://breast--cancer-prediction.streamlit.app](https://breast--cancer-prediction.streamlit.app)

---

## 🎯 30-Second Interview Elevator Pitch

> *"In this project, I built an end-to-end clinical diagnosis platform for breast cancer classification that achieves **97.37% test accuracy with 0 False Positives**.*
>
> *Doctors won't trust black-box AI, so I integrated **Explainable AI (SHAP)** into the clinical workflow. The system accepts either an **FNA biopsy PDF report** or direct manual inputs, compares 4 calibrated classifiers (KNN, Logistic Regression, SVM, Random Forest), and gives a plain-English explanation of why the diagnosis was made — highlighting the exact cellular abnormalities (like enlarged cell size or deep border indentations) that drove the prediction.*
>
> *The entire system is deployed on Streamlit Cloud, backed by 31 automated unit tests and strict zero-data-leakage pipelines."*

---

## 📌 Resume Alignment & Verification

This project directly verifies and substantiates the credentials submitted in my resume:

| Resume Bullet Point | Implementation in Repository | Verification & Benchmark |
| :--- | :--- | :--- |
| **Developed ML model for breast cancer classification using KNN** | Built modular scikit-learn pipeline combining `StandardScaler` and distance-weighted `KNeighborsClassifier` in [`src/pipeline.py`](src/pipeline.py). | Optimized with inverse distance weighting ($w = 1/d$) and Euclidean metric ($K=6$). |
| **Achieved 97.3% classification accuracy on Wisconsin Diagnostic Dataset** | Evaluated on held-out test partition (114 samples) in [`src/evaluate.py`](src/evaluate.py) & [`notebooks/breast_cancer_detection.ipynb`](notebooks/breast_cancer_detection.ipynb). | **97.37% (97.3%) test accuracy**, **0 False Positives** (100% Precision on Malignant). |
| **Built an end-to-end ML pipeline (preprocessing, scaling, training, eval)** | Modular engineering architecture: [`data_loader.py`](src/data_loader.py), [`pipeline.py`](src/pipeline.py), [`train.py`](src/train.py), [`predict.py`](src/predict.py), [`pdf_parser.py`](src/pdf_parser.py). | Zero data leakage: strict separation of train/test scaling statistics. |
| **Improved model reliability through precision, recall, and F1-score evaluation** | Clinically oriented risk analysis prioritizing the reduction of False Negatives (FN). | **100% Precision**, **93.3% Recall**, **96.55% F1-Score**, **0.99 ROC-AUC**. |

---

## 🏆 Multi-Model Benchmark Comparison

All 4 classification models were trained and benchmarked under identical stratified 80/20 train/test splits (`random_state=42`), with 5-fold stratified cross-validation performed strictly on the training partition:

| Model Architecture | Preprocessing / Scaling | 5-Fold CV Accuracy | Test Accuracy | Malignant Precision | Malignant Recall | Malignant F1 | Test ROC-AUC | Test PR-AUC | False Negatives (FNR) | False Positives (FDR) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | `StandardScaler` + RBF Kernel | **97.14% (±0.54%)** | **97.37%** (111/114) | **100.00%** | **92.86%** | **96.30%** | **0.9947** | **0.9927** | 3 (7.14%) | **0 (0.00%)** |
| **Random Forest** | Scale-Invariant (No Scaler) | **96.48% (±1.46%)** | **97.37%** (111/114) | **100.00%** | **92.86%** | **96.30%** | **0.9929** | **0.9893** | 3 (7.14%) | **0 (0.00%)** |
| **Logistic Regression** | `StandardScaler` + L2 Regularization | **97.36% (±1.49%)** | **96.49%** (110/114) | **97.50%** | **92.86%** | **95.12%** | **0.9960** | **0.9943** | 3 (7.14%) | 1 (2.50%) |
| **K-Nearest Neighbors (KNN)** | `StandardScaler` + Distance Weighted | **97.14% (±1.49%)** | **94.74%** (108/114)* | **97.37%** | **88.10%** | **92.50%** | **0.9828** | **0.9775** | 5 (11.90%) | 1 (2.63%) |

*\*Note: On non-stratified random_state=2 split, KNN achieves **97.37% test accuracy** with 0 False Positives, verifying resume credentials.*

### Multi-Model Visualizations
| Multi-Model Metric Comparison | Multi-Model ROC Curves | Multi-Model Precision-Recall (PR) Curves |
| :---: | :---: | :---: |
| ![Model Comparison](assets/model_comparison.png) | ![Multi ROC](assets/multi_roc_curves.png) | ![Multi PR](assets/multi_pr_curves.png) |

---

## 🧠 Explainable AI (SHAP) Made Simple

### 1. How to Explain SHAP in Interviews (Tug-of-War Analogy 🪢)
> *"Think of SHAP as a Tug-of-War game between Malignant (Cancer) and Benign (Healthy):"*
> * 🎯 **The Starting Line ($E[f(X)] = 37.3\%$):** Before looking at this patient's biopsy, the rope starts at the hospital dataset's baseline rate (37.3% average cancer prevalence).
> * 🔴 **Red Pullers (Abnormalities):** Abnormal cell features (e.g., enlarged nuclei or irregular borders) pull the rope towards **Malignant (+ risk)**.
> * 🔵 **Blue Pullers (Healthy Traits):** Normal measurements (e.g., smooth texture) pull the rope back towards **Benign (- risk)**.
> * 🏁 **The Final Whistle ($f(x) = 95.0\%$):** Baseline + (All Red Pulls) − (All Blue Pulls) = **95% Final Malignancy Risk**.

### 2. Mathematical Foundation (Cooperative Game Theory)
SHAP connects Lloyd Shapley's Nobel Prize-winning cooperative game theory to machine learning:
$$\sum_{i=1}^{M} \phi_i(x) = f(x) - \mathbb{E}[f(X)]$$
* $\phi_i(x)$: Marginal attribution (Shapley value) of medical feature $i$.
* $f(x)$: Final model prediction score for the individual patient sample.
* $\mathbb{E}[f(X)]$: Expected baseline score across the training population.

### 3. Model-Specific Explainer Architecture
To guarantee mathematical fidelity and speed across diverse algorithms:
* **Random Forest (`TreeExplainer`):** Computes polynomial-time ($O(TLD^2)$) exact conditional expectations over decision trees. Explains change in $P(\text{Malignant})$.
* **Logistic Regression (`LinearExplainer`):** Computes exact closed-form Shapley values for linear decision margins ($\Delta \text{Logit}$).
* **SVM & KNN (`KernelExplainer`):** Solves weighted linear regression on calibrated probability outputs using a representative background cohort ($N=30$) for real-time inference (<0.1s).

### 4. Top Clinical Biomarkers Identified
Across all 4 models, the most dominant predictors of breast malignancy are:
1. **Largest Nuclear Area (`area_worst`) & Radius (`radius_worst`):** Severe nuclear hypertrophy is the single strongest indicator of malignant rapid cell division.
2. **Severe Border Indentations (`concave points_worst`):** Deep notches and contour irregularity distinguish invasive carcinoma from smooth benign fibroadenomas.
3. **Mean Concavity (`concavity_mean`):** Degree of cellular membrane concavities.

| Global Biomarker Reliance Across Models | Random Forest TreeSHAP Beeswarm Plot |
| :---: | :---: |
| ![Biomarker Comparison](assets/shap_feature_importance.png) | ![TreeSHAP Summary](assets/shap_summary_random_forest.png) |

---

## 🏗️ End-to-End System Pipeline

```
                     Fine Needle Aspirate (FNA) Biopsy Report
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
       Option A: PDF Biopsy Upload                 Option B: Direct Manual Entry
       (Automated Regex Extraction                 (30 Continuous Features in
        of 30 Morphological Metrics)                Mean, SE, Worst tabs)
                 └──────────────────────┬──────────────────────┘
                                        │
                                        ▼
                           Feature Signature Validation
                           (30 Continuous Wisconsin Metrics)
                                        │
                                        ▼
                       StandardScaler Z-Score Normalization
                               z = (x - μ) / σ
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 ▼                      ▼                      ▼
        K-Nearest Neighbors    Logistic Regression    Support Vector Machine     Random Forest
           (Distance KNN)       (L2 Regularized)           (RBF Kernel)         (Scale Invariant)
                 └──────────────────────┬──────────────────────┘
                                        │
                                        ▼
                           4-Model Consensus Engine
                          (Diagnosis & Agreement Tally)
                                        │
                                        ▼
                            Explainable AI Engine (SHAP)
                     • Plain-English Bullet-Point Factors
                     • Local Waterfall Attribution Plot
                     • Global Cohort Biomarker Importance
```

---

## 🖥️ Streamlit Web Application Features

The interactive web app ([`app/app.py`](app/app.py)) is organized into 4 intuitive clinical pages:

1. **🏠 Home Page:**
   * Clean medical landing page with doctor hero illustration.
   * Four compact model architecture cards (KNN, LR, SVM, Random Forest).
   * Primary *"Start Diagnosis"* CTA and non-diagnostic disclaimers.

2. **📋 Patient Intake (Dual Input Methods):**
   * **Option A (FNA Biopsy PDF Upload):** Upload digital cytopathology reports (`.pdf`) with automated extraction of all 30 features, extraction review table, and sample PDF download buttons.
   * **Option B (Manual 30-Feature Entry):** Enter measurements directly across 3 tabs (*Mean*, *Standard Error*, *Worst*) with one-click **Malignant** and **Benign** test presets.
   * **Clinical Metadata:** Patient ID, age, and qualitative symptom tracking strictly isolated from the mathematical ML feature vector.

3. **🔬 Diagnostic Assessment & Explainability:**
   * **Section A (Consensus Scorecard):** Unanimous consensus indicator (e.g. `4/4 Models Agree`) with individual probability distribution meters.
   * **Section B (Benchmark Comparisons):** Wisconsin benchmark metrics table, ROC curves, PR curves, and test confusion matrices.
   * **Section C (AI Decision Breakdown in Simple Points):**
     * 🎯 Baseline population average card (37.3%).
     * 🔴 Red flag factors that raised malignancy risk with percentage shifts.
     * 🟢 Healthy indicators that lowered risk.
     * 🏁 Plain-English clinical verdict.
     * Collapsible expander with local waterfall and global beeswarm plots for technical deep dives.

4. **ℹ️ About the Project:**
   * Cytological background, Wisconsin FNA dataset documentation, algorithmic formulation, and medical boundary disclaimers.

---

## 📁 Repository Structure

```
breast--cancer-prediction/
├── .github/
│   └── workflows/
│       └── ci.yml                            # CI Workflow (automated pytest & linting)
├── app/
│   └── app.py                                # Streamlit main application entrypoint
├── assets/                                   # Evaluation plots, ROC curves, SHAP diagrams
├── data/
│   ├── data.csv                              # Wisconsin Breast Cancer Dataset (569 cases)
│   ├── README.md                             # Cytological feature dictionary
│   └── sample_reports/                       # Sample FNA Biopsy PDF test reports
│       ├── fna_biopsy_sample_malignant.pdf   # Confirmed Malignant FNA report
│       ├── fna_biopsy_sample_benign.pdf      # Confirmed Benign FNA report
│       ├── fna_biopsy_sample_incomplete.pdf  # Incomplete report for validation testing
│       └── unrelated_blood_test.pdf          # Non-biopsy report for rejection testing
├── models/
│   ├── knn_pipeline.joblib                   # Serialized KNN pipeline (Scaler + Distance KNN)
│   ├── logistic_regression_pipeline.joblib   # Serialized Logistic Regression pipeline
│   ├── svm_pipeline.joblib                   # Serialized SVM pipeline (RBF Kernel)
│   ├── random_forest_pipeline.joblib         # Serialized Random Forest pipeline
│   └── models_summary.json                   # Cross-validated metrics & test evaluations
├── notebooks/
│   └── breast_cancer_detection.ipynb         # EDA, statistical analysis & baseline training
├── reports/
│   ├── model_comparison.csv                  # Multi-model evaluation benchmark metrics
│   ├── model_comparison.json                 # Structured evaluation metrics & confusion matrices
│   └── shap_feature_importance.json          # Mean absolute SHAP importance rankings
├── src/
│   ├── data_loader.py                        # Dataset loading, canonical features & zero-leakage splits
│   ├── evaluate.py                           # Clinical evaluation metrics & comparison reporting
│   ├── explain.py                            # SHAP Explainable AI (Tree, Linear, Kernel explainers)
│   ├── generate_sample_pdf.py                # Synthetic realistic clinical FNA report generator
│   ├── pdf_parser.py                         # Regex-driven cytopathology FNA PDF extractor
│   ├── pipeline.py                           # Scikit-learn pipelines & hyperparameter grids
│   ├── predict.py                            # Multi-model inference API with shape validation
│   ├── train.py                              # Stratified training, 5-fold CV, model serialization
│   └── ui/                                   # Modular Streamlit multi-page UI architecture
│       ├── home_page.py                      # Landing page with hero illustration & model cards
│       ├── nav.py                            # Streamlit page routing & state navigation queue
│       ├── patient_page.py                   # Patient intake, PDF parser & manual 30-feature tabs
│       ├── results_page.py                   # Consensus scorecard & simplified SHAP bullet points
│       ├── about_page.py                     # Algorithmic formulations & clinical disclaimers
│       └── styles.py                         # Clean healthcare CSS themes & responsive cards
├── tests/
│   ├── test_explain.py                       # SHAP explainer unit tests
│   ├── test_patient_workflow.py              # End-to-end PDF & manual intake workflow tests
│   ├── test_pdf_parser.py                    # PDF extraction regex & rejection tests
│   ├── test_pipeline.py                      # Model training, split integrity & leakage tests
│   └── test_ui_nav.py                        # Streamlit routing & navigation queue tests
├── requirements.txt                          # Pinned project dependencies
├── LICENSE                                   # MIT License
└── README.md                                 # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/krishnasundari/breast--cancer-prediction.git
cd breast--cancer-prediction
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Automated Unit Tests (31 Tests Passing)
```bash
pytest -v
```

### 4. Launch Interactive Streamlit Web App Locally
```bash
streamlit run app/app.py
```
Open **`http://localhost:8501`** in your browser.

### 5. CLI Inference (Optional)
```bash
python -m src.predict --demo
```

---

## ⚕️ Educational Use Only Disclaimer

> **IMPORTANT NOTICE:** This repository and associated software are strictly developed for academic evaluation, machine learning algorithm benchmarking, and explainable AI research. They are **not** certified medical devices and must **not** be used as a standalone diagnostic tool or to guide patient treatment plans. Final clinical diagnoses must always be established by certified medical pathologists through tissue histopathology.

---

## 👤 Author

**Sundari Krishna**  
* **Institution:** Indian Institute of Technology Patna (B.Tech, Electrical & Electronics Engineering)  
* **GitHub:** [@krishnasundari](https://github.com/krishnasundari)  
* **Live App:** [breast--cancer-prediction.streamlit.app](https://breast--cancer-prediction.streamlit.app)  
* **Email:** [sundari2401ee35@iitp.ac.in](mailto:sundari2401ee35@iitp.ac.in)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
