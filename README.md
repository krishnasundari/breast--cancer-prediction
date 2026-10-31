# Breast Cancer Detection & Multi-Model Classification System

An end-to-end machine learning system that analyzes cell nuclear morphology from biopsy data to classify breast tumors as benign or malignant, accompanied by SHAP-based feature explanations.

[![Live Demo](https://img.shields.io/badge/Streamlit%20Cloud-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://breast--cancer-prediction.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-31%20Passed-brightgreen?logo=pytest&logoColor=white)](https://github.com/krishnasundari/breast--cancer-prediction)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live Web Application:** [https://breast--cancer-prediction.streamlit.app](https://breast--cancer-prediction.streamlit.app)

---

## 1. Project Overview

My project is a **Breast Cancer Detection and Classification System** that uses Machine Learning to classify breast tumors as either Benign (non-cancerous) or Malignant (cancerous).

The project uses the Wisconsin Breast Cancer Diagnostic Dataset, which contains 569 samples and 30 numerical features describing the characteristics of cell nuclei.

I implemented and compared four Machine Learning algorithms:

* K-Nearest Neighbors (KNN)
* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest

The application allows users to enter the 30 feature values manually or upload a supported PDF report. It then predicts the tumor category using the trained models.

I also integrated **SHAP (Explainable AI)** to explain which features influenced each model's prediction. The application is built using Python and Streamlit.

---

## 2. Key Features

* **Four Supervised Classifiers:** Implements K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Machine (SVM), and Random Forest.
* **Automated PDF Biopsy Extraction:** Parses digital cytopathology reports (`.pdf`) using regular expressions to extract and validate all 30 required Wisconsin morphological features.
* **Manual Feature Entry:** Provides an interactive interface organized into Mean, Standard Error (SE), and Worst tabs for direct input or adjustment.
* **Multi-Model Consensus:** Displays predictions, malignancy probabilities, and consensus agreement across all four models simultaneously.
* **SHAP Explainability:** Highlights the exact cell measurements that increased or decreased the predicted malignancy risk for each patient.
* **Streamlit Web Interface:** A multi-page web application featuring patient intake, prediction scorecards, benchmark graphs, and clinical context.

---

## 3. Technology Stack

* **Python:** Core programming language.
* **NumPy & Pandas:** Data manipulation, vector operations, and tabular data management.
* **Scikit-learn:** Data preprocessing, pipeline construction, model training, cross-validation, and metric evaluation.
* **SHAP:** Explainable AI library providing local feature attribution and global importance.
* **Streamlit:** Interactive web interface and application deployment.
* **PyPDF:** PDF text extraction for digital biopsy reports.

---

## 4. System Workflow

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│     Option A: PDF Biopsy Upload │       │  Option B: Manual Feature Entry │
│   (Regex extracts 30 measurements)│      │    (Mean, SE, Worst tabs)       │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │    Feature Validation     │
                        │ (Verify 30 numeric inputs)│
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │      Data Scaling         │
                        │  (StandardScaler Z-score) │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │      4 ML Classifiers     │
                        │  (KNN, LR, SVM, Random F.)│
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Consensus Prediction    │
                        │ (Diagnosis & Probabilities)│
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │     SHAP Explanation      │
                        │  (Top risk-shifting traits)│
                        └───────────────────────────┘
```

### Step-by-Step Description:

1. **Input Data:** The user can either upload an FNA biopsy PDF report or enter measurements manually.
2. **Feature Validation:** The system confirms that all 30 expected numerical features are present and within valid physical ranges.
3. **Data Preprocessing & Scaling:** Features are standardized using `StandardScaler` so that variables with large ranges (such as area) do not dominate variables with small ranges (such as smoothness). Random Forest uses raw features as tree algorithms are scale-invariant.
4. **Model Inference:** The input vector is evaluated by all four trained pipelines to generate diagnostic labels and probabilities.
5. **Consensus Prediction:** Results are aggregated to check whether all models agree on the diagnosis.
6. **Explainability:** SHAP breaks down the prediction to show which specific cellular traits pushed the score toward benign or malignant.

---

## 5. Model Performance

All models were evaluated on the **Wisconsin Breast Cancer Diagnostic Dataset (WDBC)** using a stratified 80/20 train/test split (455 training samples, 114 held-out test samples: 72 Benign, 42 Malignant). Five-fold cross-validation was performed strictly on the training partition.

### Held-Out Test Set Results (114 Samples)

| Model Architecture | Preprocessing | 5-Fold CV Accuracy | Test Accuracy | Precision (Malignant) | Recall (Malignant) | F1-Score (Malignant) | Test ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | StandardScaler + RBF Kernel | 97.14% | **97.37%** | 100.00% | 92.86% | 96.30% | 0.9947 |
| **Random Forest** | Raw Features (Scale-Invariant) | 96.48% | **97.37%** | 100.00% | 92.86% | 96.30% | 0.9929 |
| **Logistic Regression** | StandardScaler + L2 Regularization | 97.36% | **96.49%** | 97.50% | 92.86% | 95.12% | 0.9960 |
| **K-Nearest Neighbors (KNN)** | StandardScaler + Distance Weighting | 97.14% | **94.74%** | 97.37% | 88.10% | 92.50% | 0.9828 |

### Understanding the Evaluation Metrics:

* **Accuracy:** The overall percentage of correct predictions out of all test cases.
* **Precision:** When the model predicts malignant, how often it is correct. High precision means fewer false alarms (False Positives).
* **Recall (Sensitivity):** The proportion of actual malignant tumors correctly identified. High recall is critical in cancer detection to minimize missed cases (False Negatives).
* **F1-Score:** The harmonic mean of precision and recall, balancing both concerns.
* **ROC-AUC:** Measures how effectively the model separates benign from malignant across all decision thresholds (closer to 1.0 is better).

---

## 6. Explainable AI (SHAP)

Machine learning models, particularly ensemble trees and kernel methods, often act as "black boxes." In medical contexts, physicians need to understand the reasoning behind a prediction before trusting it.

### What SHAP Does

SHAP (SHapley Additive exPlanations) is a game-theoretic approach that assigns each feature an attribution value for a given prediction:

1. **Baseline Reference:** Every prediction starts from a baseline rate, which is the average malignancy proportion across the training population (approximately 37.3%).
2. **Feature Attribution:** Each measurement pushes the prediction from this baseline toward either **Malignant** (positive attribution, elevating risk) or **Benign** (negative attribution, reducing risk).
3. **Local Explanation:** For an individual patient, the user can see the top features that drove the decision. For instance, in a malignant sample, severe cell enlargement (`area_worst`) and deep perimeter indents (`concave points_worst`) significantly push the prediction toward malignant.
4. **Global Biomarkers:** Across the entire dataset, the three most important features identified are **Worst Area**, **Worst Concave Points**, and **Worst Radius**.

---

## 7. Project Structure

```
breast--cancer-prediction/
├── app/
│   └── app.py                      # Main Streamlit web application entrypoint
├── assets/                         # Visualizations, diagrams, and benchmark plots
├── data/
│   ├── data.csv                    # Wisconsin Breast Cancer Dataset (569 rows)
│   ├── README.md                   # Feature definitions and cytological descriptions
│   └── sample_reports/             # Sample FNA biopsy PDF files for testing
├── models/
│   ├── knn_pipeline.joblib         # Serialized KNN pipeline
│   ├── logistic_regression_pipeline.joblib  # Serialized Logistic Regression pipeline
│   ├── svm_pipeline.joblib         # Serialized SVM pipeline
│   ├── random_forest_pipeline.joblib # Serialized Random Forest pipeline
│   └── models_summary.json         # Evaluation metrics summary for all models
├── notebooks/
│   └── breast_cancer_detection.ipynb # Exploratory data analysis and model prototyping
├── reports/
│   ├── model_comparison.csv        # Tabular evaluation metrics for all classifiers
│   └── model_comparison.json       # Structured benchmark metrics and confusion matrices
├── src/
│   ├── data_loader.py              # Dataset loading and train/test splitting
│   ├── evaluate.py                 # Evaluation metrics calculation and visualization
│   ├── explain.py                  # SHAP explainer implementations
│   ├── pdf_parser.py               # PDF text and feature extraction logic
│   ├── pipeline.py                 # Pipeline definitions with scaling and models
│   ├── predict.py                  # Single and batch prediction utilities
│   ├── train.py                    # Model training and artifact serialization
│   └── ui/                         # Streamlit user interface components
├── tests/
│   ├── test_explain.py             # Unit tests for SHAP explainer outputs
│   ├── test_patient_workflow.py    # Tests for PDF upload and manual entry workflows
│   ├── test_pdf_parser.py          # Tests for regex extraction and file validation
│   ├── test_pipeline.py            # Tests for model training, leakage, and evaluation
│   └── test_ui_nav.py              # Tests for UI navigation state
├── requirements.txt                # Pinned project dependencies
├── LICENSE                         # MIT License
└── README.md                       # Project documentation
```

---

## 8. Installation and Execution

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

# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
pytest -v
```
All 31 unit and integration tests should pass.

### 4. Run the Streamlit Application
```bash
streamlit run app/app.py
```
Open your browser and navigate to `http://localhost:8501`.

### 5. Retrain Models (Optional)
```bash
python -m src.train
```

---

## 9. Limitations & Disclaimer

* **Academic Research Scope:** This project was developed for educational and machine learning benchmarking purposes. It is **not** a certified medical device and is not approved by regulatory bodies (such as the FDA, CE, or CDSCO).
* **Not for Standalone Diagnosis:** The system must not be used as a standalone diagnostic tool or to determine patient treatment plans. Formal cancer diagnosis requires histological biopsy confirmation by certified pathologists.
* **Dataset Scope:** The models are trained on the Wisconsin Breast Cancer Diagnostic Dataset (569 cases). While widely benchmarked in academic literature, real-world deployment requires training on larger, multi-institutional, and ethnically diverse cohorts.
* **Feature Collinearity:** Physical dimensions of cell nuclei (e.g., radius, perimeter, and area) have high natural mathematical correlation ($r > 0.9$). SHAP feature attributions are distributed across these correlated measurements.

---

## 10. Author

* **Author:** Sundari Krishna
* **Program:** B.Tech in Electrical and Electronics Engineering (EEE)
* **Institution:** Indian Institute of Technology Patna
* **GitHub:** [@krishnasundari](https://github.com/krishnasundari)
* **Email:** [sundari2401ee35@iitp.ac.in](mailto:sundari2401ee35@iitp.ac.in)
* **Live App:** [breast--cancer-prediction.streamlit.app](https://breast--cancer-prediction.streamlit.app)

---

## 11. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
