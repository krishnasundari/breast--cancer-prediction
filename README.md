# 🎗️ Breast Cancer Detection using K-Nearest Neighbors (KNN)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-97.3%25-brightgreen)](https://github.com/krishnasundari/breast--cancer-prediction)
[![F1-Score](https://img.shields.io/badge/F1--Score-96.55%25-blueviolet)](https://github.com/krishnasundari/breast--cancer-prediction)
[![Streamlit](https://img.shields.io/badge/Streamlit-App%20Ready-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning pipeline for breast cancer diagnosis and malignancy detection built on the **Wisconsin Breast Cancer Diagnostic Dataset**. The system standardizes 30 nuclear morphological features from Fine Needle Aspirate (FNA) biopsies and deploys an optimized, distance-weighted **K-Nearest Neighbors (KNN)** classifier.

---

## 📌 Resume Alignment & Verification

This project verifies and substantiates the credentials submitted in my resume:

| Resume Bullet Point | Implementation in Repository | Verification & Benchmark |
| :--- | :--- | :--- |
| **Developed ML model for breast cancer classification using KNN** | Built modular scikit-learn pipeline combining `StandardScaler` and `KNeighborsClassifier` in [`src/pipeline.py`](src/pipeline.py). | Optimized with distance weighting ($w = 1/d$) and Euclidean distance. |
| **Achieved 97.3% classification accuracy on Wisconsin Diagnostic Dataset** | Evaluated on held-out test partition (114 samples) in [`src/evaluate.py`](src/evaluate.py) & [`notebooks/breast_cancer_detection.ipynb`](notebooks/breast_cancer_detection.ipynb). | **97.37% (97.3%) test accuracy** ($K=6$, random_state=2, 0 False Positives). |
| **Built an end-to-end ML pipeline (preprocessing, scaling, training, eval)** | Modular engineering architecture: [`data_loader.py`](src/data_loader.py), [`pipeline.py`](src/pipeline.py), [`train.py`](src/train.py), [`predict.py`](src/predict.py). | Zero data leakage: strict separation of train/test scaling statistics. |
| **Improved model reliability through precision, recall, and F1-score evaluation** | Clinically oriented risk analysis prioritizing the reduction of False Negatives (FN). | **100% Precision**, **93.3% Recall**, **96.55% F1-Score**, **0.99 ROC-AUC**. |

---

## 🏗️ System Architecture

```
                                    Fine Needle Aspirate (FNA) Biopsy
                                                  │
                                                  ▼
                                      Dataset (569 Patients)
                                    30 Morphological Features
                                                  │
                                                  ▼
                                 Train-Test Split (80% / 20%)
                                 ┌───────────────────────────┐
                                 │ Training Set (455 samples)│
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                              Fit StandardScaler (Compute μ, σ)
                                               │
                                               ▼
                             Z-Score Scaling: z = (x - μ) / σ
                                               │
                                               ▼
                         K-Nearest Neighbors Classifier (K=6)
                        Inverse Distance Weighting: w = 1 / d
                                               │
                                               ▼
                                 Evaluation on Test Partition
                                      (114 Unseen Samples)
                                               │
              ┌────────────────────────────────┴────────────────────────────────┐
              ▼                                                                 ▼
   Diagnostic Classification                                          Performance Metrics
   • Benign (0)   -> 69 / 69 Correct                                  • Accuracy : 97.37% (97.3%)
   • Malignant (1) -> 42 / 45 Correct                                 • Precision: 100.0%
   • False Positives: 0 (No false alarms)                             • Recall   : 93.33%
   • False Negatives: 3 (Minimized)                                   • F1-Score : 96.55%
```

---

## 📊 Benchmark Results & Clinical Metrics

### 1. Classification Metrics Summary
| Metric | Benign Class (0) | Malignant Class (1) | Overall / Weighted Average |
| :--- | :---: | :---: | :---: |
| **Precision** | **95.83%** | **100.00%** | **97.48%** |
| **Recall / Sensitivity** | **100.00%** | **93.33%** | **97.37%** |
| **F1-Score** | **97.87%** | **96.55%** | **97.35%** |
| **Support** | 69 samples | 45 samples | 114 test samples |
| **Overall Accuracy** | — | — | **97.37% (97.3%)** |

### 2. Confusion Matrix Analysis
```
                       Predicted Benign (0)      Predicted Malignant (1)
 Actual Benign (0)             69                           0           (100% Specificity)
 Actual Malignant (1)           3                          42           (93.3% Sensitivity)
```
- **Zero False Positives ($FP = 0$):** Not a single healthy patient was falsely diagnosed with cancer, preventing biopsy trauma and clinical stress.
- **Low False Negatives ($FN = 3$):** 42 out of 45 malignant cases were caught immediately by cell nuclear geometry.

---

## 📈 Visualizations

| Hyperparameter Optimization (Elbow Plot) | Confusion Matrix Heatmap |
| :---: | :---: |
| ![Elbow Plot](assets/error_rate_k.png) | ![Confusion Matrix](assets/confusion_matrix.png) |

| Receiver Operating Characteristic (ROC) | Feature Correlation Matrix |
| :---: | :---: |
| ![ROC Curve](assets/roc_curve.png) | ![Correlation Heatmap](assets/correlation_heatmap.png) |

---

## 📁 Repository Structure

```
breast--cancer-prediction/
├── .github/
│   └── workflows/
│       └── ci.yml                            # Continuous Integration (pytest + lint)
├── data/
│   ├── data.csv                              # Wisconsin Breast Cancer Dataset (569 rows)
│   └── README.md                             # Dataset dictionary & feature descriptions
├── notebooks/
│   └── breast_cancer_detection.ipynb         # Interactive Jupyter Notebook with complete EDA
├── src/
│   ├── __init__.py
│   ├── data_loader.py                        # Dataset loading & zero-leakage splitting
│   ├── pipeline.py                           # Scikit-learn Pipeline (Scaler + KNN)
│   ├── train.py                              # Model training, CV, and hyperparameter tuning
│   ├── evaluate.py                           # Clinical evaluation metrics & reporting
│   └── predict.py                            # CLI & programmatic inference API
├── models/
│   └── knn_pipeline.joblib                   # Serialized production model pipeline
├── app/
│   └── app.py                                # Streamlit web application for real-time diagnosis
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                      # Pytest unit tests (verifies 97.3% accuracy)
├── assets/                                   # Evaluation plots and diagnostic charts
├── .gitignore
├── requirements.txt                          # Pinned dependencies
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

### 3. Run Model Training & Evaluation
```bash
python -m src.train
```

To run full hyperparameter grid search:
```bash
python -m src.train --tune
```

### 4. Run Demonstration CLI Prediction
```bash
python -m src.predict --demo
```

### 5. Launch Interactive Streamlit Web App
```bash
streamlit run app/app.py
```

### 6. Run Automated Unit Tests
```bash
pytest -v tests/test_pipeline.py
```

---

## 🔬 Key Engineering Highlights

1. **Zero Data Leakage:**
   Novice machine learning workflows frequently call `scaler.fit_transform(X_test)` or normalize data prior to splitting. This pipeline strictly fits the scaler solely on the training partition ($X_{train}$), keeping $X_{test}$ completely out-of-sample.

2. **Distance-Weighted Neighborhoods:**
   Standard KNN gives every neighbor 1 uniform vote. In this project, neighbors are weighted by inverse Euclidean distance ($w_i = 1 / d(x, x_i)$), ensuring that samples with near-identical morphological dimensions exert greater diagnostic influence.

3. **Rigorous Clinical Evaluation:**
   Prioritizes high sensitivity and recall to minimize life-threatening False Negatives while maintaining an outstanding 96.55% F1-score and 0.99 ROC-AUC.

---

## 👤 Author
**Sundari Krishna**  
- **Institution:** Indian Institute of Technology Patna (B.Tech, EEE)  
- **GitHub:** [@krishnasundari](https://github.com/krishnasundari)  
- **Email:** [sundari2401ee35@iitp.ac.in](mailto:sundari2401ee35@iitp.ac.in)

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
