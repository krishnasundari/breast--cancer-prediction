"""
test_patient_workflow.py
========================
End-to-end automated testing for both patient input workflows:
Workflow A: PDF Upload -> Extraction -> Review -> Validation -> 4 Models -> SHAP
Workflow B: Manual Entry -> Validation -> 4 Models -> SHAP
Also tests:
- Incomplete PDF extraction handling
- Incompatible medical report rejection
- Missing/invalid input detection
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data_loader import FEATURE_NAMES
from src.pdf_parser import parse_fna_pdf_report
from src.predict import predict, load_trained_pipeline
from src.explain import explain_instance


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "sample_reports")


def test_workflow_a_pdf_to_prediction_and_shap():
    """
    Workflow A:
    1. Upload FNA Biopsy PDF
    2. Extract all 30 features
    3. Review & Validate 30 features
    4. Run all 4 models (KNN, Logistic Regression, SVM, Random Forest)
    5. Generate SHAP explanation
    """
    pdf_path = os.path.join(SAMPLES_DIR, "fna_biopsy_sample_malignant.pdf")
    assert os.path.exists(pdf_path)

    # 1. Extraction
    with open(pdf_path, "rb") as f:
        parse_res = parse_fna_pdf_report(f)

    assert parse_res["success"] is True
    assert parse_res["is_complete"] is True
    assert parse_res["features_found_count"] == 30

    extracted_feats = parse_res["extracted_features"]

    # 2. Validation
    missing_feats = [f for f in FEATURE_NAMES if f not in extracted_feats or extracted_feats[f] is None]
    assert len(missing_feats) == 0

    # 3. Prediction across all 4 models
    models = ["knn", "logistic_regression", "svm", "random_forest"]
    for model_name in models:
        res = predict(extracted_feats, model_name_or_path=model_name)
        assert res["prediction_code"] in [0, 1]
        assert res["diagnosis"] in ["Benign", "Malignant"]
        assert 0.0 <= res["probability_malignant"] <= 1.0

    # 4. SHAP Explanation
    shap_res = explain_instance(extracted_feats, model_name="random_forest")
    assert len(shap_res["feature_contributions"]) == 30
    assert len(shap_res["shap_explanation"].values) == 30
    assert len(shap_res["top_positive_features"]) > 0 or len(shap_res["top_negative_features"]) > 0


def test_workflow_b_manual_entry_to_prediction_and_shap():
    """
    Workflow B:
    1. Enter measurements manually (30 features: 10 Mean, 10 SE, 10 Worst)
    2. Validate feature completeness
    3. Run all 4 models
    4. Generate SHAP explanation
    """
    # Authentic Benign case measurements
    manual_features = {
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

    # 1. Structure check: exactly 30 features
    assert len(manual_features) == 30
    for f in FEATURE_NAMES:
        assert f in manual_features

    # 2. Validation check
    missing = [f for f in FEATURE_NAMES if f not in manual_features or manual_features[f] is None]
    assert len(missing) == 0

    # 3. Model predictions
    for model_name in ["knn", "logistic_regression", "svm", "random_forest"]:
        pred_res = predict(manual_features, model_name_or_path=model_name)
        assert pred_res["prediction_code"] == 0  # Confirmed benign sample
        assert pred_res["diagnosis"] == "Benign"
        assert pred_res["probability_malignant"] < 0.5

    # 4. SHAP Explanation
    shap_res = explain_instance(manual_features, model_name="logistic_regression")
    assert len(shap_res["feature_contributions"]) == 30
    assert len(shap_res["shap_explanation"].values) == 30


def test_incomplete_pdf_handling_and_manual_correction():
    """
    Test incomplete PDF extraction:
    - Verifies parser identifies missing features
    - Verifies validation fails when incomplete
    - Verifies manually supplying missing values completes the feature set
    """
    incomplete_pdf_path = os.path.join(SAMPLES_DIR, "fna_biopsy_sample_incomplete.pdf")
    assert os.path.exists(incomplete_pdf_path)

    with open(incomplete_pdf_path, "rb") as f:
        parse_res = parse_fna_pdf_report(f)

    assert parse_res["success"] is True
    assert parse_res["is_complete"] is False
    assert len(parse_res["missing_features"]) > 0

    cur_feats = parse_res["extracted_features"].copy()

    # Pre-correction: missing features present
    missing = [f for f in FEATURE_NAMES if f not in cur_feats or cur_feats[f] is None]
    assert len(missing) > 0

    # User manually corrects by filling missing features
    for f in missing:
        cur_feats[f] = 0.05  # Fill valid placeholder float

    # Post-correction: validation now succeeds
    post_missing = [f for f in FEATURE_NAMES if f not in cur_feats or cur_feats[f] is None]
    assert len(post_missing) == 0
    assert len(cur_feats) == 30

    # Predictions succeed with corrected vector
    res = predict(cur_feats, model_name_or_path="random_forest")
    assert res["prediction_code"] in [0, 1]


def test_incompatible_pdf_rejection():
    """Verify arbitrary blood test / non-FNA reports are rejected cleanly."""
    cbc_path = os.path.join(SAMPLES_DIR, "unrelated_blood_test.pdf")
    assert os.path.exists(cbc_path)

    with open(cbc_path, "rb") as f:
        res = parse_fna_pdf_report(f)

    assert res["success"] is False
    assert "Incompatible medical report detected" in res["error"]
    assert res["is_complete"] is False


def test_invalid_and_missing_feature_validation():
    """Verify missing or None values are caught by validation logic."""
    incomplete_features = {
        'radius_mean': 15.0,
        'texture_mean': None,  # Invalid
        # Remaining 28 missing
    }
    missing = [f for f in FEATURE_NAMES if f not in incomplete_features or incomplete_features[f] is None]
    assert len(missing) == 29
    assert 'texture_mean' in missing
    assert 'area_mean' in missing
