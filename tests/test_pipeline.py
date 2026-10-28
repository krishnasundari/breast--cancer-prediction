"""
test_pipeline.py
================
Unit tests for the Breast Cancer Detection Machine Learning Pipeline.
Tests data integrity, data leakage prevention, model convergence, and accuracy.
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from src.data_loader import load_raw_dataset, prepare_features_and_target, get_train_test_split
from src.pipeline import create_knn_pipeline
from src.evaluate import evaluate_model
from src.predict import predict, load_trained_pipeline


def test_data_loader_shape_and_types():
    """Verify raw dataset has 569 instances and 30 continuous features + diagnosis."""
    df = load_raw_dataset()
    assert df.shape[0] == 569, f"Expected 569 rows, got {df.shape[0]}"
    assert "diagnosis" in df.columns, "Missing 'diagnosis' column"
    assert df["diagnosis"].isin(["M", "B"]).all(), "Diagnosis must contain only 'M' and 'B'"
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected missing values"

    X, y = prepare_features_and_target(df)
    assert X.shape[1] == 30, f"Expected 30 features, got {X.shape[1]}"
    assert set(y.unique()) == {0, 1}, "Target y must be binary {0, 1}"


def test_pipeline_no_data_leakage():
    """
    Ensure feature scaling parameters (mean, variance) are learned solely
    from training data, preventing test set contamination.
    """
    X_train, X_test, y_train, y_test = get_train_test_split(random_state=2)

    pipeline = create_knn_pipeline(n_neighbors=6)
    pipeline.fit(X_train, y_train)

    scaler: StandardScaler = pipeline.named_steps["scaler"]

    # Verify scaler mean matches X_train mean, NOT X_test or combined
    np.testing.assert_allclose(
        scaler.mean_,
        X_train.mean().values,
        rtol=1e-5,
        err_msg="StandardScaler was not fitted strictly on X_train!"
    )


def test_accuracy_benchmark():
    """
    Ensure the KNN pipeline achieves at least 97% classification accuracy
    on the Wisconsin test partition, substantiating resume credentials.
    """
    X_train, X_test, y_train, y_test = get_train_test_split(random_state=2)

    pipeline = create_knn_pipeline(n_neighbors=6, weights="distance")
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)
    accuracy = metrics["accuracy"]

    # Resume claim: 97.3% classification accuracy
    assert accuracy >= 0.970, f"Accuracy {accuracy*100:.2f}% did not meet 97.0% threshold"
    assert metrics["precision"] >= 0.95, "Precision on malignant class should be >= 95%"
    assert metrics["recall_sensitivity"] >= 0.90, "Recall on malignant class should be >= 90%"


def test_predict_function_structure():
    """Verify predict() returns valid dictionary with probabilities and diagnosis."""
    sample = {
        'radius_mean': 17.99, 'texture_mean': 10.38, 'perimeter_mean': 122.8, 'area_mean': 1001.0,
        'smoothness_mean': 0.1184, 'compactness_mean': 0.2776, 'concavity_mean': 0.3001,
        'concave points_mean': 0.1471, 'symmetry_mean': 0.2419, 'fractal_dimension_mean': 0.07871,
        'radius_se': 1.095, 'texture_se': 0.9053, 'perimeter_se': 8.589, 'area_se': 153.4,
        'smoothness_se': 0.006399, 'compactness_se': 0.04904, 'concavity_se': 0.05373,
        'concave points_se': 0.01587, 'symmetry_se': 0.03003, 'fractal_dimension_se': 0.006193,
        'radius_worst': 25.38, 'texture_worst': 17.33, 'perimeter_worst': 184.6, 'area_worst': 2019.0,
        'smoothness_worst': 0.1622, 'compactness_worst': 0.6656, 'concavity_worst': 0.7119,
        'concave points_worst': 0.2654, 'symmetry_worst': 0.4601, 'fractal_dimension_worst': 0.1189
    }
    result = predict(sample)

    assert "diagnosis" in result
    assert result["diagnosis"] in ["Benign", "Malignant"]
    assert 0.0 <= result["probability_malignant"] <= 1.0
    assert 0.0 <= result["probability_benign"] <= 1.0
    assert "clinical_risk_assessment" in result
