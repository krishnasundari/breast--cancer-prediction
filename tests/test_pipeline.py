"""
test_pipeline.py
================
Comprehensive unit test suite for the Breast Cancer Detection & Classification System.
Verifies data loading, leakage prevention, 4-model training, saving/loading,
evaluation metrics, robust inference, missing feature handling, and feature ordering invariance.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import (
    load_raw_dataset,
    prepare_features_and_target,
    get_train_test_split,
    validate_dataset,
    FEATURE_NAMES
)
from src.pipeline import (
    create_knn_pipeline,
    create_model_pipeline,
    get_all_model_pipelines
)
from src.evaluate import evaluate_model, cross_validate_model
from src.predict import predict, load_trained_pipeline
import joblib


# Canonical sample dictionaries for inference testing
MALIGNANT_SAMPLE = {
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

BENIGN_SAMPLE = {
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


def test_data_loader_shape_and_types():
    """Verify raw dataset has 569 instances, 30 continuous features, and binary target."""
    df = load_raw_dataset()
    assert df.shape[0] == 569, f"Expected 569 rows, got {df.shape[0]}"
    assert "diagnosis" in df.columns, "Missing 'diagnosis' column"
    assert df["diagnosis"].isin(["M", "B"]).all(), "Diagnosis must contain only 'M' and 'B'"
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected missing values"

    X, y = prepare_features_and_target(df)
    assert X.shape[1] == 30, f"Expected 30 features, got {X.shape[1]}"
    assert list(X.columns) == FEATURE_NAMES, "Feature columns must match canonical ordering"
    assert set(y.unique()) == {0, 1}, "Target y must be binary {0, 1}"


def test_validate_dataset_report():
    """Verify validate_dataset returns correct verification dictionary."""
    summary = validate_dataset()
    assert summary["num_samples"] == 569
    assert summary["num_features"] == 30
    assert summary["missing_values"] == 0
    assert summary["duplicates"] == 0
    assert summary["class_distribution"]["Benign (0)"] == 357
    assert summary["class_distribution"]["Malignant (1)"] == 212


def test_pipeline_no_data_leakage():
    """
    Ensure feature scaling parameters (mean, variance) are learned solely
    from training data, preventing test set contamination.
    """
    X_train, X_test, y_train, y_test = get_train_test_split(random_state=42, stratify=True)

    pipeline = create_knn_pipeline(n_neighbors=6)
    pipeline.fit(X_train, y_train)

    scaler: StandardScaler = pipeline.named_steps["scaler"]

    np.testing.assert_allclose(
        scaler.mean_,
        X_train.mean().values,
        rtol=1e-5,
        err_msg="StandardScaler was not fitted strictly on X_train!"
    )


def test_scaling_specification_per_model():
    """
    Verify scaling rule: StandardScaler is applied to KNN, Logistic Regression, and SVM,
    but omitted for Random Forest.
    """
    pipelines = get_all_model_pipelines()
    assert "scaler" in pipelines["knn"].named_steps, "KNN must include StandardScaler"
    assert "scaler" in pipelines["logistic_regression"].named_steps, "Logistic Regression must include StandardScaler"
    assert "scaler" in pipelines["svm"].named_steps, "SVM must include StandardScaler"
    assert "scaler" not in pipelines["random_forest"].named_steps, "Random Forest should NOT include StandardScaler"


def test_train_all_four_models():
    """Train all four classification pipelines and assert predictions are valid."""
    X_train, X_test, y_train, y_test = get_train_test_split(random_state=42, stratify=True)
    pipelines = get_all_model_pipelines()

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        assert len(preds) == len(X_test), f"{name}: Prediction count mismatch"
        assert set(np.unique(preds)).issubset({0, 1}), f"{name}: Predictions must be binary 0 or 1"

        probs = pipe.predict_proba(X_test)
        assert probs.shape == (len(X_test), 2), f"{name}: Probability matrix shape mismatch"
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5, err_msg=f"{name}: Probabilities do not sum to 1")


def test_evaluation_metrics_generation():
    """Verify evaluation and 5-fold cross-validation output comprehensive metrics."""
    X_train, X_test, y_train, y_test = get_train_test_split(random_state=42, stratify=True)
    pipe = create_model_pipeline("logistic_regression")
    pipe.fit(X_train, y_train)

    metrics = evaluate_model(pipe, X_test, y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall_sensitivity"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["false_negative_rate_malignant"] <= 1.0
    assert 0.0 <= metrics["false_discovery_rate"] <= 1.0
    assert "confusion_matrix" in metrics
    cm = metrics["confusion_matrix"]
    assert cm["true_positives"] + cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] == len(y_test)

    # Cross-validation
    cv_res = cross_validate_model(pipe, X_train, y_train, cv=5)
    assert 0.0 <= cv_res["cv_accuracy_mean"] <= 1.0
    assert 0.0 <= cv_res["cv_recall_mean"] <= 1.0


def test_save_and_load_pipelines():
    """Verify that all four pipelines can be serialized, deserialized, and yield identical predictions."""
    X_train, X_test, y_train, y_test = get_train_test_split(random_state=42, stratify=True)
    pipelines = get_all_model_pipelines()

    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, pipe in pipelines.items():
            pipe.fit(X_train, y_train)
            orig_preds = pipe.predict(X_test)

            tmp_path = os.path.join(tmp_dir, f"{name}.joblib")
            joblib.dump({"pipeline": pipe, "feature_names": FEATURE_NAMES, "model_name": name}, tmp_path)

            loaded_artifact = load_trained_pipeline(tmp_path)
            loaded_pipe = loaded_artifact["pipeline"]
            loaded_preds = loaded_pipe.predict(X_test)

            np.testing.assert_array_equal(orig_preds, loaded_preds, err_msg=f"{name}: Loaded pipeline output differed!")


def test_predict_function_all_four_models():
    """Verify predict() works across all 4 models and produces valid output schema."""
    for model_key in ["knn", "logistic_regression", "svm", "random_forest"]:
        res_m = predict(MALIGNANT_SAMPLE, model_name_or_path=model_key)
        assert res_m["prediction_code"] in [0, 1]
        assert res_m["diagnosis"] in ["Benign", "Malignant"]
        assert 0.0 <= res_m["probability_malignant"] <= 1.0
        assert 0.0 <= res_m["probability_benign"] <= 1.0
        assert "clinical_risk_assessment" in res_m

        res_b = predict(BENIGN_SAMPLE, model_name_or_path=model_key)
        assert res_b["prediction_code"] in [0, 1]
        assert res_b["diagnosis"] in ["Benign", "Malignant"]


def test_missing_and_invalid_features():
    """Verify predict() raises ValueError when required features are missing or NaN."""
    # Missing columns
    incomplete_sample = {'radius_mean': 15.0, 'texture_mean': 18.0}
    with pytest.raises(ValueError, match="missing .* required features"):
        predict(incomplete_sample, model_name_or_path="knn")

    # Incomplete list
    with pytest.raises(ValueError, match="Feature count mismatch"):
        predict([1.0, 2.0, 3.0], model_name_or_path="knn")

    # NaN in DataFrame
    df_nan = pd.DataFrame([MALIGNANT_SAMPLE])
    df_nan.loc[0, "radius_mean"] = np.nan
    with pytest.raises(ValueError, match="contains NaN/null values"):
        predict(df_nan, model_name_or_path="knn")


def test_feature_reordering_invariance():
    """Ensure predict() yields identical results regardless of dictionary or DataFrame column order."""
    reversed_sample = {k: MALIGNANT_SAMPLE[k] for k in reversed(list(MALIGNANT_SAMPLE.keys()))}

    res_orig = predict(MALIGNANT_SAMPLE, model_name_or_path="svm")
    res_rev = predict(reversed_sample, model_name_or_path="svm")

    assert res_orig["prediction_code"] == res_rev["prediction_code"]
    np.testing.assert_almost_equal(res_orig["probability_malignant"], res_rev["probability_malignant"], decimal=5)


def test_accuracy_benchmark():
    """
    Ensure the KNN pipeline achieves at least 97% classification accuracy
    on the Wisconsin test partition (random_state=2), substantiating resume credentials.
    """
    X_train, X_test, y_train, y_test = get_train_test_split(random_state=2, stratify=False)

    pipeline = create_knn_pipeline(n_neighbors=6, weights="distance")
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)
    accuracy = metrics["accuracy"]

    # Resume claim: 97.3% classification accuracy
    assert accuracy >= 0.970, f"Accuracy {accuracy*100:.2f}% did not meet 97.0% threshold"
    assert metrics["precision"] >= 0.95, "Precision on malignant class should be >= 95%"
    assert metrics["recall_sensitivity"] >= 0.90, "Recall on malignant class should be >= 90%"
