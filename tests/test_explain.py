"""
test_explain.py
===============
Unit test suite for Phase 2: Explainable AI with SHAP.
Tests explainer initialization across all 4 models, local instance explanations,
feature ordering invariance, global feature importance generation, and error handling.
"""

import os
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.data_loader import get_train_test_split, FEATURE_NAMES
from src.predict import load_trained_pipeline
from src.explain import (
    get_model_explainer,
    explain_instance,
    plot_local_waterfall,
    compute_global_feature_importance,
    get_training_background
)

# Reference samples
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


def test_explainer_initialization_all_models():
    """Verify appropriate explainer class is assigned to each of the 4 models."""
    X_train, _, _, _ = get_train_test_split(random_state=42, stratify=True)

    rf_pipe = load_trained_pipeline("random_forest")["pipeline"]
    rf_explainer = get_model_explainer("random_forest", rf_pipe, X_train)
    assert rf_explainer["explainer_type"] == "TreeExplainer"
    assert isinstance(rf_explainer["explainer"], shap.TreeExplainer)

    lr_pipe = load_trained_pipeline("logistic_regression")["pipeline"]
    lr_explainer = get_model_explainer("logistic_regression", lr_pipe, X_train)
    assert lr_explainer["explainer_type"] == "LinearExplainer"
    assert isinstance(lr_explainer["explainer"], shap.LinearExplainer)

    svm_pipe = load_trained_pipeline("svm")["pipeline"]
    svm_explainer = get_model_explainer("svm", svm_pipe, X_train)
    assert svm_explainer["explainer_type"] == "KernelExplainer"
    assert isinstance(svm_explainer["explainer"], shap.KernelExplainer)

    knn_pipe = load_trained_pipeline("knn")["pipeline"]
    knn_explainer = get_model_explainer("knn", knn_pipe, X_train)
    assert knn_explainer["explainer_type"] == "KernelExplainer"
    assert isinstance(knn_explainer["explainer"], shap.KernelExplainer)


def test_unsupported_model_raises_value_error():
    """Verify unsupported model identifier raises descriptive ValueError."""
    rf_pipe = load_trained_pipeline("random_forest")["pipeline"]
    with pytest.raises(ValueError, match="Unsupported model architecture"):
        get_model_explainer("neural_network", rf_pipe)


def test_local_explanation_structure_and_types():
    """Verify explain_instance outputs correct structure, Malignant class focus, and types."""
    # Test Probability-based model (Random Forest)
    exp_rf = explain_instance(MALIGNANT_SAMPLE, model_name="random_forest")

    assert exp_rf["model_name"] == "random_forest"
    assert exp_rf["explainer_type"] == "TreeExplainer"
    assert exp_rf["diagnosis"] == "Malignant"
    assert exp_rf["prediction_code"] == 1
    assert exp_rf["score_type"] == "probability"
    assert "Probability" in exp_rf["unit_label"]
    assert 0.0 <= exp_rf["probability_malignant"] <= 1.0
    assert "score_name" in exp_rf
    assert isinstance(exp_rf["shap_explanation"], shap.Explanation)
    assert len(exp_rf["shap_explanation"].values) == len(FEATURE_NAMES)
    assert exp_rf["shap_explanation"].feature_names == FEATURE_NAMES

    # Ensure contributions are partitioned and ranked
    assert len(exp_rf["top_positive_features"]) > 0
    for item in exp_rf["top_positive_features"]:
        assert item["shap_value"] > 0
        assert "feature" in item
        assert "direction" in item

    # Test Log-Odds margin model (Logistic Regression)
    exp_lr = explain_instance(MALIGNANT_SAMPLE, model_name="logistic_regression")
    assert exp_lr["model_name"] == "logistic_regression"
    assert exp_lr["explainer_type"] == "LinearExplainer"
    assert exp_lr["score_type"] == "log_odds"
    assert "Log-Odds" in exp_lr["unit_label"]
    assert isinstance(exp_lr["model_score"], float)


def test_feature_ordering_invariance_in_explanations():
    """Ensure scrambled feature dictionary yields identical SHAP values."""
    keys = list(MALIGNANT_SAMPLE.keys())
    reversed_sample = {k: MALIGNANT_SAMPLE[k] for k in reversed(keys)}

    exp_orig = explain_instance(MALIGNANT_SAMPLE, model_name="random_forest")
    exp_rev = explain_instance(reversed_sample, model_name="random_forest")

    np.testing.assert_allclose(
        exp_orig["shap_explanation"].values,
        exp_rev["shap_explanation"].values,
        rtol=1e-5,
        err_msg="SHAP values differed when input dictionary keys were scrambled!"
    )


def test_invalid_input_handling():
    """Verify explain_instance rejects missing features, nulls, and wrong types."""
    # Missing columns
    incomplete = {"radius_mean": 15.0, "texture_mean": 18.0}
    with pytest.raises(ValueError, match="Sample missing required feature columns"):
        explain_instance(incomplete, model_name="random_forest")

    # Incomplete list
    with pytest.raises(ValueError, match="Expected 30 features"):
        explain_instance([1.0, 2.0], model_name="random_forest")

    # NaN in dataframe
    df_nan = pd.DataFrame([MALIGNANT_SAMPLE])
    df_nan.loc[0, "area_worst"] = np.nan
    with pytest.raises(ValueError, match="contains NaN/null feature values"):
        explain_instance(df_nan, model_name="random_forest")

    # Invalid type
    with pytest.raises(TypeError, match="sample must be a DataFrame, dict, or list"):
        explain_instance("invalid_input", model_name="random_forest")


def test_waterfall_plot_rendering():
    """Verify plot_local_waterfall generates a valid matplotlib Figure."""
    exp = explain_instance(BENIGN_SAMPLE, model_name="random_forest")
    fig = plot_local_waterfall(exp["shap_explanation"], max_display=8)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_global_feature_importance_computation():
    """Verify compute_global_feature_importance returns valid ranked dataframe."""
    df_imp = compute_global_feature_importance("random_forest", n_samples=25)

    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == 30
    assert list(df_imp.columns) == ["feature", "mean_abs_shap", "importance_rank", "model"]
    assert df_imp["mean_abs_shap"].is_monotonic_decreasing
    assert (df_imp["mean_abs_shap"] >= 0.0).all()
    assert df_imp.iloc[0]["importance_rank"] == 1


def test_background_data_never_uses_test_set():
    """Ensure training background data is strictly drawn from training partition."""
    X_train, X_test, _, _ = get_train_test_split(random_state=42, stratify=True)
    bg = get_training_background(X_train, n_samples=25)

    assert len(bg) == 25
    # None of the background sample rows should exist in held-out X_test
    overlap = bg.merge(X_test, how="inner")
    assert len(overlap) == 0, "Data leakage detected: Background sample contains instances from X_test!"
