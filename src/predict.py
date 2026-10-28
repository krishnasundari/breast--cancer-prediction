"""
predict.py
==========
Inference module for serving predictions from the trained KNN pipeline.
Provides both programmatic API and Command Line Interface (CLI).
"""

import argparse
import os
from typing import Dict, Any, Union, List
import joblib
import numpy as np
import pandas as pd


def load_trained_pipeline(model_path: str = None) -> Dict[str, Any]:
    """Loads the serialized model payload from disk."""
    if model_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "knn_pipeline.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python -m src.train' first to generate the artifact."
        )

    return joblib.load(model_path)


def predict(
    sample: Union[pd.DataFrame, Dict[str, float], List[float]],
    model_path: str = None
) -> Dict[str, Any]:
    """
    Perform breast cancer classification inference on an input sample.

    Parameters
    ----------
    sample : pd.DataFrame, dict, or list of float
        Cell nucleus measurement values for the 30 FNA features.
    model_path : str, optional
        Custom path to serialized joblib artifact.

    Returns
    -------
    dict
        Contains predicted class label, diagnosis name, probabilities, and risk score.
    """
    artifact = load_trained_pipeline(model_path)
    pipeline = artifact["pipeline"]
    feature_names = artifact["feature_names"]

    # Convert dictionary or list to DataFrame
    if isinstance(sample, dict):
        df_input = pd.DataFrame([sample])
    elif isinstance(sample, list):
        if len(sample) != len(feature_names):
            raise ValueError(
                f"Expected {len(feature_names)} features, got {len(sample)}."
            )
        df_input = pd.DataFrame([sample], columns=feature_names)
    elif isinstance(sample, pd.DataFrame):
        df_input = sample.copy()
    else:
        raise TypeError("sample must be a DataFrame, dict, or list of numbers.")

    # Reorder columns to match pipeline training signature
    df_input = df_input[feature_names]

    pred_class = int(pipeline.predict(df_input)[0])
    pred_label = "Malignant" if pred_class == 1 else "Benign"

    probabilities = pipeline.predict_proba(df_input)[0]
    prob_malignant = float(probabilities[1])
    prob_benign = float(probabilities[0])

    # Clinical risk categorization based on probability of malignancy
    if prob_malignant < 0.20:
        risk_level = "Low Risk (Likely Benign)"
    elif prob_malignant < 0.60:
        risk_level = "Borderline / Intermediate (Recommend Biopsy Review)"
    else:
        risk_level = "High Risk (Strong Indication of Malignancy)"

    return {
        "prediction_code": pred_class,
        "diagnosis": pred_label,
        "probability_malignant": prob_malignant,
        "probability_benign": prob_benign,
        "confidence": max(prob_malignant, prob_benign),
        "clinical_risk_assessment": risk_level
    }


def demo_sample_inference():
    """Demonstrates inference using typical benign and malignant test samples."""
    print("=" * 60)
    print("          BREAST CANCER KNN INFERENCE DEMOSTRATION           ")
    print("=" * 60)

    # Example 1: Known Malignant biopsy sample
    malignant_sample = {
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

    # Example 2: Known Benign biopsy sample
    benign_sample = {
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

    print("\n[Sample 1 - Expected: Malignant]")
    res1 = predict(malignant_sample)
    print(f"  Diagnosis: {res1['diagnosis']} (Confidence: {res1['confidence'] * 100:.1f}%)")
    print(f"  P(Malignant): {res1['probability_malignant']:.4f} | Assessment: {res1['clinical_risk_assessment']}")

    print("\n[Sample 2 - Expected: Benign]")
    res2 = predict(benign_sample)
    print(f"  Diagnosis: {res2['diagnosis']} (Confidence: {res2['confidence'] * 100:.1f}%)")
    print(f"  P(Benign): {res2['probability_benign']:.4f} | Assessment: {res2['clinical_risk_assessment']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Breast Cancer Malignancy with KNN")
    parser.add_argument("--demo", action="store_true", help="Run demonstration inference samples")
    args = parser.parse_args()

    demo_sample_inference()
