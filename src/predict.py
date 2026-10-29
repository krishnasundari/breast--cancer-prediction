"""
predict.py
==========
Inference module for serving predictions from trained breast cancer pipelines
(KNN, Logistic Regression, SVM, and Random Forest).
Provides both programmatic API and Command Line Interface (CLI).
"""

import argparse
import os
from typing import Dict, Any, Union, List
import joblib
import numpy as np
import pandas as pd


MODEL_FILE_MAP = {
    "knn": "knn_pipeline.joblib",
    "k-nearest neighbors": "knn_pipeline.joblib",
    "logistic_regression": "logistic_regression_pipeline.joblib",
    "logistic regression": "logistic_regression_pipeline.joblib",
    "lr": "logistic_regression_pipeline.joblib",
    "svm": "svm_pipeline.joblib",
    "support vector machine": "svm_pipeline.joblib",
    "random_forest": "random_forest_pipeline.joblib",
    "random forest": "random_forest_pipeline.joblib",
    "rf": "random_forest_pipeline.joblib",
}


def get_models_dir() -> str:
    """Returns the absolute path to the models directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "models")


def load_trained_pipeline(model_name_or_path: str = "knn") -> Dict[str, Any]:
    """
    Loads a serialized model payload from disk by model identifier or file path.

    Parameters
    ----------
    model_name_or_path : str, default="knn"
        Key identifier ('knn', 'logistic_regression', 'svm', 'random_forest')
        or absolute/relative filesystem path to a .joblib artifact.

    Returns
    -------
    dict
        Deserialized artifact dictionary containing 'pipeline', 'model_name',
        'feature_names', 'metrics', etc.
    """
    if model_name_or_path is None:
        model_name_or_path = "knn"

    models_dir = get_models_dir()
    lookup_key = str(model_name_or_path).strip().lower()

    if lookup_key in MODEL_FILE_MAP:
        model_path = os.path.join(models_dir, MODEL_FILE_MAP[lookup_key])
    elif os.path.isabs(model_name_or_path) or os.path.exists(model_name_or_path):
        model_path = model_name_or_path
    else:
        # Check if it was provided as a bare filename in models/
        candidate = os.path.join(models_dir, model_name_or_path)
        if os.path.exists(candidate):
            model_path = candidate
        else:
            model_path = os.path.join(models_dir, f"{lookup_key}_pipeline.joblib")

    if not os.path.exists(model_path):
        available = [f for f in os.listdir(models_dir) if f.endswith(".joblib")] if os.path.exists(models_dir) else []
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            f"Available models in models/: {available}. "
            "Please run 'python -m src.train' to train and serialize pipelines."
        )

    return joblib.load(model_path)


def predict(
    sample: Union[pd.DataFrame, Dict[str, float], List[float]],
    model_name_or_path: str = "knn"
) -> Dict[str, Any]:
    """
    Perform breast cancer classification inference on an input sample.

    Parameters
    ----------
    sample : pd.DataFrame, dict, or list of float
        Cell nucleus measurement values for the 30 FNA features.
    model_name_or_path : str, default="knn"
        Identifier or path of the trained model to use for prediction.

    Returns
    -------
    dict
        Contains predicted class label, diagnosis name, probabilities, and clinical risk.
    """
    artifact = load_trained_pipeline(model_name_or_path)
    pipeline = artifact["pipeline"]
    feature_names = artifact["feature_names"]
    model_name = artifact.get("model_name", str(model_name_or_path))

    # Convert dictionary or list to DataFrame
    if isinstance(sample, dict):
        df_input = pd.DataFrame([sample])
    elif isinstance(sample, list):
        if len(sample) != len(feature_names):
            raise ValueError(
                f"Feature count mismatch: Expected {len(feature_names)} features, got {len(sample)}."
            )
        df_input = pd.DataFrame([sample], columns=feature_names)
    elif isinstance(sample, pd.DataFrame):
        df_input = sample.copy()
    else:
        raise TypeError("sample must be a pandas DataFrame, dictionary, or list of numbers.")

    # Validate presence of all required feature columns
    missing_cols = [col for col in feature_names if col not in df_input.columns]
    if missing_cols:
        raise ValueError(
            f"Input sample is missing {len(missing_cols)} required features: {missing_cols[:5]}"
            + ("..." if len(missing_cols) > 5 else "")
        )

    # Reorder columns to match canonical pipeline training signature
    df_aligned = df_input[feature_names].copy()

    # Validate no missing/NaN values
    if df_aligned.isnull().any().any():
        nan_cols = df_aligned.columns[df_aligned.isnull().any()].tolist()
        raise ValueError(f"Input sample contains NaN/null values in columns: {nan_cols}")

    pred_class = int(pipeline.predict(df_aligned)[0])
    pred_label = "Malignant" if pred_class == 1 else "Benign"

    probabilities = pipeline.predict_proba(df_aligned)[0]
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
        "model_name": model_name,
        "prediction_code": pred_class,
        "diagnosis": pred_label,
        "probability_malignant": prob_malignant,
        "probability_benign": prob_benign,
        "confidence": max(prob_malignant, prob_benign),
        "clinical_risk_assessment": risk_level,
        "features_evaluated": len(feature_names)
    }


def demo_sample_inference():
    """Demonstrates inference using typical benign and malignant test samples across all 4 models."""
    print("=" * 70)
    print("      BREAST CANCER MULTI-MODEL INFERENCE DEMONSTRATION           ")
    print("=" * 70)

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

    models_to_test = ["knn", "logistic_regression", "svm", "random_forest"]

    print("\n--- SAMPLE 1: KNOWN MALIGNANT BIOPSY ---")
    for m in models_to_test:
        try:
            res = predict(malignant_sample, model_name_or_path=m)
            print(f"[{res['model_name']:<24}] -> Diagnosis: {res['diagnosis']:<9} "
                  f"| P(Malignant): {res['probability_malignant']:.4f} "
                  f"| Confidence: {res['confidence']*100:.1f}%")
        except Exception as e:
            print(f"[{m:<24}] -> Error: {e}")

    print("\n--- SAMPLE 2: KNOWN BENIGN BIOPSY ---")
    for m in models_to_test:
        try:
            res = predict(benign_sample, model_name_or_path=m)
            print(f"[{res['model_name']:<24}] -> Diagnosis: {res['diagnosis']:<9} "
                  f"| P(Benign): {res['probability_benign']:.4f} "
                  f"| Confidence: {res['confidence']*100:.1f}%")
        except Exception as e:
            print(f"[{m:<24}] -> Error: {e}")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Breast Cancer Malignancy")
    parser.add_argument("--demo", action="store_true", help="Run demonstration inference samples")
    parser.add_argument("--model", type=str, default="knn", help="Model name (knn, logistic_regression, svm, random_forest)")
    args = parser.parse_args()

    demo_sample_inference()
