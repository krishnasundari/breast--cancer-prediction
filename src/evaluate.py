"""
evaluate.py
===========
Model evaluation metrics, clinical risk analysis, and report generation.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Computes key classification metrics with particular focus on clinical reliability.

    Parameters
    ----------
    model : fitted sklearn estimator/pipeline
    X_test : pd.DataFrame
    y_test : pd.Series

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, f1, auc, confusion matrix.
    """
    y_pred = model.predict(X_test)

    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc_score = float(roc_auc_score(y_test, y_prob))
    else:
        y_prob = None
        auc_score = None

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall_sensitivity": rec,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": auc_score,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Benign (0)", "Malignant (1)"], digits=4
        )
    }
    return metrics


def print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    """Prints a formatted evaluation report for console output and logging."""
    cm = metrics["confusion_matrix"]
    print("=" * 60)
    print("        BREAST CANCER CLASSIFICATION EVALUATION REPORT       ")
    print("=" * 60)
    print(f"Overall Classification Accuracy : {metrics['accuracy'] * 100:.2f}% (Benchmark: 97.3%)")
    print(f"Precision (Malignant)           : {metrics['precision'] * 100:.2f}%")
    print(f"Recall / Sensitivity (Malignant): {metrics['recall_sensitivity'] * 100:.2f}%")
    print(f"Specificity (Benign)            : {metrics['specificity'] * 100:.2f}%")
    print(f"F1-Score (Harmonic Mean)        : {metrics['f1_score'] * 100:.2f}%")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC Score                   : {metrics['roc_auc']:.4f}")
    print("-" * 60)
    print("CONFUSION MATRIX BREAKDOWN:")
    print(f"  • True Negatives  (Benign correctly identified)    : {cm['true_negatives']}")
    print(f"  • False Positives (Benign misclassified as Malig.) : {cm['false_positives']}")
    print(f"  • False Negatives (Malignant MISSED — Critical)    : {cm['false_negatives']}")
    print(f"  • True Positives  (Malignant correctly identified) : {cm['true_positives']}")
    print("-" * 60)
    print("DETAILED SCIKIT-LEARN CLASSIFICATION REPORT:")
    print(metrics["classification_report"])
    print("=" * 60)
