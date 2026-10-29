"""
evaluate.py
===========
Model evaluation, stratified cross-validation, clinical metrics,
report generation (CSV/JSON), and visual analytics.
"""

from typing import Dict, Any, Optional, List, Tuple
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Computes key classification metrics with particular focus on clinical reliability.
    The positive class (1) corresponds to Malignant diagnosis.

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

    # Probabilities for ROC-AUC and PR-AUC
    y_prob = None
    auc_score = None
    pr_auc_score = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            auc_score = float(roc_auc_score(y_test, y_prob))
            pr_auc_score = float(average_precision_score(y_test, y_prob))
        except Exception:
            y_prob = None
            auc_score = None
            pr_auc_score = None
    elif hasattr(model, "decision_function"):
        try:
            y_decision = model.decision_function(X_test)
            auc_score = float(roc_auc_score(y_test, y_decision))
            pr_auc_score = float(average_precision_score(y_test, y_decision))
        except Exception:
            auc_score = None
            pr_auc_score = None

    acc = float(accuracy_score(y_test, y_pred))
    # Malignant metrics (pos_label=1)
    prec_mal = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec_mal = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1_mal = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    # Benign metrics (pos_label=0)
    prec_ben = float(precision_score(y_test, y_pred, pos_label=0, zero_division=0))
    rec_ben = float(recall_score(y_test, y_pred, pos_label=0, zero_division=0))
    f1_ben = float(f1_score(y_test, y_pred, pos_label=0, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr_malignant = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    fdr = float(fp / (fp + tp)) if (fp + tp) > 0 else 0.0

    metrics = {
        "accuracy": acc,
        "precision": prec_mal,
        "recall_sensitivity": rec_mal,
        "f1_score": f1_mal,
        "malignant_precision": prec_mal,
        "malignant_recall": rec_mal,
        "malignant_f1": f1_mal,
        "benign_precision": prec_ben,
        "benign_recall": rec_ben,
        "benign_f1": f1_ben,
        "specificity": specificity,
        "false_negative_rate_malignant": fnr_malignant,
        "false_discovery_rate": fdr,
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "roc_auc": auc_score,
        "pr_auc": pr_auc_score,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "raw_confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Benign (0)", "Malignant (1)"], digits=4
        )
    }
    return metrics


def cross_validate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs Stratified K-Fold cross-validation strictly on the training partition.
    Ensures zero data leakage as scaling/preprocessing inside pipeline is fit per fold.

    Parameters
    ----------
    model : sklearn Pipeline
    X_train : pd.DataFrame
    y_train : pd.Series
    cv : int, default=5
    random_state : int, default=42

    Returns
    -------
    dict
        Mean and standard deviation for accuracy, precision, recall, f1, and roc_auc.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    cv_results = cross_validate(
        model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=1
    )

    return {
        "cv_folds": cv,
        "cv_accuracy_mean": float(cv_results["test_accuracy"].mean()),
        "cv_accuracy_std": float(cv_results["test_accuracy"].std()),
        "cv_precision_mean": float(cv_results["test_precision"].mean()),
        "cv_precision_std": float(cv_results["test_precision"].std()),
        "cv_recall_mean": float(cv_results["test_recall"].mean()),
        "cv_recall_std": float(cv_results["test_recall"].std()),
        "cv_f1_mean": float(cv_results["test_f1"].mean()),
        "cv_f1_std": float(cv_results["test_f1"].std()),
        "cv_roc_auc_mean": float(cv_results["test_roc_auc"].mean()),
        "cv_roc_auc_std": float(cv_results["test_roc_auc"].std())
    }


def compare_all_models(
    models_dict: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trains and evaluates all provided models.
    Executes cross-validation on training data and held-out test evaluation.

    Returns
    -------
    comparison_df : pd.DataFrame
    full_results : dict
    """
    records = []
    full_results = {}

    for name, pipeline in models_dict.items():
        print(f"--> Evaluating model: {name.upper()}...")
        # 1. Stratified Cross-Validation on training data
        cv_metrics = cross_validate_model(
            pipeline, X_train, y_train, cv=cv, random_state=random_state
        )

        # 2. Fit pipeline on full training data
        pipeline.fit(X_train, y_train)

        # 3. Evaluate on held-out test set
        test_metrics = evaluate_model(pipeline, X_test, y_test)

        full_results[name] = {
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics
        }

        cm = test_metrics["confusion_matrix"]
        records.append({
            "Model": name,
            "CV_Accuracy_Mean": cv_metrics["cv_accuracy_mean"],
            "CV_Accuracy_Std": cv_metrics["cv_accuracy_std"],
            "CV_Recall_Mean": cv_metrics["cv_recall_mean"],
            "CV_F1_Mean": cv_metrics["cv_f1_mean"],
            "CV_ROC_AUC_Mean": cv_metrics["cv_roc_auc_mean"],
            "Test_Accuracy": test_metrics["accuracy"],
            "Test_Malignant_Precision": test_metrics["malignant_precision"],
            "Test_Malignant_Recall": test_metrics["malignant_recall"],
            "Test_Malignant_F1": test_metrics["malignant_f1"],
            "Test_ROC_AUC": test_metrics["roc_auc"],
            "Test_PR_AUC": test_metrics.get("pr_auc"),
            "Test_Malignant_FNR": test_metrics.get("false_negative_rate_malignant"),
            "Test_FDR": test_metrics.get("false_discovery_rate"),
            "Test_TP": cm["true_positives"],
            "Test_TN": cm["true_negatives"],
            "Test_FP": cm["false_positives"],
            "Test_FN": cm["false_negatives"]
        })

    comparison_df = pd.DataFrame(records).sort_values(by="Test_Accuracy", ascending=False)
    return comparison_df, full_results


def save_comparison_reports(
    comparison_df: pd.DataFrame,
    full_results: Dict[str, Any],
    output_dir: str
) -> Tuple[str, str]:
    """Saves comparison reports in CSV and JSON formats."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "model_comparison.csv")
    json_path = os.path.join(output_dir, "model_comparison.json")

    comparison_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    return csv_path, json_path


def plot_model_comparison(comparison_df: pd.DataFrame, output_path: str):
    """Generates comparison bar chart across key metrics."""
    models = comparison_df["Model"].tolist()
    metrics = ["Test_Accuracy", "Test_Malignant_Recall", "Test_Malignant_F1", "Test_ROC_AUC"]
    labels = ["Accuracy", "Malignant Recall", "Malignant F1", "ROC-AUC"]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ["#4A148C", "#1E88E5", "#00897B", "#F4511E"]

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [comparison_df.loc[comparison_df["Model"] == m, metric].values[0] for m in models]
        rects = ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color, alpha=0.9)
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height):
                ax.annotate(f"{height*100:.1f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Four-Model Benchmark: Held-Out Test Set Performance", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models], fontsize=10, fontweight="bold")
    ax.set_ylim(0.85, 1.05)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_multi_roc_curves(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str
):
    """Plots combined ROC curves for all trained models."""
    plt.figure(figsize=(8, 6), dpi=300)
    colors = {
        "knn": "#4A148C",
        "logistic_regression": "#1E88E5",
        "svm": "#00897B",
        "random_forest": "#E53935"
    }

    for name, pipeline in models_dict.items():
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            display_name = name.replace("_", " ").title()
            color = colors.get(name, "#333333")
            plt.plot(fpr, tpr, lw=2.2, color=color, label=f"{display_name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], color="#9E9E9E", lw=1.5, linestyle="--", label="Chance (AUC = 0.50)")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Multi-Model Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_multi_pr_curves(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str
):
    """
    Plots combined Precision-Recall curves across all trained models.
    Particularly informative in clinical diagnostics where Malignant is the positive event.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    colors = {
        "knn": "#4A148C",
        "logistic_regression": "#1E88E5",
        "svm": "#00897B",
        "random_forest": "#E53935"
    }

    for name, pipeline in models_dict.items():
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_prob = pipeline.decision_function(X_test)
        else:
            continue

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        display_name = name.replace("_", " ").title()
        color = colors.get(name, "#333333")
        plt.plot(recall, precision, lw=2.2, color=color, label=f"{display_name} (PR-AUC = {pr_auc:.4f})")

    # Baseline rate (prevalence of Malignant class in test set)
    prevalence = float(y_test.sum() / len(y_test))
    plt.axhline(y=prevalence, color="#9E9E9E", lw=1.5, linestyle="--", label=f"Baseline Prevalence ({prevalence*100:.1f}%)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([0.4, 1.05])
    plt.xlabel("Malignant Recall / Sensitivity", fontsize=11)
    plt.ylabel("Malignant Precision / PPV", fontsize=11)
    plt.title("Multi-Model Precision-Recall (PR) Curves", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def print_evaluation_summary(metrics: Dict[str, Any], model_name: str = "Model") -> None:
    """Prints a formatted evaluation report for console output and logging."""
    cm = metrics["confusion_matrix"]
    print("=" * 65)
    print(f"        {model_name.upper()} CLASSIFICATION EVALUATION REPORT       ")
    print("=" * 65)
    print(f"Overall Classification Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Malignant Class Precision       : {metrics['malignant_precision'] * 100:.2f}%")
    print(f"Malignant Class Recall (Sens.)  : {metrics['malignant_recall'] * 100:.2f}%")
    print(f"Benign Class Specificity        : {metrics['specificity'] * 100:.2f}%")
    print(f"Malignant F1-Score              : {metrics['malignant_f1'] * 100:.2f}%")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC Score                   : {metrics['roc_auc']:.4f}")
    print("-" * 65)
    print("CONFUSION MATRIX BREAKDOWN:")
    print(f"  • True Negatives  (Benign correctly identified)    : {cm['true_negatives']}")
    print(f"  • False Positives (Benign misclassified as Malig.) : {cm['false_positives']}")
    print(f"  • False Negatives (Malignant MISSED — Critical)    : {cm['false_negatives']}")
    print(f"  • True Positives  (Malignant correctly identified) : {cm['true_positives']}")
    print("-" * 65)
    print("DETAILED SCIKIT-LEARN CLASSIFICATION REPORT:")
    print(metrics["classification_report"])
    print("=" * 65)
