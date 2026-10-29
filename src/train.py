"""
train.py
========
Training and benchmarking script for the Breast Cancer Wisconsin classifiers.
Trains and compares four machine learning models:
1. K-Nearest Neighbors (KNN)
2. Logistic Regression
3. Support Vector Machine (SVM)
4. Random Forest

Saves trained pipelines to models/, reports to reports/, and visual charts to assets/.
"""

import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.data_loader import get_train_test_split, FEATURE_NAMES
from src.pipeline import (
    create_knn_pipeline,
    create_logistic_regression_pipeline,
    create_svm_pipeline,
    create_random_forest_pipeline,
    get_all_model_pipelines,
    get_hyperparameter_grid
)
from src.evaluate import (
    evaluate_model,
    cross_validate_model,
    compare_all_models,
    save_comparison_reports,
    plot_model_comparison,
    plot_multi_roc_curves,
    print_evaluation_summary
)


def train_single_model(
    model_name: str = "knn",
    random_state: int = 42,
    stratify: bool = True,
    tune: bool = False,
    save_model: bool = True
):
    """
    Trains and evaluates a single selected model.
    """
    print(f"\n[+] Loading Wisconsin Diagnostic Dataset (seed={random_state}, stratified={stratify})...")
    X_train, X_test, y_train, y_test = get_train_test_split(
        random_state=random_state, stratify=stratify
    )
    print(f"    Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    models_dict = get_all_model_pipelines(random_state=random_state)
    name = model_name.lower().replace("-", "_")
    if name not in models_dict:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(models_dict.keys())}")

    pipeline = models_dict[name]

    if tune:
        print(f"\n[+] Running GridSearchCV hyperparameter tuning for {name.upper()}...")
        param_grid = get_hyperparameter_grid(name)
        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=1)
        grid.fit(X_train, y_train)
        print(f"    Best Parameters: {grid.best_params_}")
        print(f"    Best CV Accuracy: {grid.best_score_ * 100:.2f}%")
        pipeline = grid.best_estimator_
    else:
        print(f"\n[+] Performing 5-Fold Stratified Cross-Validation for {name.upper()}...")
        cv_metrics = cross_validate_model(pipeline, X_train, y_train, cv=5, random_state=random_state)
        print(f"    Mean CV Accuracy: {cv_metrics['cv_accuracy_mean'] * 100:.2f}% (±{cv_metrics['cv_accuracy_std'] * 100:.2f}%)")
        print(f"    Mean CV Recall:   {cv_metrics['cv_recall_mean'] * 100:.2f}% (±{cv_metrics['cv_recall_std'] * 100:.2f}%)")
        print(f"    Mean CV F1-Score: {cv_metrics['cv_f1_mean'] * 100:.2f}% (±{cv_metrics['cv_f1_std'] * 100:.2f}%)")

        print(f"\n[+] Fitting {name.upper()} pipeline on complete training partition...")
        pipeline.fit(X_train, y_train)

    # Evaluate on held-out test partition
    print(f"\n[+] Evaluating on Held-Out Test Set (114 samples)...")
    metrics = evaluate_model(pipeline, X_test, y_test)
    print_evaluation_summary(metrics, model_name=name)

    if save_model:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{name}_pipeline.joblib")

        payload = {
            "model_name": name,
            "pipeline": pipeline,
            "feature_names": FEATURE_NAMES,
            "target_mapping": {0: "Benign", 1: "Malignant"},
            "metrics": metrics
        }
        joblib.dump(payload, model_path)
        print(f"\n[OK] Saved {name} pipeline artifact to: {model_path}")

    return pipeline, metrics


def train_and_compare_all_models(
    random_state: int = 42,
    stratify: bool = True,
    cv: int = 5,
    save_models: bool = True
):
    """
    Trains, cross-validates, and evaluates all four classification models on the same split:
    1. K-Nearest Neighbors (KNN)
    2. Logistic Regression
    3. Support Vector Machine (SVM)
    4. Random Forest

    Outputs a structured comparative table and saves all pipeline artifacts.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, "reports")
    assets_dir = os.path.join(base_dir, "assets")
    models_dir = os.path.join(base_dir, "models")

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70)
    print("   FOUR-MODEL BENCHMARK: BREAST CANCER CLASSIFICATION PIPELINE   ")
    print("=" * 70)
    print(f"Configuration: Stratified={stratify}, Test Size=20%, Random Seed={random_state}, CV Folds={cv}\n")

    X_train, X_test, y_train, y_test = get_train_test_split(
        random_state=random_state, stratify=stratify
    )
    print(f"Data Partition: {X_train.shape[0]} training samples | {X_test.shape[0]} test samples")
    print(f"Class Distribution: {int((y_train == 0).sum())} Benign, {int((y_train == 1).sum())} Malignant (Train)")
    print(f"                    {int((y_test == 0).sum())} Benign, {int((y_test == 1).sum())} Malignant (Test)\n")

    models_dict = get_all_model_pipelines(random_state=random_state)

    # Run full comparative training and evaluation
    comparison_df, full_results = compare_all_models(
        models_dict=models_dict,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv=cv,
        random_state=random_state
    )

    # Print clean benchmark table
    print("\n" + "=" * 90)
    print("                     FOUR-MODEL EVALUATION COMPARISON TABLE                     ")
    print("=" * 90)
    display_cols = [
        "Model", "CV_Accuracy_Mean", "CV_Recall_Mean", "Test_Accuracy",
        "Test_Malignant_Precision", "Test_Malignant_Recall", "Test_Malignant_F1", "Test_ROC_AUC",
        "Test_FP", "Test_FN"
    ]
    formatted_df = comparison_df[display_cols].copy()
    for col in ["CV_Accuracy_Mean", "CV_Recall_Mean", "Test_Accuracy", "Test_Malignant_Precision", "Test_Malignant_Recall", "Test_Malignant_F1", "Test_ROC_AUC"]:
        formatted_df[col] = formatted_df[col].apply(lambda v: f"{v*100:.2f}%" if not pd.isna(v) else "N/A")
    print(formatted_df.to_string(index=False))
    print("=" * 90)

    # Save reports
    csv_rep, json_rep = save_comparison_reports(comparison_df, full_results, reports_dir)
    print(f"\n[OK] Machine-readable comparison CSV saved to : {csv_rep}")
    print(f"[OK] Machine-readable comparison JSON saved to: {json_rep}")

    # Generate evaluation plots
    comp_plot = os.path.join(assets_dir, "model_comparison.png")
    plot_model_comparison(comparison_df, comp_plot)
    print(f"[OK] Model comparison bar chart saved to       : {comp_plot}")

    roc_plot = os.path.join(assets_dir, "multi_roc_curves.png")
    plot_multi_roc_curves(models_dict, X_test, y_test, roc_plot)
    print(f"[OK] Multi-model ROC-AUC curve chart saved to  : {roc_plot}")

    # Save all four pipeline artifacts
    if save_models:
        for name, pipeline in models_dict.items():
            model_path = os.path.join(models_dir, f"{name}_pipeline.joblib")
            payload = {
                "model_name": name,
                "pipeline": pipeline,
                "feature_names": FEATURE_NAMES,
                "target_mapping": {0: "Benign", 1: "Malignant"},
                "metrics": full_results[name]["test_metrics"],
                "cv_metrics": full_results[name]["cv_metrics"]
            }
            joblib.dump(payload, model_path)
            print(f"[OK] Saved pipeline artifact: {model_path}")

        # Summary registry
        summary_path = os.path.join(models_dir, "models_summary.json")
        summary_data = {
            "best_model": comparison_df.iloc[0]["Model"],
            "models_evaluated": list(models_dict.keys()),
            "evaluation_metrics": comparison_df.to_dict(orient="records")
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            import json
            json.dump(summary_data, f, indent=2)
        print(f"[OK] Saved model registry summary: {summary_path}")

    return models_dict, comparison_df, full_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Compare Breast Cancer Classification Models")
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "knn", "logistic_regression", "svm", "random_forest"],
        help="Model to train ('all' trains and compares all four algorithms)"
    )
    parser.add_argument("--random-state", type=int, default=42, help="Reproducible random seed")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--non-stratified", action="store_true", help="Disable stratified splitting")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter search for single model")
    args = parser.parse_args()

    stratified = not args.non_stratified

    if args.model == "all":
        train_and_compare_all_models(
            random_state=args.random_state,
            stratify=stratified,
            cv=args.cv,
            save_models=True
        )
    else:
        train_single_model(
            model_name=args.model,
            random_state=args.random_state,
            stratify=stratified,
            tune=args.tune,
            save_model=True
        )
