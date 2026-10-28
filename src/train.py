"""
train.py
========
Training script for the Breast Cancer Detection KNN pipeline.
Handles model training, cross-validation, hyperparameter tuning, and serialization.
"""

import argparse
import os
import joblib
from sklearn.model_selection import cross_val_score, GridSearchCV

from src.data_loader import get_train_test_split
from src.pipeline import create_knn_pipeline, get_hyperparameter_grid
from src.evaluate import evaluate_model, print_evaluation_summary


def train(
    tune: bool = False,
    n_neighbors: int = 6,
    weights: str = "distance",
    metric: str = "euclidean",
    save_model: bool = True
):
    """
    Train and evaluate the KNN classification pipeline.

    Parameters
    ----------
    tune : bool, default=False
        Whether to perform 5-fold cross-validation grid search.
    n_neighbors : int, default=6
        Optimal K chosen from elbow analysis and tuning.
    weights : str, default='distance'
        Weighting function for neighbors.
    metric : str, default='euclidean'
        Distance metric.
    save_model : bool, default=True
        Whether to persist model to models/knn_pipeline.joblib.
    """
    print("Loading Wisconsin Breast Cancer Diagnostic Dataset...")
    X_train, X_test, y_train, y_test = get_train_test_split(random_state=2)
    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    if tune:
        print("\nExecuting GridSearchCV across K in [1..30], weights, and metrics...")
        base_pipe = create_knn_pipeline()
        param_grid = get_hyperparameter_grid()
        grid = GridSearchCV(base_pipe, param_grid, cv=5, scoring="accuracy", n_jobs=1)
        grid.fit(X_train, y_train)
        print(f"Best Parameters: {grid.best_params_}")
        print(f"Best 5-Fold CV Accuracy: {grid.best_score_ * 100:.2f}%")
        pipeline = grid.best_estimator_
    else:
        print(f"\nBuilding Pipeline: StandardScaler + KNN(k={n_neighbors}, weights='{weights}', metric='{metric}')")
        pipeline = create_knn_pipeline(n_neighbors=n_neighbors, weights=weights, metric=metric)
        pipeline.fit(X_train, y_train)

        # Cross-validation check
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
        print(f"5-Fold Cross-Validation Accuracy on Training Set: {cv_scores.mean() * 100:.2f}% (±{cv_scores.std() * 100:.2f}%)")

    # Evaluate on held-out test partition
    metrics = evaluate_model(pipeline, X_test, y_test)
    print_evaluation_summary(metrics)

    if save_model:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "knn_pipeline.joblib")

        payload = {
            "pipeline": pipeline,
            "feature_names": list(X_train.columns),
            "target_mapping": {0: "Benign", 1: "Malignant"},
            "metrics": metrics
        }
        joblib.dump(payload, model_path)
        print(f"\nModel pipeline successfully saved to: {model_path}")

    return pipeline, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Breast Cancer KNN Classifier")
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter grid search")
    parser.add_argument("--k", type=int, default=6, help="K neighbors (default=6)")
    parser.add_argument("--weights", type=str, default="distance", choices=["uniform", "distance"])
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "manhattan"])
    args = parser.parse_args()

    train(tune=args.tune, n_neighbors=args.k, weights=args.weights, metric=args.metric)
