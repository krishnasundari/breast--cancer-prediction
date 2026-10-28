"""
pipeline.py
===========
Defines the end-to-end Machine Learning pipeline using scikit-learn.
Combines z-score standardization (StandardScaler) with K-Nearest Neighbors (KNN).
"""

from typing import Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def create_knn_pipeline(
    n_neighbors: int = 6,
    weights: str = "distance",
    metric: str = "euclidean",
    p: int = 2
) -> Pipeline:
    """
    Constructs an end-to-end classification pipeline:
    1. StandardScaler:
       Scales features to zero mean and unit variance. Prevents high-magnitude
       features (e.g. area ~1000) from dominating distance calculations over
       sub-unit features (e.g. smoothness ~0.1).
    2. KNeighborsClassifier:
       Instance-based nonparametric classifier using majority / distance-weighted
       voting among the K nearest samples.

    Parameters
    ----------
    n_neighbors : int, default=6
        Number of nearest neighbors to query.
    weights : {'uniform', 'distance'}, default='distance'
        Weight function used in prediction. 'distance' weights closer neighbors
        more heavily via inverse distance (1/d).
    metric : str, default='euclidean'
        Distance metric to use for the tree (Euclidean distance = Minkowski with p=2).
    p : int, default=2
        Power parameter for Minkowski metric.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Encapsulated preprocessor and estimator pipeline.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
            n_jobs=1
        ))
    ])
    return pipeline


def get_hyperparameter_grid() -> Dict[str, Any]:
    """
    Returns search space for GridSearchCV / RandomizedSearchCV tuning.
    """
    return {
        "knn__n_neighbors": list(range(1, 31)),
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"]
    }
