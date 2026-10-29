"""
pipeline.py
===========
Defines the end-to-end Machine Learning pipelines using Scikit-learn.
Supports four classification algorithms:
1. K-Nearest Neighbors (KNN) - with StandardScaler
2. Logistic Regression       - with StandardScaler
3. Support Vector Machine    - with StandardScaler
4. Random Forest             - without feature scaling (tree-based, scale-invariant)
"""

from typing import Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def create_knn_pipeline(
    n_neighbors: int = 6,
    weights: str = "distance",
    metric: str = "euclidean",
    p: int = 2
) -> Pipeline:
    """
    Constructs an end-to-end KNN classification pipeline:
    1. StandardScaler:
       Scales features to zero mean and unit variance. Prevents high-magnitude
       features (e.g. area ~1000) from dominating distance calculations over
       sub-unit features (e.g. smoothness ~0.1).
    2. KNeighborsClassifier:
       Instance-based nonparametric classifier using inverse distance-weighted
       voting among the K nearest samples.

    Parameters
    ----------
    n_neighbors : int, default=6
        Number of nearest neighbors to query.
    weights : {'uniform', 'distance'}, default='distance'
        Weight function used in prediction.
    metric : str, default='euclidean'
        Distance metric for Minkowski tree.
    p : int, default=2
        Power parameter for Minkowski metric (2 = Euclidean).

    Returns
    -------
    Pipeline
        StandardScaler + KNeighborsClassifier.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
            n_jobs=1
        ))
    ])


def create_logistic_regression_pipeline(
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    solver: str = "lbfgs",
    penalty: str = "l2"
) -> Pipeline:
    """
    Constructs an end-to-end Logistic Regression classification pipeline:
    1. StandardScaler:
       Standardizes features for stable and fast gradient-based L-BFGS convergence.
    2. LogisticRegression:
       Regularized linear probabilistic model for binary classification.

    Parameters
    ----------
    C : float, default=1.0
        Inverse regularization strength.
    max_iter : int, default=1000
        Maximum iterations for solver convergence.
    random_state : int, default=42
        Reproducible seed.
    solver : str, default='lbfgs'
        Optimization solver.
    penalty : str, default='l2'
        Norm of the penalty.

    Returns
    -------
    Pipeline
        StandardScaler + LogisticRegression.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver=solver,
            penalty=penalty
        ))
    ])


def create_svm_pipeline(
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    probability: bool = True,
    random_state: int = 42
) -> Pipeline:
    """
    Constructs an end-to-end Support Vector Machine (SVM) classification pipeline:
    1. StandardScaler:
       Strictly required for SVM RBF kernel since margin maximization is highly
       sensitive to feature variance.
    2. SVC:
       Support Vector Classifier with radial basis function (RBF) kernel and Platt
       scaling (probability=True) for calibrated posterior probabilities.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter.
    kernel : str, default='rbf'
        Kernel type.
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for RBF.
    probability : bool, default=True
        Enable probability estimates via Platt scaling.
    random_state : int, default=42
        Reproducible seed.

    Returns
    -------
    Pipeline
        StandardScaler + SVC.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=probability,
            random_state=random_state
        ))
    ])


def create_random_forest_pipeline(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    random_state: int = 42
) -> Pipeline:
    """
    Constructs an end-to-end Random Forest classification pipeline:
    Note: Tree-based ensemble methods are invariant to monotonic feature scaling,
    so no StandardScaler is included, avoiding unnecessary computations.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of decision trees in the forest.
    max_depth : int, optional
        Maximum depth of the individual trees.
    min_samples_split : int, default=2
        Minimum samples required to split an internal node.
    random_state : int, default=42
        Reproducible seed.

    Returns
    -------
    Pipeline
        RandomForestClassifier.
    """
    return Pipeline([
        ("random_forest", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=1
        ))
    ])


def create_model_pipeline(model_name: str, **kwargs) -> Pipeline:
    """
    Factory function to construct a classification pipeline by name.

    Parameters
    ----------
    model_name : str
        One of {'knn', 'logistic_regression', 'svm', 'random_forest'}.
    **kwargs : dict
        Hyperparameters passed to the respective model creator.

    Returns
    -------
    Pipeline
    """
    name = model_name.lower().replace("-", "_").replace(" ", "_")
    if name in ["knn", "k_nearest_neighbors"]:
        return create_knn_pipeline(**kwargs)
    elif name in ["lr", "logistic_regression"]:
        return create_logistic_regression_pipeline(**kwargs)
    elif name in ["svm", "support_vector_machine", "svc"]:
        return create_svm_pipeline(**kwargs)
    elif name in ["rf", "random_forest"]:
        return create_random_forest_pipeline(**kwargs)
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Choose from: 'knn', 'logistic_regression', 'svm', 'random_forest'."
        )


def get_all_model_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    """
    Returns an instantiated dictionary of all four standard classification pipelines.

    Parameters
    ----------
    random_state : int, default=42
        Reproducible seed for stochastic models.

    Returns
    -------
    dict
        Mapping model identifier to configured Pipeline.
    """
    return {
        "knn": create_knn_pipeline(n_neighbors=6, weights="distance", metric="euclidean"),
        "logistic_regression": create_logistic_regression_pipeline(max_iter=1000, random_state=random_state),
        "svm": create_svm_pipeline(C=1.0, kernel="rbf", probability=True, random_state=random_state),
        "random_forest": create_random_forest_pipeline(n_estimators=100, random_state=random_state)
    }


def get_hyperparameter_grid(model_name: str = "knn") -> Dict[str, Any]:
    """
    Returns search space for GridSearchCV / RandomizedSearchCV tuning.
    """
    name = model_name.lower().replace("-", "_")
    if name in ["knn", "k_nearest_neighbors"]:
        return {
            "knn__n_neighbors": list(range(1, 31)),
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan"]
        }
    elif name in ["lr", "logistic_regression"]:
        return {
            "logistic_regression__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "logistic_regression__penalty": ["l2"]
        }
    elif name in ["svm", "support_vector_machine"]:
        return {
            "svm__C": [0.1, 1.0, 10.0],
            "svm__kernel": ["rbf", "linear"],
            "svm__gamma": ["scale", "auto"]
        }
    elif name in ["rf", "random_forest"]:
        return {
            "random_forest__n_estimators": [50, 100, 200],
            "random_forest__max_depth": [None, 5, 10, 20],
            "random_forest__min_samples_split": [2, 5]
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")
