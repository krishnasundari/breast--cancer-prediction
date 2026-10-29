"""
data_loader.py
==============
Data ingestion, validation, and preprocessing module for the Breast Cancer Wisconsin
(Diagnostic) Dataset.
"""

from typing import Tuple, Optional, Dict, Any, List
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


# Canonical 30 feature names matching the Wisconsin Diagnostic Dataset
FEATURE_NAMES: List[str] = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]


def get_default_data_path() -> str:
    """Returns the default absolute path to data/data.csv."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", "data.csv")


def load_raw_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw Wisconsin dataset from CSV or fallback to scikit-learn dataset.

    Parameters
    ----------
    csv_path : str, optional
        Custom path to the data.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with 30 cell nucleus features and 'diagnosis' target.
    """
    if csv_path is None:
        csv_path = get_default_data_path()

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Fallback to sklearn dataset if csv not found
        sklearn_data = load_breast_cancer(as_frame=True)
        raw_df = sklearn_data.frame
        cols = []
        for c in sklearn_data.feature_names:
            if c.startswith("mean "):
                base = c.replace("mean ", "")
                if base == "fractal dimension":
                    base = "fractal_dimension"
                cols.append(base + "_mean")
            elif c.endswith(" error"):
                base = c.replace(" error", "")
                if base == "fractal dimension":
                    base = "fractal_dimension"
                cols.append(base + "_se")
            elif c.startswith("worst "):
                base = c.replace("worst ", "")
                if base == "fractal dimension":
                    base = "fractal_dimension"
                cols.append(base + "_worst")
            else:
                cols.append(c.replace(" ", "_"))
        df = raw_df.drop(columns=["target"])
        df.columns = cols
        df.insert(0, "diagnosis", raw_df["target"].map({0: "M", 1: "B"}))
        df.insert(0, "id", range(842302, 842302 + len(df)))

    # Drop non-feature and artifact columns
    drop_cols = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Check and remove duplicates if any
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()

    # Handle missing values if any exist
    if df.isnull().sum().sum() > 0:
        # Fill numeric missing values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df


def validate_dataset(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Verifies dataset integrity, shapes, nulls, duplicates, and class distributions.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Dataset to validate. If None, loads the raw dataset automatically.

    Returns
    -------
    dict
        Verification summary including sample count, feature count, class counts.
    """
    if df is None:
        df = load_raw_dataset()

    feature_cols = [c for c in df.columns if c != "diagnosis"]
    class_counts = df["diagnosis"].value_counts().to_dict() if "diagnosis" in df.columns else {}

    return {
        "sample_count": int(df.shape[0]),
        "num_samples": int(df.shape[0]),
        "feature_count": int(len(feature_cols)),
        "num_features": int(len(feature_cols)),
        "feature_names": feature_cols,
        "target_column": "diagnosis" if "diagnosis" in df.columns else None,
        "class_counts": class_counts,
        "class_distribution": {
            "Benign (0)": int(class_counts.get("B", 0)),
            "Malignant (1)": int(class_counts.get("M", 0))
        },
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "is_valid": len(feature_cols) == 30 and df.isnull().sum().sum() == 0 and df.shape[0] > 0
    }


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix X and target vector y with binary encoding.

    Encoding:
    ---------
    Malignant (M) -> 1 (Positive class in clinical diagnosis)
    Benign (B)    -> 0 (Negative class)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing features and 'diagnosis' column.

    Returns
    -------
    X : pd.DataFrame (shape: [N, 30])
        Continuous numerical features from fine needle aspirate (FNA).
    y : pd.Series (shape: [N])
        Binary target vector (0 or 1).
    """
    if "diagnosis" not in df.columns:
        raise ValueError("Missing 'diagnosis' column in dataset.")

    y = df["diagnosis"].map({"M": 1, "B": 0}).astype(int)
    X = df.drop(columns=["diagnosis"])

    # Ensure consistent canonical feature ordering
    ordered_cols = [col for col in FEATURE_NAMES if col in X.columns]
    if len(ordered_cols) == len(FEATURE_NAMES):
        X = X[FEATURE_NAMES]

    return X, y


def get_train_test_split(
    csv_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data and split into train and test partitions.

    Parameters
    ----------
    csv_path : str, optional
        Path to data.csv.
    test_size : float, default=0.2
        Fraction of data reserved for out-of-sample evaluation.
    random_state : int, default=42
        Seed to ensure deterministic reproducibility.
    stratify : bool, default=True
        Whether to perform stratified sampling based on the target class.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    df = load_raw_dataset(csv_path)
    X, y = prepare_features_and_target(df)
    stratify_target = y if stratify else None
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )


if __name__ == "__main__":
    df_raw = load_raw_dataset()
    val_report = validate_dataset(df_raw)
    print("Dataset Validation Report:")
    for k, v in val_report.items():
        if k != "feature_names":
            print(f"  {k}: {v}")
    X_train, X_test, y_train, y_test = get_train_test_split()
    print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"Train Malignant ratio: {y_train.mean():.2%}, Test Malignant ratio: {y_test.mean():.2%}")
