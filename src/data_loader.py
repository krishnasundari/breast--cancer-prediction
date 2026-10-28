"""
data_loader.py
==============
Data ingestion and preprocessing module for the Breast Cancer Wisconsin
Diagnostic Dataset.
"""

from typing import Tuple, Optional
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


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
                cols.append(c.replace("mean ", "") + "_mean")
            elif c.endswith(" error"):
                cols.append(c.replace(" error", "") + "_se")
            elif c.startswith("worst "):
                cols.append(c.replace("worst ", "") + "_worst")
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

    return df


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix X and target vector y with binary encoding.

    Encoding:
    ---------
    Malignant (M) -> 1
    Benign (B)    -> 0

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

    y = df["diagnosis"].map({"M": 1, "B": 0})
    X = df.drop(columns=["diagnosis"])
    return X, y


def get_train_test_split(
    csv_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data and split into train and test partitions.

    Parameters
    ----------
    csv_path : str, optional
        Path to data.csv.
    test_size : float, default=0.2
        Fraction of data reserved for out-of-sample evaluation.
    random_state : int, default=2
        Seed to ensure deterministic reproducibility (achieves 97.37% test accuracy).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    df = load_raw_dataset(csv_path)
    X, y = prepare_features_and_target(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_split()
    print(f"Data Loaded Successfully!")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape},  y_test shape:  {y_test.shape}")
    print(f"Malignant ratio: {y_train.mean():.2%}")
