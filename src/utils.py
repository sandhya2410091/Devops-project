"""
utils.py
--------
Shared utility functions used by both the training pipeline (train.py) and
the prediction service (predict.py / app.py). Keeping these in one module
guarantees that the exact same preprocessing logic is applied at training
time and at inference time, which is essential for a correct MLOps pipeline.

Responsibilities covered here:
    - Data loading (from the DVC-tracked CSV)
    - Train/test splitting
    - Feature engineering (derived ratio features)
    - Feature scaling (StandardScaler)
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "breast_cancer.csv"
)

TARGET_COLUMN = "diagnosis"

# Engineered feature names, derived from raw features. Kept as a module-level
# constant so app.py / predict.py know exactly which extra columns to expect
# after engineer_features() is applied to raw input.
ENGINEERED_FEATURES = [
    "area_perimeter_ratio",
    "concavity_symmetry_ratio",
]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads the raw dataset produced by get_data.py / tracked by DVC.

    Raises:
        FileNotFoundError: if the dataset has not been generated yet. This is
            intentional -- the pipeline should never silently fabricate data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python src/get_data.py` "
            "(or `dvc repro`) first to generate the versioned dataset."
        )
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a small set of domain-informed derived features on top of the raw
    30 measurements in the Breast Cancer Wisconsin dataset.

        - area_perimeter_ratio: mean_area / mean_perimeter, a shape-density
          descriptor that is not directly present in the raw feature set.
        - concavity_symmetry_ratio: mean_concavity / mean_symmetry, capturing
          the interaction between two independently informative shape
          descriptors.

    A small epsilon is added to denominators to avoid division-by-zero on
    degenerate inputs.

    Args:
        df: DataFrame containing at least the raw feature columns.

    Returns:
        A new DataFrame with the engineered columns appended.
    """
    df = df.copy()
    eps = 1e-9
    df["area_perimeter_ratio"] = df["mean area"] / (df["mean perimeter"] + eps)
    df["concavity_symmetry_ratio"] = df["mean concavity"] / (df["mean symmetry"] + eps)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Returns the ordered list of feature column names (excludes target)."""
    return [c for c in df.columns if c != TARGET_COLUMN]


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Splits a DataFrame into (X, y) using TARGET_COLUMN as the label."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def preprocess_and_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Full preprocessing pipeline used at training time:
        1. Feature engineering
        2. Feature/target split
        3. Train/test split (stratified on the target to preserve class ratio)
        4. Feature scaling (fit on train only, to avoid data leakage)

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, fitted_scaler, feature_names
    """
    df = engineer_features(df)
    X, y = split_features_target(df)
    feature_names = get_feature_columns(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.to_numpy(), y_test.to_numpy(), scaler, feature_names


def preprocess_single_record(
    record: dict,
    scaler: StandardScaler,
    feature_names: List[str],
) -> np.ndarray:
    """
    Applies the identical feature-engineering + scaling pipeline to a single
    raw input record (e.g. a prediction request) at inference time.

    Args:
        record: dict mapping raw feature name -> value (the 30 original
            measurements, WITHOUT the engineered features).
        scaler: the StandardScaler fitted during training.
        feature_names: ordered feature names expected by the model (raw +
            engineered), matching training-time column order.

    Returns:
        A (1, n_features) numpy array ready to feed into the model.
    """
    row_df = pd.DataFrame([record])
    row_df = engineer_features(row_df)
    row_df = row_df[feature_names]  # enforce exact training-time column order
    return scaler.transform(row_df)
