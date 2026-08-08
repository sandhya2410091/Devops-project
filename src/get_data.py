"""
get_data.py
-----------
Fetches the Breast Cancer Prediction dataset (Wisconsin Diagnostic Breast
Cancer dataset, a well-known structured binary classification dataset) and
writes it to disk as a CSV file at data/raw/breast_cancer.csv.

This CSV is the artifact that gets version-controlled with DVC. Running this
script is the first stage of the DVC pipeline defined in dvc.yaml.

Usage:
    python src/get_data.py
"""

import os

import pandas as pd
from sklearn.datasets import load_breast_cancer

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "breast_cancer.csv")


def fetch_and_save_dataset(output_path: str = RAW_DATA_PATH) -> str:
    """
    Loads the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn
    and persists it as a single CSV file (features + target column).

    Args:
        output_path: Destination path for the CSV file.

    Returns:
        The path to the saved CSV file.
    """
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()

    # sklearn stores the label under "target"; rename to something explicit
    # and human readable for downstream clarity.
    df = df.rename(columns={"target": "diagnosis"})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Class balance:\n{df['diagnosis'].value_counts()}")

    return output_path


if __name__ == "__main__":
    fetch_and_save_dataset()
