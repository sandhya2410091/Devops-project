"""
conftest.py
-----------
Ensures the `src/` package is importable from the tests without needing an
installed package, and centralizes fixtures shared across test modules.
"""

import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pytest
from sklearn.datasets import load_breast_cancer

from get_data import RAW_DATA_PATH, fetch_and_save_dataset


@pytest.fixture(scope="session", autouse=True)
def ensure_dataset_exists():
    """
    Guarantees data/raw/breast_cancer.csv exists before any test runs, so the
    test suite is self-sufficient even if `python src/get_data.py` wasn't run
    manually first (mirrors what the CI workflow does explicitly).
    """
    if not os.path.exists(RAW_DATA_PATH):
        fetch_and_save_dataset()


@pytest.fixture(scope="session")
def sample_record() -> dict:
    """A single real raw record (as a dict of raw feature name -> value)."""
    dataset = load_breast_cancer(as_frame=True)
    row = dataset.frame.iloc[0].drop("target")
    return {name: float(val) for name, val in row.items()}
