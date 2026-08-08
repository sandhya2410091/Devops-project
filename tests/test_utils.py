"""
test_utils.py
-------------
Unit tests for src/utils.py: data loading, feature engineering, and the
train/test preprocessing pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from utils import (
    ENGINEERED_FEATURES,
    engineer_features,
    get_feature_columns,
    load_data,
    preprocess_and_split,
    preprocess_single_record,
    split_features_target,
)


def test_load_data_returns_dataframe_with_expected_shape():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 569  # known size of the Breast Cancer Wisconsin dataset
    assert "diagnosis" in df.columns


def test_load_data_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_data(str(missing_path))


def test_engineer_features_adds_expected_columns():
    df = load_data()
    engineered = engineer_features(df)
    for col in ENGINEERED_FEATURES:
        assert col in engineered.columns
    # Original columns must still be present.
    for col in df.columns:
        assert col in engineered.columns


def test_split_features_target_shapes():
    df = load_data()
    X, y = split_features_target(df)
    assert "diagnosis" not in X.columns
    assert len(y) == len(df)
    assert set(y.unique()) == {0, 1}


def test_get_feature_columns_excludes_target():
    df = engineer_features(load_data())
    cols = get_feature_columns(df)
    assert "diagnosis" not in cols
    assert len(cols) == df.shape[1] - 1


def test_preprocess_and_split_shapes_and_scaling():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_and_split(
        df, test_size=0.2, random_state=42
    )

    n_total = df.shape[0]
    assert X_train.shape[0] + X_test.shape[0] == n_total
    assert X_train.shape[1] == len(feature_names)
    assert X_test.shape[1] == len(feature_names)
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]

    # Scaled training data should have ~zero mean and ~unit variance per feature.
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(X_train.std(axis=0), 1, atol=1e-6)


def test_preprocess_single_record_matches_feature_order(sample_record):
    df = load_data()
    _, _, _, _, scaler, feature_names = preprocess_and_split(df)

    X = preprocess_single_record(sample_record, scaler, feature_names)
    assert X.shape == (1, len(feature_names))
