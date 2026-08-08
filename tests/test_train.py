"""
test_train.py
--------------
Tests for src/train.py: candidate model construction and evaluation metrics.
These tests deliberately avoid calling train_and_log_all_models() directly
(that is exercised as an explicit CI step via `python src/train.py`, since it
trains 3 models and writes to the MLflow registry -- an integration-level
action rather than a fast unit test).
"""

from sklearn.linear_model import LogisticRegression

from train import evaluate, get_candidate_models
from utils import load_data, preprocess_and_split


def test_get_candidate_models_returns_three_models():
    models = get_candidate_models()
    assert len(models) == 3
    assert "logistic_regression" in models
    assert "random_forest" in models
    assert "gradient_boosting" in models
    assert isinstance(models["logistic_regression"], LogisticRegression)


def test_evaluate_returns_expected_metric_keys():
    df = load_data()
    X_train, X_test, y_train, y_test, _, _ = preprocess_and_split(df)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)

    expected_keys = {"accuracy", "precision", "recall", "f1_score", "roc_auc"}
    assert set(metrics.keys()) == expected_keys
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
