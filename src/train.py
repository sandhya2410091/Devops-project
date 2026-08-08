"""
train.py
--------
End-to-end training pipeline for the Breast Cancer classification task.

Steps performed:
    1. Load the DVC-tracked dataset (data/raw/breast_cancer.csv)
    2. Preprocess: feature engineering, train/test split, scaling
    3. Train THREE candidate models:
           - Logistic Regression
           - Random Forest
           - Gradient Boosting
    4. For every run: log params, evaluation metrics, and the model artifact
       to MLflow. The fitted scaler and feature-name ordering are logged as
       artifacts alongside each model so inference can reproduce the exact
       training-time preprocessing.
    5. Compare all runs on F1-score (chosen because the dataset, while fairly
       balanced, is a medical diagnosis task where both false positives and
       false negatives matter -- F1 balances precision and recall).
    6. Register the best model in the MLflow Model Registry and promote it
       via the "production" alias, so the FastAPI service always serves the
       best-performing model.

Usage:
    python src/train.py
"""

import json
import os
import tempfile

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from utils import load_data, preprocess_and_split

MODEL_NAME = "breast_cancer_classifier"
EXPERIMENT_NAME = "breast_cancer_capstone"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")
MLFLOW_ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "mlruns")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH}")


def get_candidate_models() -> dict:
    """
    Returns the three candidate models with the hyperparameters that will be
    logged to MLflow. Using a dict keeps run naming and param logging tidy.
    """
    return {
        "logistic_regression": LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Computes the standard classification metrics used to compare models."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def log_preprocessing_artifacts(scaler, feature_names) -> None:
    """
    Logs the fitted scaler and the ordered feature-name list as MLflow
    artifacts of the *current* active run, so a served model can be paired
    with the exact preprocessing that trained it.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        scaler_path = os.path.join(tmp_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="preprocessing")

        features_path = os.path.join(tmp_dir, "feature_names.json")
        with open(features_path, "w") as f:
            json.dump(list(feature_names), f)
        mlflow.log_artifact(features_path, artifact_path="preprocessing")


def train_and_log_all_models() -> str:
    """
    Trains all candidate models, logs each to MLflow, registers the best one
    in the Model Registry under the "production" alias, and returns the
    registered model version.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.makedirs(MLFLOW_ARTIFACT_DIR, exist_ok=True)
    try:
        mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=f"file:{MLFLOW_ARTIFACT_DIR}"
        )
    except mlflow.exceptions.MlflowException:
        pass  # experiment already exists
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_and_split(df)

    results = []  # list of (run_id, f1_score, model_name)

    for model_name, model in get_candidate_models().items():
        with mlflow.start_run(run_name=model_name) as run:
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("n_train_samples", X_train.shape[0])
            mlflow.log_param("n_test_samples", X_test.shape[0])
            mlflow.log_param("n_features", X_train.shape[1])

            mlflow.sklearn.log_model(model, name="model")
            log_preprocessing_artifacts(scaler, feature_names)

            print(f"[{model_name}] run_id={run.info.run_id} metrics={metrics}")
            results.append((run.info.run_id, metrics["f1_score"], model_name))

    # Pick the best run by F1-score.
    best_run_id, best_f1, best_model_name = max(results, key=lambda r: r[1])
    print(f"\nBest model: {best_model_name} (run_id={best_run_id}, f1_score={best_f1:.4f})")

    # Register the best model in the MLflow Model Registry.
    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    # Promote it via a "production" alias so downstream consumers (the
    # FastAPI service) always load the current best model without needing to
    # know a specific version number.
    client.set_registered_model_alias(
        name=MODEL_NAME, alias="production", version=registered.version
    )
    client.set_model_version_tag(
        name=MODEL_NAME, version=registered.version, key="f1_score", value=str(best_f1)
    )
    client.set_model_version_tag(
        name=MODEL_NAME, version=registered.version, key="algorithm", value=best_model_name
    )

    print(
        f"Registered '{MODEL_NAME}' version {registered.version} "
        f"and set alias 'production' -> version {registered.version}"
    )
    return registered.version


if __name__ == "__main__":
    train_and_log_all_models()
