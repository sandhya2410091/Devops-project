"""
predict.py
----------
Inference helper layer used by the FastAPI service (app.py). Kept separate
from app.py so the model-loading / prediction logic can be unit-tested
without spinning up the web server.

Loads the model currently tagged with the "production" alias in the MLflow
Model Registry, along with the scaler and feature-name ordering that were
logged as artifacts alongside that specific training run.
"""

import json
import os
import tempfile
from typing import Tuple

import joblib
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "breast_cancer_classifier"
MODEL_ALIAS = "production"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH}")


class PredictionService:
    """
    Wraps the registered model + its matching preprocessing artifacts
    (scaler, feature ordering) behind a simple `.predict(record)` API.
    """

    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.client = MlflowClient()
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_version = None

    def load(self) -> None:
        """
        Loads the production-aliased model and its preprocessing artifacts.

        Raises:
            RuntimeError: if no model has been registered/promoted yet. This
                surfaces a clear error instead of the API silently serving
                stale or missing predictions.
        """
        try:
            model_version_info = self.client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        except Exception as exc:
            raise RuntimeError(
                f"No model found for '{MODEL_NAME}' with alias '{MODEL_ALIAS}'. "
                "Run `python src/train.py` first to train and register a model."
            ) from exc

        self.model_version = model_version_info.version
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        self.model = mlflow.sklearn.load_model(model_uri)

        run_id = model_version_info.run_id
        with tempfile.TemporaryDirectory() as tmp_dir:
            scaler_path = self.client.download_artifacts(run_id, "preprocessing/scaler.joblib", tmp_dir)
            self.scaler = joblib.load(scaler_path)

            features_path = self.client.download_artifacts(
                run_id, "preprocessing/feature_names.json", tmp_dir
            )
            with open(features_path) as f:
                self.feature_names = json.load(f)

    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    def predict(self, record: dict) -> Tuple[int, float]:
        """
        Runs inference on a single raw record.

        Args:
            record: dict of the 30 raw feature values.

        Returns:
            (predicted_class, probability_of_malignant) where predicted_class
            is 0 (benign) or 1 (malignant), matching the sklearn breast
            cancer dataset's target encoding.
        """
        if not self.is_ready():
            raise RuntimeError("PredictionService.load() must be called before predict().")

        # Local import to avoid a circular import at module load time.
        from utils import preprocess_single_record

        X = preprocess_single_record(record, self.scaler, self.feature_names)
        predicted_class = int(self.model.predict(X)[0])
        probability = float(self.model.predict_proba(X)[0][1])
        return predicted_class, probability
