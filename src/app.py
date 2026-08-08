"""
app.py
------
FastAPI prediction service for the Breast Cancer classifier.

On startup, it loads the model currently promoted to the "production" alias
in the MLflow Model Registry (see predict.py / train.py), along with the
scaler that was fit during training.

Endpoints:
    GET  /            -> basic service info
    GET  /health       -> readiness probe (used by Docker healthcheck)
    POST /predict      -> runs inference on a single record of the 30 raw
                           Breast Cancer Wisconsin measurements

Run locally:
    uvicorn src.app:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from predict import PredictionService

prediction_service = PredictionService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the registered production model once, so every request
    # reuses the already-loaded model instead of reloading it per call.
    prediction_service.load()
    yield
    # No teardown resources to release.


app = FastAPI(
    title="Breast Cancer Prediction API",
    description="Serves predictions from the MLflow-registered production model.",
    version="1.0.0",
    lifespan=lifespan,
)


class BreastCancerFeatures(BaseModel):
    """
    The 30 raw diagnostic measurements from the Breast Cancer Wisconsin
    dataset. Field aliases match the original dataset column names (which
    contain spaces), while the Python attribute names use underscores for
    valid identifiers.
    """

    model_config = ConfigDict(populate_by_name=True)

    mean_radius: float = Field(..., alias="mean radius")
    mean_texture: float = Field(..., alias="mean texture")
    mean_perimeter: float = Field(..., alias="mean perimeter")
    mean_area: float = Field(..., alias="mean area")
    mean_smoothness: float = Field(..., alias="mean smoothness")
    mean_compactness: float = Field(..., alias="mean compactness")
    mean_concavity: float = Field(..., alias="mean concavity")
    mean_concave_points: float = Field(..., alias="mean concave points")
    mean_symmetry: float = Field(..., alias="mean symmetry")
    mean_fractal_dimension: float = Field(..., alias="mean fractal dimension")

    radius_error: float = Field(..., alias="radius error")
    texture_error: float = Field(..., alias="texture error")
    perimeter_error: float = Field(..., alias="perimeter error")
    area_error: float = Field(..., alias="area error")
    smoothness_error: float = Field(..., alias="smoothness error")
    compactness_error: float = Field(..., alias="compactness error")
    concavity_error: float = Field(..., alias="concavity error")
    concave_points_error: float = Field(..., alias="concave points error")
    symmetry_error: float = Field(..., alias="symmetry error")
    fractal_dimension_error: float = Field(..., alias="fractal dimension error")

    worst_radius: float = Field(..., alias="worst radius")
    worst_texture: float = Field(..., alias="worst texture")
    worst_perimeter: float = Field(..., alias="worst perimeter")
    worst_area: float = Field(..., alias="worst area")
    worst_smoothness: float = Field(..., alias="worst smoothness")
    worst_compactness: float = Field(..., alias="worst compactness")
    worst_concavity: float = Field(..., alias="worst concavity")
    worst_concave_points: float = Field(..., alias="worst concave points")
    worst_symmetry: float = Field(..., alias="worst symmetry")
    worst_fractal_dimension: float = Field(..., alias="worst fractal dimension")

    def to_raw_dict(self) -> dict:
        """Returns a dict keyed by the ORIGINAL dataset column names (with spaces),
        which is what utils.preprocess_single_record expects."""
        return self.model_dump(by_alias=True)


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = benign, 1 = malignant")
    diagnosis: str
    probability_malignant: float
    model_version: str


@app.get("/")
def root():
    return {
        "service": "Breast Cancer Prediction API",
        "status": "running",
        "model_ready": prediction_service.is_ready(),
    }


@app.get("/health")
def health():
    if not prediction_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_version": prediction_service.model_version}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: BreastCancerFeatures):
    if not prediction_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predicted_class, probability_malignant = prediction_service.predict(features.to_raw_dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictionResponse(
        prediction=predicted_class,
        diagnosis="malignant" if predicted_class == 1 else "benign",
        probability_malignant=probability_malignant,
        model_version=str(prediction_service.model_version),
    )
