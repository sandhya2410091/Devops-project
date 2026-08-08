# Breast Cancer Prediction — MLOps Capstone Pipeline

An end-to-end MLOps pipeline: **DVC-versioned data → training pipeline (3 models) → MLflow tracking & registry → FastAPI serving → Docker → GitHub Actions CI/CD**.

Dataset: **Breast Cancer Wisconsin (Diagnostic)** — a structured binary classification problem (`0` = benign, `1` = malignant), loaded via `sklearn.datasets.load_breast_cancer`.

## Project Structure

```
project/
|
|-- data/
|   `-- raw/breast_cancer.csv      # DVC-tracked dataset (generated, not committed)
|-- models/                        # reserved output dir (models live in MLflow registry)
|-- src/
|   |-- get_data.py                # fetches & saves the raw dataset
|   |-- utils.py                   # shared preprocessing / feature engineering
|   |-- train.py                   # trains 3 models, logs to MLflow, registers best
|   |-- predict.py                 # loads production model, runs inference
|   `-- app.py                     # FastAPI service (/predict)
|
|-- tests/                         # pytest suite (18 tests)
|-- .github/workflows/ci.yml       # CI: checkout -> setup -> install -> test -> docker build
|-- Dockerfile
|-- requirements.txt
|-- dvc.yaml                       # DVC pipeline: prepare_data stage
|-- README.md
`-- .gitignore
```

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 2. Data Versioning with DVC

```bash
git init                      # DVC requires a Git repo
dvc init

# Point DVC at storage for the versioned data. Swap the local path for a
# real remote (S3, GCS, Azure, GDrive) in a production setting, e.g.:
#   dvc remote add -d storage s3://my-bucket/dvc-store
dvc remote add -d localstorage ../dvc-storage

dvc repro          # runs `python src/get_data.py`, tracks data/raw/breast_cancer.csv
dvc push           # uploads the tracked data to the remote

git add dvc.yaml dvc.lock .dvc/config .gitignore
git commit -m "Track dataset with DVC"
```


## 3. ML Pipeline + MLflow Tracking & Registry

```bash
cd src
python train.py
```

This trains **three models** — Logistic Regression, Random Forest, Gradient Boosting — on an 80/20 stratified split of engineered + scaled features. For every run it logs:
- **Parameters** (all hyperparameters, sample counts, feature count)
- **Metrics** (accuracy, precision, recall, F1-score, ROC-AUC)
- **Model artifact** (via `mlflow.sklearn.log_model`)
- **Preprocessing artifacts** (fitted `StandardScaler`, feature-name ordering) so inference reproduces training-time preprocessing exactly

The run with the best **F1-score** is registered in the **MLflow Model Registry** as `breast_cancer_classifier` and promoted with the `production` alias.

Launch the MLflow UI to compare experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://localhost:5000
```


## 4. Prediction API (FastAPI)

```bash
cd src
uvicorn app:app --host 0.0.0.0 --port 8000
```

- `GET /health` — readiness probe
- `POST /predict` — accepts the 30 raw feature values, returns the prediction

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mean radius": 17.99, "mean texture": 10.38, "mean perimeter": 122.8,
    "mean area": 1001.0, "mean smoothness": 0.1184, "mean compactness": 0.2776,
    "mean concavity": 0.3001, "mean concave points": 0.1471, "mean symmetry": 0.2419,
    "mean fractal dimension": 0.07871, "radius error": 1.095, "texture error": 0.9053,
    "perimeter error": 8.589, "area error": 153.4, "smoothness error": 0.006399,
    "compactness error": 0.04904, "concavity error": 0.05373, "concave points error": 0.01587,
    "symmetry error": 0.03003, "fractal dimension error": 0.006193, "worst radius": 25.38,
    "worst texture": 17.33, "worst perimeter": 184.6, "worst area": 2019.0,
    "worst smoothness": 0.1622, "worst compactness": 0.6656, "worst concavity": 0.7119,
    "worst concave points": 0.2654, "worst symmetry": 0.4601, "worst fractal dimension": 0.1189
  }'
```

Response:
```json
{
  "prediction": 1,
  "diagnosis": "malignant",
  "probability_malignant": 0.97,
  "model_version": "1"
}
```




## 5. Tests

```bash
pytest tests/ -v
```

18 tests across `test_utils.py`, `test_train.py`, `test_predict.py`, and `test_api.py` cover data loading, feature engineering, preprocessing correctness, model evaluation, registry loading, and the API contract (including validation errors).

## 6. Docker

The image trains and registers the model **during the build**, so it is fully self-contained.

```bash
docker build -t breast-cancer-prediction-api .
docker run -p 8000:8000 breast-cancer-prediction-api
curl http://localhost:8000/health
```


## 7. CI/CD — GitHub Actions

`.github/workflows/ci.yml` runs on every push/PR:
1. Checkout repository
2. Set up Python 3.11
3. Install dependencies
4. Generate dataset + train/register model
5. Run `pytest`
6. Build the Docker image and smoke-test `/health`




