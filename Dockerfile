# -----------------------------------------------------------------------------
# Dockerfile - Breast Cancer Prediction API
#
# The image contains:
#   - The FastAPI application (src/app.py, src/predict.py, src/utils.py)
#   - All required dependencies (requirements.txt)
#   - A registered model: training runs DURING the image build, so the
#     resulting image already ships with a populated MLflow registry
#     (mlflow.db + mlruns/) containing the "production"-aliased model.
#   - Startup configuration (CMD runs uvicorn, HEALTHCHECK verifies readiness)
# -----------------------------------------------------------------------------

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed to build some scientific-python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (leverages Docker layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and pipeline code.
COPY src/ ./src/
COPY dvc.yaml .

# Generate the dataset and train + register the model AT BUILD TIME, so the
# image is self-contained and requires no external services to serve
# predictions once started.
RUN python src/get_data.py && python src/train.py

# Non-root user for defense-in-depth.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

WORKDIR /app/src
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
