"""
test_api.py
-----------
Integration tests for the FastAPI service defined in src/app.py, exercised
through Starlette's TestClient (no real network socket needed). Requires
that a production model has already been registered.
"""

import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["model_ready"] is True


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_valid_input(client, sample_record):
    response = client.post("/predict", json=sample_record)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["diagnosis"] in ("benign", "malignant")
    assert 0.0 <= body["probability_malignant"] <= 1.0
    assert "model_version" in body


def test_predict_endpoint_missing_field_returns_422(client, sample_record):
    incomplete = dict(sample_record)
    del incomplete["mean radius"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_endpoint_invalid_type_returns_422(client, sample_record):
    bad_payload = dict(sample_record)
    bad_payload["mean radius"] = "not_a_number"
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
