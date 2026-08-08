"""
test_predict.py
----------------
Tests for src/predict.py: PredictionService loading and single-record
inference. Requires that a production model has already been registered
(the CI workflow runs `python src/train.py` before pytest for this reason).
"""

import pytest

from predict import PredictionService


@pytest.fixture(scope="module")
def loaded_service() -> PredictionService:
    svc = PredictionService()
    svc.load()
    return svc


def test_service_not_ready_before_load():
    svc = PredictionService()
    assert svc.is_ready() is False


def test_service_ready_after_load(loaded_service):
    assert loaded_service.is_ready() is True
    assert loaded_service.model_version is not None


def test_predict_raises_if_not_loaded(sample_record):
    svc = PredictionService()
    with pytest.raises(RuntimeError):
        svc.predict(sample_record)


def test_predict_returns_valid_output(loaded_service, sample_record):
    predicted_class, probability = loaded_service.predict(sample_record)
    assert predicted_class in (0, 1)
    assert 0.0 <= probability <= 1.0
