"""Contrato del paso 1: clave, /v1, registro y checksum.

No llama a MLflow. El modelo está reemplazado en memoria.
"""

import json
import os
import tempfile
from pathlib import Path

os.environ["API_KEY"] = "test-key"
os.environ["PREDICTION_LOG_PATH"] = str(
    Path(tempfile.mkdtemp()) / "predictions.jsonl"
)
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:9"

from fastapi.testclient import TestClient

from api.main import app
from api.model_service import (
    ArtifactChecksumError,
    enforce_artifact_checksum,
    model_service,
    sha256_tree,
)
from api.prediction_log import prediction_log_path


VALID = {
    "Gender": "Male",
    "Customer Type": "Loyal Customer",
    "Type of Travel": "Business travel",
    "Class": "Business",
    "Age": 41,
    "Flight Distance": 600,
    "Departure Delay in Minutes": 10,
    "Arrival Delay in Minutes": 10,
}

HEADERS = {"X-API-Key": "test-key"}


class FakeClient:
    def __init__(self):
        self.tags = {}

    def set_model_version_tag(self, name, version, key, value):
        self.tags[(name, str(version), key)] = value


def test_http_contract():
    model_service.reload_model = lambda: {
        "previous_version": "7",
        "model_version": "7",
        "changed": False,
    }
    model_service.predict = lambda features: {
        "prediction": 1,
        "label": "satisfied",
        "probability": 0.8,
        "model_name": "airline_satisfaction_without_scores",
        "model_alias": "champion",
        "model_version": "7",
    }
    model_service._model = object()
    model_service._version = "7"

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        missing = client.post("/v1/predict", json=VALID)
        assert missing.status_code == 401

        old_missing = client.post("/predict", json=VALID)
        assert old_missing.status_code == 401

        invalid = dict(VALID)
        invalid["Age"] = 200
        rejected = client.post("/v1/predict", json=invalid, headers=HEADERS)
        assert rejected.status_code == 422

        ok = client.post("/v1/predict", json=VALID, headers=HEADERS)
        assert ok.status_code == 200
        body = ok.json()
        assert body["model_version"] == "7"
        assert body["probability"] == 0.8

        legacy = client.post("/predict", json=VALID, headers=HEADERS)
        assert legacy.status_code == 200

        reload_missing = client.post("/v1/reload-model")
        assert reload_missing.status_code == 401

        reloaded = client.post("/v1/reload-model", headers=HEADERS)
        assert reloaded.status_code == 200
        assert reloaded.json()["model_version"] == "7"

    lines = prediction_log_path().read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["features"]["Age"] == 41
    assert first["model_version"] == "7"
    assert first["probability"] == 0.8
    assert "timestamp" in first


def test_checksum():
    directory = Path(tempfile.mkdtemp())
    (directory / "b.txt").write_text("dos", encoding="utf-8")
    (directory / "a.txt").write_text("uno", encoding="utf-8")
    checksum = sha256_tree(directory)
    assert checksum == sha256_tree(directory)

    client = FakeClient()
    enforce_artifact_checksum(client, "modelo", 7, checksum, {})
    assert client.tags[("modelo", "7", "artifact_sha256")] == checksum

    enforce_artifact_checksum(
        client,
        "modelo",
        7,
        checksum,
        {"artifact_sha256": checksum},
    )

    try:
        enforce_artifact_checksum(
            client,
            "modelo",
            7,
            "otro",
            {"artifact_sha256": checksum},
        )
    except ArtifactChecksumError:
        return
    raise AssertionError("un checksum distinto tendría que rechazarse")


if __name__ == "__main__":
    test_checksum()
    test_http_contract()
    print("ok")
