import hashlib
import os
import threading
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient


CHECKSUM_TAG = "artifact_sha256"


class ArtifactChecksumError(RuntimeError):
    """The champion file does not match the checksum stored in MLflow."""


class ModelService:

    def __init__(self):

        self.tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://mlflow:5000",
        )

        self.model_name = os.getenv(
            "MODEL_NAME",
            "airline_satisfaction_without_scores",
        )

        self.model_alias = os.getenv(
            "MODEL_ALIAS",
            "champion",
        )

        mlflow.set_tracking_uri(
            self.tracking_uri
        )

        self.client = MlflowClient()

        self._model = None
        self._version = None
        self._checksum = None

        self._lock = threading.RLock()

    def reload_model(self):
        """
        Consulta qué versión tiene actualmente el alias
        champion y carga ese modelo desde MLflow.

        El checksum del artefacto se guarda como tag de esa
        versión. Si el tag ya existe y no coincide, el modelo
        en memoria no se reemplaza.
        """

        model_version = (
            self.client.get_model_version_by_alias(
                self.model_name,
                self.model_alias,
            )
        )

        model_uri = (
            f"models:/{self.model_name}"
            f"@{self.model_alias}"
        )

        print(
            f"Cargando modelo: {model_uri}"
        )

        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri
        )
        checksum = sha256_tree(local_path)
        enforce_artifact_checksum(
            self.client,
            self.model_name,
            model_version.version,
            checksum,
            dict(model_version.tags or {}),
        )

        new_model = mlflow.sklearn.load_model(
            local_path
        )

        with self._lock:

            previous_version = self._version

            self._model = new_model
            self._version = str(
                model_version.version
            )
            self._checksum = checksum

        print(
            f"Modelo cargado. "
            f"Version: {self._version}"
        )

        return {
            "previous_version": previous_version,
            "model_version": self._version,
            "changed": (
                previous_version
                != self._version
            ),
        }

    def predict(
        self,
        features: dict,
    ):
        """
        Realiza una predicción utilizando el champion
        actualmente cargado en memoria.
        """

        with self._lock:

            model = self._model
            version = self._version

        if model is None:
            raise RuntimeError(
                "El modelo no está cargado."
            )

        input_df = pd.DataFrame(
            [features]
        )

        prediction = int(
            model.predict(
                input_df
            )[0]
        )

        probability = float(
            model.predict_proba(
                input_df
            )[0, 1]
        )

        label = (
            "satisfied"
            if prediction == 1
            else "neutral or dissatisfied"
        )

        return {
            "prediction": prediction,
            "label": label,
            "probability": probability,
            "model_name": self.model_name,
            "model_alias": self.model_alias,
            "model_version": version,
        }

    def health(self):

        return {
            "status": (
                "ok"
                if self._model is not None
                else "model_not_loaded"
            ),
            "model_name": self.model_name,
            "model_alias": self.model_alias,
            "model_version": self._version,
            "artifact_sha256": self._checksum,
        }


def sha256_tree(root: str) -> str:
    """Stable checksum of a file or of every file under a directory."""

    base = Path(root)
    digest = hashlib.sha256()
    if base.is_file():
        digest.update(base.read_bytes())
        return digest.hexdigest()

    files = sorted(
        path for path in base.rglob("*") if path.is_file()
    )
    for path in files:
        rel = path.relative_to(base).as_posix().encode()
        digest.update(rel)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def enforce_artifact_checksum(
    client,
    model_name: str,
    version,
    checksum: str,
    tags: dict,
) -> None:
    """Register the checksum the first time. Reject a later mismatch."""

    stored = (tags or {}).get(CHECKSUM_TAG)
    if not stored:
        client.set_model_version_tag(
            model_name,
            str(version),
            CHECKSUM_TAG,
            checksum,
        )
        return

    if stored != checksum:
        raise ArtifactChecksumError(
            "El artefacto del champion no coincide con el "
            "checksum registrado. El modelo en memoria no se reemplazó."
        )


model_service = ModelService()