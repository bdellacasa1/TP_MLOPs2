import os
import threading

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient


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

        self._lock = threading.RLock()

    def reload_model(self):
        """
        Consulta qué versión tiene actualmente el alias
        champion y carga ese modelo desde MLflow.
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

        new_model = mlflow.sklearn.load_model(
            model_uri
        )

        with self._lock:

            previous_version = self._version

            self._model = new_model
            self._version = str(
                model_version.version
            )

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
        }


model_service = ModelService()