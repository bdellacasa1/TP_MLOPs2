"""Copia al data lake lo que el entrenamiento ya produjo.

`raw/data.zip` no se toca: de ahí sigue leyendo `load_data.py`.
La API sigue cargando el champion desde MLflow. `models/` es una copia.
"""

import os
from io import BytesIO
from pathlib import Path

import boto3
import mlflow
import pandas as pd

from src.config import (
    FEATURES_WITHOUT_SCORES,
    REGISTRY_WITHOUT_SCORES,
    TARGET_COL,
)

CURATED_TRAIN_KEY = "curated/train.parquet"
CURATED_TEST_KEY = "curated/test.parquet"


def _bucket_name() -> str:
    return os.getenv("MLFLOW_BUCKET_NAME", "mlflow")


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv(
            "MLFLOW_S3_ENDPOINT_URL",
            "http://minio:9000",
        ),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
        aws_secret_access_key=os.getenv(
            "AWS_SECRET_ACCESS_KEY",
            "minio123",
        ),
        region_name="us-east-1",
    )


def _table(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    table = features.loc[:, FEATURES_WITHOUT_SCORES].copy()
    table[TARGET_COL] = target.to_numpy()
    return table


def _put_parquet(client, key: str, table: pd.DataFrame) -> None:
    buffer = BytesIO()
    table.to_parquet(buffer, index=False)
    client.put_object(
        Bucket=_bucket_name(),
        Key=key,
        Body=buffer.getvalue(),
    )
    print(f"Lake: {key} ({len(table)} filas)")


def publish_curated_tables(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Sube la tabla de las ocho columnas y el objetivo, en train y en test."""

    client = s3_client()
    _put_parquet(client, CURATED_TRAIN_KEY, _table(X_train, y_train))
    _put_parquet(client, CURATED_TEST_KEY, _table(X_test, y_test))


def publish_model_copy(version) -> None:
    """Copia el artefacto de una versión ya registrada. No mueve el alias."""

    model_uri = f"models:/{REGISTRY_WITHOUT_SCORES}/{version}"
    local_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri))
    client = s3_client()
    bucket = _bucket_name()
    prefix = f"models/v{version}"

    files = [local_path] if local_path.is_file() else [
        path for path in local_path.rglob("*") if path.is_file()
    ]
    for path in files:
        relative = path.name if local_path.is_file() else path.relative_to(local_path).as_posix()
        key = f"{prefix}/{relative}"
        client.upload_file(str(path), bucket, key)
        print(f"Lake: {key}")


def publish_training_outputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    version,
) -> None:
    publish_curated_tables(X_train, X_test, y_train, y_test)
    publish_model_copy(version)


def publish_current_champion() -> None:
    """Publica el champion que ya está en MLflow. No entrena."""

    from mlflow import MlflowClient

    from src.data.load_data import load_and_prepare_data
    from src.models.preprocessing import prepare_features
    from src.tracking.mlflow_utils import setup_mlflow

    setup_mlflow()
    train_df, test_df = load_and_prepare_data()
    X_train, X_test, y_train, y_test = prepare_features(train_df, test_df)
    champion = MlflowClient().get_model_version_by_alias(
        REGISTRY_WITHOUT_SCORES,
        "champion",
    )
    publish_training_outputs(
        X_train,
        X_test,
        y_train,
        y_test,
        champion.version,
    )


if __name__ == "__main__":
    publish_current_champion()
