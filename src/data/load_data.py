import os
import zipfile
import boto3
from botocore.exceptions import ClientError
import gdown
import pandas as pd

from src.config import (
    GOOGLE_DRIVE_FILE_ID,
    RAW_DATA_DIR,
    TRAIN_FILE,
    TEST_FILE,
    ZIP_FILE,
    TARGET_COL,
    TARGET_MAPPING,
)


def download_dataset(force_download: bool = False) -> None:
    """
    Descarga y extrae el dataset si todavía no existe.
    Prioriza el uso de caché local y de almacenamiento interno (MinIO).
    """

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        TRAIN_FILE.exists()
        and TEST_FILE.exists()
        and not force_download
    ):
        print("✅ Dataset ya disponible en caché local.")
        return

    # 1. Intentar descarga interna desde MinIO
    minio_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    aws_key = os.getenv("AWS_ACCESS_KEY_ID", "minio")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
    bucket_name = os.getenv("MLFLOW_BUCKET_NAME", "mlflow")
    object_name = "raw/data.zip"

    minio_success = False
    print(f"🔄 Intentando descargar dataset desde MinIO ({minio_endpoint})...")
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=minio_endpoint,
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name="us-east-1",
        )
        s3.download_file(bucket_name, object_name, str(ZIP_FILE))
        print("✅ Dataset descargado exitosamente de MinIO.")
        minio_success = True
    except Exception as exc:
        print(f"⚠️ No se pudo descargar desde MinIO: {exc}")

    # 2. Fallback a Google Drive si falló la descarga de MinIO
    if not minio_success:
        print("📥 Iniciando descarga fallback desde Google Drive...")
        url = (
            "https://drive.google.com/uc"
            f"?id={GOOGLE_DRIVE_FILE_ID}"
        )
        try:
            gdown.download(url, str(ZIP_FILE), quiet=False, fuzzy=True)
            print("✅ Dataset descargado correctamente de Google Drive.")
        except Exception as exc:
            print(f"❌ Error crítico: Falló también la descarga de fallback: {exc}")
            raise RuntimeError("No se pudo obtener el dataset por ningún medio.") from exc

    # 3. Extracción del zip
    print("📂 Extrayendo dataset...")
    try:
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print("✅ Extracción completada.")
    except Exception as exc:
        print(f"❌ Error al descomprimir {ZIP_FILE}: {exc}")
        raise exc


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga train.csv y test.csv.
    """

    download_dataset()

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    return train_df, test_df


def clean_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Realiza la limpieza básica utilizada
    en la notebook original.
    """

    df = df.copy()

    df = df.drop(
        columns=["Unnamed: 0", "id"],
        errors="ignore",
    )

    return df


def encode_target(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convierte satisfaction a 0/1.
    """

    df = df.copy()

    df[TARGET_COL] = (
        df[TARGET_COL]
        .map(TARGET_MAPPING)
    )

    return df


def load_and_prepare_data():
    """
    Pipeline completo de carga y limpieza.
    """

    train_df, test_df = load_dataset()

    train_df = clean_dataset(train_df)
    test_df = clean_dataset(test_df)

    train_df = encode_target(train_df)
    test_df = encode_target(test_df)

    return train_df, test_df


if __name__ == "__main__":

    train_df, test_df = load_and_prepare_data()

    print("Train:", train_df.shape)
    print("Test:", test_df.shape)

    print("\nTarget train:")
    print(
        train_df[TARGET_COL]
        .value_counts()
    )