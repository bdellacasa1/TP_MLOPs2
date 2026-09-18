import os
import sys
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import gdown

# Añadimos la raíz del proyecto al PYTHONPATH para poder importar src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_DIR,
    ZIP_FILE,
    GOOGLE_DRIVE_FILE_ID,
)

MINIO_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
BUCKET_NAME = os.getenv("MLFLOW_BUCKET_NAME", "mlflow")
OBJECT_NAME = "raw/data.zip"

def seed_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Conectando a MinIO en: {MINIO_ENDPOINT}...")
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name="us-east-1",
    )
    
    try:
        # Verificar si el archivo ya existe en MinIO
        s3.head_object(Bucket=BUCKET_NAME, Key=OBJECT_NAME)
        print("✅ El dataset ya se encuentra subido en MinIO.")
        return
    except ClientError as e:
        # Si el error no es 404 (No Encontrado), lanzar la excepción
        if e.response['Error']['Code'] != '404':
            print(f"❌ Error al consultar MinIO: {e}")
            raise e
        
        print("🔍 El dataset no se encuentra en MinIO.")
        
        # Verificar si el archivo local data.zip ya existe
        if not ZIP_FILE.exists():
            print(f"📥 data.zip no encontrado en {ZIP_FILE}. Descargándolo desde Google Drive...")
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            try:
                gdown.download(url, str(ZIP_FILE), quiet=False)
                print("✅ Descarga completa.")
            except Exception as exc:
                print(f"❌ Falla al descargar desde Google Drive usando gdown: {exc}")
                print("Por favor, asegúrate de colocar manualmente data.zip en data/raw/data.zip")
                return
        else:
            print(f"✅ data.zip encontrado localmente en {ZIP_FILE}")
            
        if ZIP_FILE.exists():
            print(f"🚀 Subiendo {ZIP_FILE} a MinIO como '{OBJECT_NAME}' en el bucket '{BUCKET_NAME}'...")
            try:
                s3.upload_file(str(ZIP_FILE), BUCKET_NAME, OBJECT_NAME)
                print("✅ Dataset subido exitosamente a MinIO.")
            except Exception as exc:
                print(f"❌ Error al subir el archivo a MinIO: {exc}")
        else:
            print("❌ No se pudo proceder debido a la ausencia del archivo local data.zip")

if __name__ == "__main__":
    seed_data()
