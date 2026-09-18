from pathlib import Path
import os


# Rutas

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
ZIP_FILE = RAW_DATA_DIR / "data.zip"


# Dataset

GOOGLE_DRIVE_FILE_ID = "1zcn7uJZIukCo5Y9GLfLdHih9_S6nTzvQ"


# Target

TARGET_COL = "satisfaction"

TARGET_MAPPING = {
    "neutral or dissatisfied": 0,
    "satisfied": 1,
}


# Variables categóricas

CAT_COLS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]


# Variables numéricas base

NUM_BASE_COLS = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

FEATURES_WITHOUT_SCORES = CAT_COLS + NUM_BASE_COLS

# MLflow

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000",
)

MLFLOW_EXPERIMENT_NAME = "tp_airline_satisfaction"

REGISTRY_WITHOUT_SCORES = "airline_satisfaction_without_scores"

# Modelo

RANDOM_STATE = 42