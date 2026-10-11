"""Filas del test para el cierre de vuelos.

El archivo no trae un id de vuelo. El id es la posición de la fila.
La demora de llegada vacía se publica como 0: el contrato del champion
no acepta nulo, y el experimento en memoria ya imputaba ese valor.
"""

import os
from pathlib import Path

import pandas as pd

TEST_FILE = Path(os.getenv("TEST_FILE", "/data/raw/test.csv"))
TOPIC = os.getenv("KAFKA_TOPIC", "flight-close")
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "redpanda:9092")
WINDOW = int(os.getenv("WINDOW", "2000"))
DRIFT_DROP = float(os.getenv("DRIFT_DROP", "0.05"))

COLUMNS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]


def load_rows() -> pd.DataFrame:
    if not TEST_FILE.is_file():
        raise SystemExit(f"No está el test en {TEST_FILE}")

    frame = pd.read_csv(TEST_FILE)
    missing = [name for name in COLUMNS if name not in frame.columns]
    if missing:
        raise SystemExit(f"Al test le faltan columnas: {missing}")

    rows = frame.loc[:, COLUMNS].copy()
    rows["row_id"] = rows.index.astype(int)
    empty_arrival = int(rows["Arrival Delay in Minutes"].isna().sum())
    rows["Arrival Delay in Minutes"] = rows["Arrival Delay in Minutes"].fillna(0)
    rows = rows.sort_values(
        ["Arrival Delay in Minutes", "row_id"],
        kind="mergesort",
    )
    rows.attrs["empty_arrival"] = empty_arrival
    return rows


def as_message(row: pd.Series) -> dict:
    return {
        "row_id": int(row["row_id"]),
        "Gender": str(row["Gender"]),
        "Customer Type": str(row["Customer Type"]),
        "Type of Travel": str(row["Type of Travel"]),
        "Class": str(row["Class"]),
        "Age": int(row["Age"]),
        "Flight Distance": int(row["Flight Distance"]),
        "Departure Delay in Minutes": float(row["Departure Delay in Minutes"]),
        "Arrival Delay in Minutes": float(row["Arrival Delay in Minutes"]),
    }
