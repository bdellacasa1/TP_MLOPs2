"""Lee el cierre de vuelos y puntúa cada pasajero por gRPC.

Cada 2000 predicciones compara la media con la primera ventana.
Si cae 0,05, imprime la alerta. Un id ya puntuado en esta corrida no se vuelve a contar.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grpc
from kafka import KafkaConsumer

from api import scoring_pb2, scoring_pb2_grpc
from scripts.flight_close_rows import (
    BOOTSTRAP,
    DRIFT_DROP,
    TOPIC,
    WINDOW,
    load_rows,
)

GRPC_TARGET = os.getenv("GRPC_TARGET", "grpc:50051")
API_KEY = os.getenv("API_KEY", "")
GROUP = os.getenv("KAFKA_GROUP", "flight-close")


def score(stub, message: dict) -> scoring_pb2.Prediction:
    passenger = scoring_pb2.Passenger(
        gender=message["Gender"],
        customer_type=message["Customer Type"],
        type_of_travel=message["Type of Travel"],
        travel_class=message["Class"],
        age=int(message["Age"]),
        flight_distance=int(message["Flight Distance"]),
        departure_delay=float(message["Departure Delay in Minutes"]),
        arrival_delay=float(message["Arrival Delay in Minutes"]),
    )
    return stub.Predict(
        passenger,
        metadata=(("x-api-key", API_KEY),),
        timeout=30,
    )


def main() -> None:
    if not API_KEY:
        raise SystemExit("API_KEY no está configurada.")

    expected = int(load_rows().shape[0])
    channel = grpc.insecure_channel(GRPC_TARGET)
    stub = scoring_pb2_grpc.ScoringStub(channel)
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        group_id=GROUP,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=20000,
        value_deserializer=lambda raw: json.loads(raw.decode()),
    )

    seen: set[int] = set()
    window: list[float] = []
    windows = 0
    alerts = 0
    duplicates = 0
    first_mean = None
    last_mean = None
    version = ""

    try:
        for record in consumer:
            message = record.value
            row_id = int(message["row_id"])
            if row_id in seen:
                duplicates += 1
                consumer.commit()
                continue

            prediction = score(stub, message)
            seen.add(row_id)
            version = prediction.model_version
            window.append(float(prediction.probability))
            if len(window) == WINDOW:
                windows += 1
                mean_p = sum(window) / WINDOW
                if first_mean is None:
                    first_mean = mean_p
                last_mean = mean_p
                drop = first_mean - mean_p
                print(
                    f"ventana={windows} media={mean_p:.4f} "
                    f"primera={first_mean:.4f} caida={drop:.4f}",
                    flush=True,
                )
                if drop >= DRIFT_DROP:
                    alerts += 1
                    print(
                        f"alerta ventana={windows} media={mean_p:.4f} "
                        f"umbral_caida={DRIFT_DROP}",
                        flush=True,
                    )
                window = []
            consumer.commit()
            if len(seen) >= expected:
                break
    finally:
        consumer.close()
        channel.close()

    remainder = len(window)
    print(f"eventos={len(seen)}", flush=True)
    print(f"duplicados={duplicates}", flush=True)
    print(f"eventos_fuera_de_ventana={remainder}", flush=True)
    print(f"modelo_version={version}", flush=True)
    if first_mean is None:
        raise SystemExit("No se completó ninguna ventana.")
    print(f"proba_primera_ventana={first_mean:.4f}", flush=True)
    print(f"proba_ultima_ventana={last_mean:.4f}", flush=True)
    print(
        f"alertas_drift={alerts} umbral_caida={DRIFT_DROP} ventana={WINDOW}",
        flush=True,
    )
    if len(seen) != expected:
        raise SystemExit(
            f"Se puntuaron {len(seen)} filas y el test tiene {expected}."
        )


if __name__ == "__main__":
    main()
