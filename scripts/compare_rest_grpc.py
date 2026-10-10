"""Mismo pasajero por REST y por gRPC. Imprime las dos predicciones y los tiempos."""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grpc

from api import scoring_pb2, scoring_pb2_grpc

REST_URL = os.getenv("REST_URL", "http://api:8000")
GRPC_TARGET = os.getenv("GRPC_TARGET", "127.0.0.1:50051")
API_KEY = os.getenv("API_KEY", "")

PASSENGER = {
    "Gender": "Male",
    "Customer Type": "Loyal Customer",
    "Type of Travel": "Business travel",
    "Class": "Business",
    "Age": 41,
    "Flight Distance": 600,
    "Departure Delay in Minutes": 10,
    "Arrival Delay in Minutes": 10,
}


def call_rest() -> tuple[dict, float]:
    body = json.dumps(PASSENGER).encode()
    request = urllib.request.Request(
        f"{REST_URL}/v1/predict",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": API_KEY,
        },
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode())
    elapsed = time.perf_counter() - started
    return payload, elapsed


def call_grpc() -> tuple[scoring_pb2.Prediction, float]:
    channel = grpc.insecure_channel(GRPC_TARGET)
    stub = scoring_pb2_grpc.ScoringStub(channel)
    message = scoring_pb2.Passenger(
        gender=PASSENGER["Gender"],
        customer_type=PASSENGER["Customer Type"],
        type_of_travel=PASSENGER["Type of Travel"],
        travel_class=PASSENGER["Class"],
        age=PASSENGER["Age"],
        flight_distance=PASSENGER["Flight Distance"],
        departure_delay=PASSENGER["Departure Delay in Minutes"],
        arrival_delay=PASSENGER["Arrival Delay in Minutes"],
    )
    started = time.perf_counter()
    try:
        prediction = stub.Predict(
            message,
            metadata=(("x-api-key", API_KEY),),
            timeout=30,
        )
    finally:
        elapsed = time.perf_counter() - started
        channel.close()
    return prediction, elapsed


def main() -> None:
    rest, rest_seconds = call_rest()
    grpc_prediction, grpc_seconds = call_grpc()

    same = (
        rest["prediction"] == grpc_prediction.prediction
        and rest["label"] == grpc_prediction.label
        and rest["model_version"] == grpc_prediction.model_version
        and abs(rest["probability"] - grpc_prediction.probability) < 1e-12
    )

    print(f"rest_ms={rest_seconds * 1000:.2f}")
    print(f"grpc_ms={grpc_seconds * 1000:.2f}")
    print(
        "rest="
        f"{rest['label']} {rest['probability']:.6f} v{rest['model_version']}"
    )
    print(
        "grpc="
        f"{grpc_prediction.label} {grpc_prediction.probability:.6f} "
        f"v{grpc_prediction.model_version}"
    )
    print(f"coinciden={same}")
    if not same:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        print(exc.read().decode())
        raise
