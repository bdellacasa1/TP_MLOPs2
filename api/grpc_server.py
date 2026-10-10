import os
from concurrent import futures

import grpc
from pydantic import ValidationError

from api import scoring_pb2, scoring_pb2_grpc
from api.model_service import model_service
from api.prediction_log import log_prediction
from api.schemas import PredictionRequest


def _metadata_api_key(context) -> str:
    for key, value in context.invocation_metadata():
        if key.lower() == "x-api-key":
            return value
    return ""


class ScoringService(scoring_pb2_grpc.ScoringServicer):
    def Predict(self, request, context):
        expected = os.getenv("API_KEY", "")
        if not expected:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "API_KEY no está configurada.",
            )
        if _metadata_api_key(context) != expected:
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                "API key inválida o ausente.",
            )

        try:
            payload = PredictionRequest.model_validate(
                {
                    "Gender": request.gender,
                    "Customer Type": request.customer_type,
                    "Type of Travel": request.type_of_travel,
                    "Class": request.travel_class,
                    "Age": request.age,
                    "Flight Distance": request.flight_distance,
                    "Departure Delay in Minutes": request.departure_delay,
                    "Arrival Delay in Minutes": request.arrival_delay,
                }
            )
        except ValidationError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        features = payload.model_dump(by_alias=True)
        try:
            result = model_service.predict(features)
        except RuntimeError as exc:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))

        log_prediction(features, result)
        return scoring_pb2.Prediction(
            prediction=result["prediction"],
            label=result["label"],
            probability=result["probability"],
            model_name=result["model_name"],
            model_alias=result["model_alias"],
            model_version=result["model_version"],
        )


def serve() -> None:
    print("Iniciando servicio gRPC...")
    try:
        model_service.reload_model()
        print("Modelo cargado exitosamente.")
    except Exception as exc:
        print(
            "Advertencia: No se encontró el modelo en MLflow. "
            f"El servicio iniciará sin modelo. Detalle: {exc}"
        )

    port = os.getenv("GRPC_PORT", "50051")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    scoring_pb2_grpc.add_ScoringServicer_to_server(ScoringService(), server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    print(f"gRPC escuchando en {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
