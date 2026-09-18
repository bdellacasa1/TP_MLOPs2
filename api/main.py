from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
)

from api.model_service import model_service
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando API...")
    try:
        # Intentamos cargar el modelo
        model_service.reload_model()
        print("Modelo cargado exitosamente.")
    except Exception as e:
        # Si el modelo no existe, atrapamos el error pero NO apagamos la API
        print(f"Advertencia: No se encontró el modelo en MLflow. La API iniciará sin modelo. Detalle: {e}")
    
    yield
    
    print("Apagando API...")

app = FastAPI(
    title="Airline Satisfaction API",
    description=(
        "API de predicción de satisfacción "
        "de pasajeros."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():

    return {
        "message": (
            "Airline Satisfaction API"
        )
    }


@app.get("/health")
def health():

    return model_service.health()


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
):

    try:
        features = request.model_dump(
            by_alias=True
        )
        return model_service.predict(features)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="El modelo aún no está cargado. Ejecuta el pipeline en Airflow primero.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/reload-model")
def reload_model():
    """
    Recarga el modelo que actualmente tenga
    el alias champion.

    Airflow podrá llamar este endpoint
    después de un reentrenamiento exitoso.
    """

    try:

        return model_service.reload_model()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc