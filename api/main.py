import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.model_service import ArtifactChecksumError, model_service
from api.prediction_log import log_prediction
from api.schemas import PredictionRequest, PredictionResponse


PROTECTED_PATHS = {
    "/v1/predict",
    "/predict",
    "/v1/reload-model",
    "/reload-model",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando API...")
    try:
        model_service.reload_model()
        print("Modelo cargado exitosamente.")
    except Exception as exc:
        print(
            "Advertencia: No se encontró el modelo en MLflow. "
            f"La API iniciará sin modelo. Detalle: {exc}"
        )

    yield

    print("Apagando API...")


app = FastAPI(
    title="Airline Satisfaction API",
    description="API de predicción de satisfacción de pasajeros.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def require_api_key(request, call_next):
    if request.method == "POST" and request.url.path in PROTECTED_PATHS:
        expected = os.getenv("API_KEY", "")
        if not expected:
            return JSONResponse(
                status_code=500,
                content={"detail": "API_KEY no está configurada."},
            )
        provided = request.headers.get("x-api-key")
        if provided != expected:
            return JSONResponse(
                status_code=401,
                content={"detail": "API key inválida o ausente."},
            )
    return await call_next(request)


@app.get("/")
def root():
    return {"message": "Airline Satisfaction API"}


@app.get("/health")
def health():
    return model_service.health()


def _predict(request: PredictionRequest):
    try:
        features = request.model_dump(by_alias=True)
        result = model_service.predict(features)
        log_prediction(features, result)
        return result
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo aún no está cargado. "
                "Ejecuta el pipeline en Airflow primero."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/v1/predict", response_model=PredictionResponse)
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return _predict(request)


def _reload():
    try:
        return model_service.reload_model()
    except ArtifactChecksumError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/v1/reload-model")
@app.post("/reload-model")
def reload_model():
    """Recarga el modelo que actualmente tenga el alias champion."""

    return _reload()
