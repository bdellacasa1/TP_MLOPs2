FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Instalar dependencias necesarias
RUN pip install --upgrade pip && \
    pip install streamlit requests pandas

# Copiar la aplicación y la carpeta src (si es necesaria para imports compartidos)
# En este caso, streamlit_app.py es independiente pero necesita la ruta correcta
COPY src/frontend/streamlit_app.py ./app.py

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
