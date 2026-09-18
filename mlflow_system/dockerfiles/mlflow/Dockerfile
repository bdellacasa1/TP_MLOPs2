FROM python:3.12-slim

ARG MLFLOW_VERSION=3.14.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update -o Acquire::AllowInsecureRepositories=true \
    || true \
    && apt-get install -y --no-install-recommends --allow-unauthenticated \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

#RUN apt-get update \
#    && apt-get install -y --no-install-recommends curl \
#    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip \
    && sed "s/__MLFLOW_VERSION__/${MLFLOW_VERSION}/g" /tmp/requirements.txt > /tmp/requirements.lock \
    && python -m pip install -r /tmp/requirements.lock

EXPOSE 5000
