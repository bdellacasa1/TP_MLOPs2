FROM apache/airflow:3.3.0-python3.12

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements/airflow.txt /tmp/requirements.txt

RUN pip install --no-cache-dir \
    "apache-airflow==3.3.0" \
    -r /tmp/requirements.txt

WORKDIR /workspace