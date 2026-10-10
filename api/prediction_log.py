import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path


_lock = threading.Lock()


def prediction_log_path() -> Path:
    configured = os.getenv(
        "PREDICTION_LOG_PATH",
        "/var/log/airline/predictions.jsonl",
    )
    return Path(configured)


def log_prediction(features: dict, result: dict) -> None:
    """Append one successful prediction. Invalid requests are not logged."""

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "prediction": result["prediction"],
        "label": result["label"],
        "probability": result["probability"],
        "model_name": result["model_name"],
        "model_alias": result["model_alias"],
        "model_version": result["model_version"],
    }
    path = prediction_log_path()
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
