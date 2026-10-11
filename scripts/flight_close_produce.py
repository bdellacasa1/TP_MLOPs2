"""Publica el test en Redpanda, ordenado por demora de llegada.

Un topic, una partición. El orden de la partición es el orden de la ventana.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

from scripts.flight_close_rows import BOOTSTRAP, TOPIC, as_message, load_rows


def ensure_topic() -> None:
    admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP, client_id="flight-close")
    try:
        admin.create_topics(
            [NewTopic(name=TOPIC, num_partitions=1, replication_factor=1)]
        )
    except TopicAlreadyExistsError:
        pass
    finally:
        admin.close()

    probe = KafkaConsumer(bootstrap_servers=BOOTSTRAP)
    try:
        partitions = probe.partitions_for_topic(TOPIC)
    finally:
        probe.close()
    if not partitions:
        raise SystemExit(f"No se pudo crear el topic {TOPIC}.")
    if len(partitions) != 1:
        raise SystemExit(
            f"El topic {TOPIC} tiene {len(partitions)} particiones. "
            "Tiene que tener una, para no mezclar el orden de la demora."
        )


def main() -> None:
    rows = load_rows()
    ensure_topic()
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        acks="all",
        value_serializer=lambda value: json.dumps(value).encode(),
    )
    published = 0
    for _, row in rows.iterrows():
        producer.send(TOPIC, as_message(row))
        published += 1
        if published % 2000 == 0:
            producer.flush()
            print(f"publicados={published}", flush=True)
    producer.flush()
    producer.close()
    print(f"publicados={published} topic={TOPIC} particiones=1", flush=True)
    print(f"demora_nula_como_cero={rows.attrs['empty_arrival']}", flush=True)


if __name__ == "__main__":
    main()
