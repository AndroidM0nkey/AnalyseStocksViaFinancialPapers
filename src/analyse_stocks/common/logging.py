from __future__ import annotations

import logging


def configure_logging(service_name: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | {service_name} | %(levelname)s | %(message)s",
    )
