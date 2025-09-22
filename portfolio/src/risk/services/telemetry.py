from __future__ import annotations

import json
import logging
import logging.config
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator


_LOGGER_NAME = "portfolio.risk"


def setup_logging(config_path: Path | str | None) -> logging.Logger:
    """Configure and return the risk logger.

    If the configuration file cannot be loaded, a sane default configuration is
    applied so logging still functions during tests.
    """

    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    if config_path:
        try:
            import yaml

            data = yaml.safe_load(Path(config_path).read_text())  # type: ignore[no-redef]
            logging.config.dictConfig(data)
        except Exception:  # pragma: no cover - fallback path
            logging.basicConfig(level=logging.INFO)
            logger.warning("Failed to load logging config; using basicConfig")
    else:  # pragma: no cover - rarely used
        logging.basicConfig(level=logging.INFO)
    return logging.getLogger(_LOGGER_NAME)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload))


@contextmanager
def record_timing(logger: logging.Logger, metric: str, **fields: Any) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        log_event(logger, metric, duration_ms=round(duration * 1000, 3), **fields)

