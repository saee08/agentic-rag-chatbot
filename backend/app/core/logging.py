"""Structured logging with request-ID propagation.

Log records never contain document text, prompts, passwords, tokens or API keys: callers pass
only identifiers, counts and timings via ``extra={"fields": {...}}``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = []
        request_id = request_id_var.get()
        if request_id:
            extras.append(f"request_id={request_id}")
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            extras.extend(f"{k}={v}" for k, v in fields.items())
        return f"{base} {' '.join(extras)}".rstrip()


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if json_output else TextFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # Third-party libraries are noisy at INFO and some log request URLs/headers at DEBUG.
    for noisy in ("httpx", "httpcore", "sentence_transformers", "urllib3", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, msg, extra={"fields": fields})


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    """Measure elapsed wall time in milliseconds: ``with timed() as t: ...; t["ms"]``."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["ms"] = round((time.perf_counter() - start) * 1000, 2)
