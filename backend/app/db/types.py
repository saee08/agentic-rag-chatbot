"""Portable column types.

``EmbeddingVector`` is a native pgvector ``vector(dim)`` on PostgreSQL (indexed, searched in
SQL) and a little-endian float32 BLOB elsewhere (SQLite, for local development and tests).
Both directions validate the dimension so a model/schema mismatch fails loudly.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import CHAR, LargeBinary
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator, TypeEngine


class EmbeddingVector(TypeDecorator[np.ndarray]):
    impl = LargeBinary
    cache_ok = True

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(LargeBinary())

    def _validate(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.dim:
            raise ValueError(f"embedding has dimension {arr.shape[0]}, column expects {self.dim}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("embedding contains NaN or infinite values")
        return arr

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        arr = self._validate(value)
        if dialect.name == "postgresql":
            return arr  # pgvector's bind processor serialises numpy arrays
        return arr.astype("<f4").tobytes()

    def process_result_value(self, value: Any, dialect: Dialect) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, bytes | bytearray | memoryview):
            return np.frombuffer(bytes(value), dtype="<f4").astype(np.float32)
        return np.asarray(value, dtype=np.float32)


class GUID(TypeDecorator[str]):
    """UUIDs stored as 36-char strings: portable and readable in both databases."""

    impl = CHAR(36)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> str | None:
        if value is None:
            return None
        try:
            return str(uuid.UUID(str(value)))
        except ValueError:
            # Malformed ids from URLs simply match no row (-> 404) instead of erroring.
            return str(value)[:36]

    def process_result_value(self, value: Any, dialect: Dialect) -> str | None:
        return value
