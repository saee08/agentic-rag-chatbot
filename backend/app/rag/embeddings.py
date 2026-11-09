"""Embedding providers.

One embedder instance is created per process (in the app lifespan) and shared by ingestion and
retrieval. The model is loaded lazily, exactly once, under a lock.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol, runtime_checkable

import numpy as np

from app.core.logging import log_event, timed

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    model_name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""
        ...

    @property
    def loaded(self) -> bool: ...


def validate_embeddings(vectors: np.ndarray, expected_rows: int, dim: int) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape != (expected_rows, dim):
        raise ValueError(f"embedding output has shape {arr.shape}, expected ({expected_rows}, {dim})")
    if not np.all(np.isfinite(arr)):
        raise ValueError("embedding output contains NaN or infinite values")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding output contains zero vectors")
    return arr / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, dim: int, batch_size: int = 32, device: str | None = None):
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from sentence_transformers import SentenceTransformer

            with timed() as t:
                model = SentenceTransformer(self.model_name, device=self.device)
            actual = model.get_embedding_dimension()
            if actual != self.dim:
                raise RuntimeError(
                    f"Embedding model {self.model_name} produces {actual}-d vectors but EMBEDDING_DIM="
                    f"{self.dim}. Update EMBEDDING_DIM (and re-run migrations on a fresh database)."
                )
            self._model = model
            log_event(logger, "embedding model loaded", model=self.model_name, dim=actual, load_ms=t["ms"])

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self.load()
        assert self._model is not None
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return validate_embeddings(vectors, len(texts), self.dim)
