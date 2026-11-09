"""Query -> embedding -> owner-scoped vector search, with timing."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.core.logging import log_event, timed
from app.rag.embeddings import Embedder
from app.rag.vector_store import ScoredChunk, VectorStore

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 2000


@dataclass
class RetrievalResult:
    query: str
    chunks: list[ScoredChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def best_score(self) -> float | None:
        return max((c.score for c in self.chunks), default=None)


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, default_top_k: int, min_score: float):
        self.embedder = embedder
        self.store = store
        self.default_top_k = default_top_k
        self.min_score = min_score

    def retrieve(
        self,
        session: Session,
        owner_id: str,
        query: str,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> RetrievalResult:
        query = " ".join(query.split())[:MAX_QUERY_CHARS]
        if not query:
            return RetrievalResult(query, [])
        with timed() as t_embed:
            vector = self.embedder.embed([query])[0]
        with timed() as t_search:
            chunks = self.store.search(
                session, owner_id, vector, top_k or self.default_top_k, self.min_score, document_ids
            )
        timings = {"embed_ms": t_embed["ms"], "search_ms": t_search["ms"]}
        log_event(logger, "retrieval", hits=len(chunks),
                  best_score=max((c.score for c in chunks), default=None), **timings)
        return RetrievalResult(query, chunks, timings)
