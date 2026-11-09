"""Hybrid retrieval: owner-scoped vector search + keyword search, fused with reciprocal rank
fusion (RRF). Every returned chunk carries its true cosine score and how it was matched."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.core.logging import log_event, timed
from app.rag.embeddings import Embedder
from app.rag.keyword_search import KeywordIndex
from app.rag.vector_store import ScoredChunk, VectorStore

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 2000
_RRF_K = 60


@dataclass
class RetrievalResult:
    query: str
    chunks: list[ScoredChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def best_score(self) -> float | None:
        return max((c.score for c in self.chunks), default=None)


def fuse(semantic: list[ScoredChunk], keyword: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
    """Reciprocal rank fusion of two ranked lists (inputs must already be relevance-filtered)."""
    scores: dict[str, float] = {}
    by_id: dict[str, ScoredChunk] = {}
    matched: dict[str, set[str]] = {}
    for label, ranking in (("semantic", semantic), ("keyword", keyword)):
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
            by_id.setdefault(chunk.chunk_id, chunk)
            matched.setdefault(chunk.chunk_id, set()).add(label)
    ordered = sorted(scores, key=lambda cid: (-scores[cid], -by_id[cid].score))[:top_k]
    return [
        dataclasses.replace(by_id[cid], match="+".join(sorted(matched[cid], reverse=True)))
        for cid in ordered
    ]


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, default_top_k: int, min_score: float,
                 keyword_index: KeywordIndex | None = None):
        self.embedder = embedder
        self.store = store
        self.default_top_k = default_top_k
        self.min_score = min_score
        self.keywords = keyword_index or KeywordIndex()

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
        k = top_k or self.default_top_k
        with timed() as t_embed:
            vector = self.embedder.embed([query])[0]
        with timed() as t_vector:
            semantic = self.store.search(session, owner_id, vector, k, self.min_score, document_ids)
        with timed() as t_keyword:
            hits = self.keywords.search(session, owner_id, query, k, document_ids)
            known = {c.chunk_id for c in semantic}
            extra = {c.chunk_id: c for c in self.store.score_chunks(
                session, owner_id, vector, [h.chunk_id for h in hits if h.chunk_id not in known])}
            by_id = {c.chunk_id: c for c in semantic} | extra
            keyword = [by_id[h.chunk_id] for h in hits if h.chunk_id in by_id]
        chunks = fuse(semantic, keyword, k)
        timings = {"embed_ms": t_embed["ms"], "vector_ms": t_vector["ms"], "keyword_ms": t_keyword["ms"]}
        log_event(logger, "retrieval", hits=len(chunks), semantic_hits=len(semantic), keyword_hits=len(hits),
                  best_score=max((c.score for c in chunks), default=None), **timings)
        return RetrievalResult(query, chunks, timings)
